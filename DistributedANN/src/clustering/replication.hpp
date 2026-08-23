#pragma once

// SPANN-style ε-replication for graph stitching (§3 of the DistributedANN paper).
//
// For each vector v with nearest centroid c₁ at score s₁ (lower is better), v is
// replicated into cluster c' if:
//   (1) s' is within the ε-neighbourhood of s₁ (metric-specific; see below)
//   (2) MRNG condition holds for c' given already-accepted centroids A
//   (3) |A| < max_replicas
//
// MRNG condition (Monotone Relative Neighbor Graph):
//   c' is skipped if ∃ a ∈ A  s.t.  score(centroid[a], centroid[c']) < score(v, centroid[c'])
//
// Metric behaviour:
//   L2  — score = Euclidean distance; ε-ball: dist ≤ (1+ε)·dist_nearest (unchanged).
//   IP  — unit vectors; score = -dot (via dist_score); ε-ball on the sphere:
//         dot(v,c') ≥ dot(v,c₁) − ε·(1 − dot(v,c₁)), equivalent to chord-squared
//         dist ≤ (1+ε)·chord_sq_nearest.  Matches spherical k-means / query scoring.
//
// Pyramid (Distributed-HNSW) MIPS uses centroid-centric top-r replication:
// for each centroid c, pull the r dataset vectors with highest IP to c.
// See top_r_mips_replicate_mpi() below.

#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>
#include <limits>

#include "../common/types.hpp"
#include "../common/distance.hpp"
#include "../common/metric.hpp"

namespace dann {

// Point–centroid score: lower is better (L2 distance or -dot for IP).
inline float replication_score(Metric metric,
                               const float* a,
                               const float* b,
                               int dim) noexcept
{
    if (metric == Metric::L2)
        return l2(a, b, dim);
    return dist_score(metric, a, b, dim);
}

// ε-ball membership for a candidate centroid score.
inline bool within_epsilon_ball(Metric metric,
                                float score,
                                float nearest_score,
                                float epsilon) noexcept
{
    if (metric == Metric::L2) {
        return score <= (1.0f + epsilon) * nearest_score;
    }
    // IP on unit sphere: dot' >= dot₁ − ε·(1 − dot₁).
    const float dot_nearest = -nearest_score;
    const float dot_cand    = -score;
    return dot_cand >= dot_nearest - epsilon * (1.0f - dot_nearest);
}

// Precompute K×K replication scores between centroids.
inline std::vector<float> centroid_score_matrix(Metric metric,
                                                const float* centroids,
                                                int K,
                                                int dim)
{
    std::vector<float> mat((size_t)K * K, 0.0f);
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            if (i != j) {
                mat[(size_t)i * K + j] =
                    replication_score(metric,
                                      centroids + (size_t)i * dim,
                                      centroids + (size_t)j * dim,
                                      dim);
            }
        }
    }
    return mat;
}

// ---------------------------------------------------------------------------
// Compute ε-replication assignments for local_n vectors.
//
// Returns assignments[i] = sorted list of cluster ids (nearest first).
// Vectors should already be L2-normalised when metric == IP (build_index does this).
// ---------------------------------------------------------------------------
inline std::vector<std::vector<clust_t>>
epsilon_replicate(const float* local_data,
                  int          local_n,
                  int          dim,
                  const float* centroids,
                  int          K,
                  float        epsilon,
                  int          max_replicas,
                  Metric       metric = Metric::L2)
{
    auto cscore = centroid_score_matrix(metric, centroids, K, dim);

    std::vector<std::vector<clust_t>> result((size_t)local_n);
    std::vector<std::pair<float, int>> scores((size_t)K);

    for (int i = 0; i < local_n; i++) {
        const float* v = local_data + (size_t)i * dim;

        for (int k = 0; k < K; k++)
            scores[k] = {replication_score(metric, v, centroids + (size_t)k * dim, dim), k};

        std::sort(scores.begin(), scores.end());

        const float nearest_score = scores[0].first;

        std::vector<clust_t> assigned;
        assigned.reserve(max_replicas);
        assigned.push_back((clust_t)scores[0].second);

        for (int j = 1; j < K && (int)assigned.size() < max_replicas; j++) {
            const float dv_c = scores[j].first;
            if (!within_epsilon_ball(metric, dv_c, nearest_score, epsilon))
                break;

            const int c = scores[j].second;

            bool dominated = false;
            for (clust_t a : assigned) {
                if (cscore[(size_t)a * K + c] < dv_c) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated)
                assigned.push_back((clust_t)c);
        }

        result[i] = std::move(assigned);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Primary assignment only (nearest centroid).  Vectors must be normalised for IP.
// ---------------------------------------------------------------------------
inline std::vector<std::vector<clust_t>>
primary_assign(const float* local_data,
               int          local_n,
               int          dim,
               const float* centroids,
               int          K,
               Metric       metric = Metric::L2)
{
    std::vector<std::vector<clust_t>> result((size_t)local_n);
    for (int i = 0; i < local_n; i++) {
        int c = nearest_centroid(local_data + (size_t)i * dim,
                                 centroids, K, dim, metric).first;
        result[i] = {(clust_t)c};
    }
    return result;
}

// Keep top `r` pairs by IP score (higher dot = better).
inline void keep_top_r_by_score(std::vector<std::pair<float, idx_t>>& acc, int r)
{
    if (r <= 0 || acc.empty()) return;
    if ((int)acc.size() <= r) {
        std::sort(acc.begin(), acc.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        return;
    }
    std::nth_element(acc.begin(), acc.begin() + r, acc.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    acc.resize((size_t)r);
    std::sort(acc.begin(), acc.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
}

// ---------------------------------------------------------------------------
// Pyramid-style MIPS replication (distributed).
//
// 1. Primary: nearest centroid per vector (spherical k-means assignment).
// 2. For each centroid c, globally find top replica_r vectors by dot(v, c).
// 3. Union primary + replica clusters per vector.
//
// local_data must be unit-normalised when used for IP/MIPS.
// ---------------------------------------------------------------------------
inline std::vector<std::vector<clust_t>>
top_r_mips_replicate_mpi(const float* local_data,
                         int          local_n,
                         idx_t        rank_start,
                         int          dim,
                         const float* centroids,
                         int          K,
                         int          replica_r,
                         MPI_Comm     comm)
{
    int rank = 0, nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    if (replica_r <= 0)
        return primary_assign(local_data, local_n, dim, centroids, K, Metric::IP);

    auto result = primary_assign(local_data, local_n, dim, centroids, K, Metric::IP);

    const int slot = replica_r;
    const int pkt  = 2;   // (score, global_id) per slot
    const int row  = slot * pkt;

    std::vector<float> sendbuf((size_t)K * row, -std::numeric_limits<float>::infinity());
    std::vector<float> recvbuf;

    for (int c = 0; c < K; c++) {
        const float* cent = centroids + (size_t)c * dim;
        std::vector<std::pair<float, idx_t>> local_top;
        local_top.reserve((size_t)local_n);

        for (int i = 0; i < local_n; i++) {
            float s = dot(local_data + (size_t)i * dim, cent, dim);
            local_top.emplace_back(s, rank_start + (idx_t)i);
        }

        int keep = std::min(replica_r, (int)local_top.size());
        if (keep > 0) {
            if (keep < (int)local_top.size()) {
                std::nth_element(local_top.begin(), local_top.begin() + keep, local_top.end(),
                                 [](const auto& a, const auto& b) { return a.first > b.first; });
            }
            local_top.resize((size_t)keep);
            std::sort(local_top.begin(), local_top.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
        } else {
            local_top.clear();
        }

        float* row_out = sendbuf.data() + (size_t)c * row;
        for (int j = 0; j < replica_r; j++) {
            if (j < (int)local_top.size()) {
                row_out[j * pkt + 0] = local_top[(size_t)j].first;
                row_out[j * pkt + 1] = (float)local_top[(size_t)j].second;
            }
        }
    }

    if (rank == 0) recvbuf.resize((size_t)nranks * K * row);
    MPI_Gather(sendbuf.data(), K * row, MPI_FLOAT,
               rank == 0 ? recvbuf.data() : nullptr, K * row, MPI_FLOAT,
               0, comm);

    std::vector<float> global_top((size_t)K * row, -std::numeric_limits<float>::infinity());

    if (rank == 0) {
        for (int c = 0; c < K; c++) {
            std::vector<std::pair<float, idx_t>> merged;
            merged.reserve((size_t)nranks * replica_r);
            for (int r = 0; r < nranks; r++) {
                const float* src = recvbuf.data() + ((size_t)r * K + c) * row;
                for (int j = 0; j < replica_r; j++) {
                    float score = src[j * pkt + 0];
                    if (!std::isfinite(score)) continue;
                    merged.emplace_back(score, (idx_t)src[j * pkt + 1]);
                }
            }
            keep_top_r_by_score(merged, replica_r);

            float* dst = global_top.data() + (size_t)c * row;
            for (int j = 0; j < replica_r; j++) {
                if (j < (int)merged.size()) {
                    dst[j * pkt + 0] = merged[(size_t)j].first;
                    dst[j * pkt + 1] = (float)merged[(size_t)j].second;
                }
            }
        }
    }

    MPI_Bcast(global_top.data(), K * row, MPI_FLOAT, 0, comm);

    for (int c = 0; c < K; c++) {
        const float* src = global_top.data() + (size_t)c * row;
        for (int j = 0; j < replica_r; j++) {
            float score = src[j * pkt + 0];
            if (!std::isfinite(score)) continue;
            idx_t gid = (idx_t)src[j * pkt + 1];
            if (gid < rank_start || gid >= rank_start + (idx_t)local_n) continue;

            idx_t li = gid - rank_start;
            auto& clusters = result[(size_t)li];
            clust_t cc = (clust_t)c;
            if (std::find(clusters.begin(), clusters.end(), cc) == clusters.end())
                clusters.push_back(cc);
        }
    }

    return result;
}

} // namespace dann
