#pragma once

// Distributed k-means (Lloyd's algorithm) over MPI.
//
// Data partitioning: each MPI rank owns a contiguous slice of the dataset.
// Centroids are computed globally via MPI_Allreduce and are replicated on
// every rank after each iteration.
//
// Initialisation: k-means++ on a gathered sample (at most SAMPLE_PER_RANK *
// nranks vectors), run entirely on rank 0 and then broadcast.

#include <mpi.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>
#include <cstdio>
#include <cassert>

#include "../common/types.hpp"
#include "../common/distance.hpp"
#include "../common/metric.hpp"

namespace dann {

// ---------------------------------------------------------------------------
// k-means++ initialisation (sequential, on a gathered sample).
// Returns centroids as flat row-major vector: K * dim.
// ---------------------------------------------------------------------------
inline std::vector<float> kmeanspp_init(const float* sample,
                                         int n_sample,
                                         int dim,
                                         int K,
                                         unsigned seed = 42)
{
    assert(n_sample >= K);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int>  pick_int(0, n_sample - 1);
    std::uniform_real_distribution<float> pick_real(0.0f, 1.0f);

    std::vector<int>   chosen;  chosen.reserve(K);
    std::vector<float> min_d2(n_sample, F_INF);

    chosen.push_back(pick_int(rng));

    for (int k = 1; k < K; k++) {
        // Update min squared distances using the last chosen centroid.
        const float* last = sample + (size_t)chosen.back() * dim;
        for (int i = 0; i < n_sample; i++) {
            float d2 = l2_sq(sample + (size_t)i * dim, last, dim);
            if (d2 < min_d2[i]) min_d2[i] = d2;
        }

        // Sample next centroid with probability proportional to D^2.
        float total = std::accumulate(min_d2.begin(), min_d2.end(), 0.0f);
        float r     = pick_real(rng) * total;
        float cum   = 0.0f;
        int   next  = n_sample - 1;   // fallback: last element
        for (int i = 0; i < n_sample; i++) {
            cum += min_d2[i];
            if (cum >= r) { next = i; break; }
        }
        chosen.push_back(next);
    }

    std::vector<float> centroids((size_t)K * dim);
    for (int k = 0; k < K; k++)
        std::copy(sample + (size_t)chosen[k] * dim,
                  sample + (size_t)chosen[k] * dim + dim,
                  centroids.data() + (size_t)k * dim);
    return centroids;
}

// ---------------------------------------------------------------------------
// Result returned by kmeans_mpi.
// ---------------------------------------------------------------------------
struct KMeansResult {
    std::vector<float> centroids;   // K * dim, row-major; replicated on all ranks
    int   K;
    int   dim;
    int   iters_run;
    float final_displacement;       // max centroid shift in the last iteration (L2)
};

// ---------------------------------------------------------------------------
// Distributed k-means.
//
// local_data : local_n * dim, row-major (rank's slice of the dataset)
// local_n    : number of vectors owned by this rank
// dim        : vector dimension (must be identical on all ranks)
// K          : number of clusters
// max_iter   : maximum Lloyd's iterations
// tol        : stop when max centroid displacement (L2) < tol
// comm       : MPI communicator
//
// All ranks return the same KMeansResult (centroids replicated everywhere).
// ---------------------------------------------------------------------------
inline KMeansResult kmeans_mpi(const float* local_data,
                                int          local_n,
                                int          dim,
                                int          K,
                                int          max_iter = 20,
                                float        tol      = 1e-4f,
                                MPI_Comm     comm     = MPI_COMM_WORLD,
                                Metric       metric   = Metric::L2)
{
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    // ------------------------------------------------------------------
    // Initialisation: each rank contributes up to SAMPLE_PER_RANK evenly-
    // spaced vectors; rank 0 gathers them, runs k-means++, broadcasts.
    // ------------------------------------------------------------------
    const int SAMPLE_PER_RANK = std::min(2000, local_n);
    const int total_sample    = SAMPLE_PER_RANK * nranks;

    // Build local sample (evenly strided through local_data)
    std::vector<float> local_sample((size_t)SAMPLE_PER_RANK * dim);
    for (int i = 0; i < SAMPLE_PER_RANK; i++) {
        int idx = (int)((long long)i * local_n / SAMPLE_PER_RANK);
        std::copy(local_data + (size_t)idx * dim,
                  local_data + (size_t)idx * dim + dim,
                  local_sample.data() + (size_t)i * dim);
    }

    std::vector<float> centroids((size_t)K * dim);

    if (rank == 0) {
        std::vector<float> all_samples((size_t)total_sample * dim);
        MPI_Gather(local_sample.data(), SAMPLE_PER_RANK * dim, MPI_FLOAT,
                   all_samples.data(), SAMPLE_PER_RANK * dim, MPI_FLOAT,
                   0, comm);
        centroids = kmeanspp_init(all_samples.data(), total_sample, dim, K);
        printf("[rank 0] k-means++ initialisation complete (sample size=%d)\n",
               total_sample);
        fflush(stdout);
    } else {
        MPI_Gather(local_sample.data(), SAMPLE_PER_RANK * dim, MPI_FLOAT,
                   nullptr,             SAMPLE_PER_RANK * dim, MPI_FLOAT,
                   0, comm);
    }
    MPI_Bcast(centroids.data(), K * dim, MPI_FLOAT, 0, comm);
    if (metric == Metric::IP) {
        for (int k = 0; k < K; k++)
            normalize_inplace(centroids.data() + (size_t)k * dim, dim);
    }

    // ------------------------------------------------------------------
    // Lloyd's iterations
    // ------------------------------------------------------------------
    // Use double-precision accumulators to avoid large rounding errors
    // when summing ~N/nranks vectors per centroid per rank.
    std::vector<double>   partial_sums((size_t)K * dim);
    std::vector<uint64_t> partial_counts((size_t)K);
    std::vector<double>   global_sums((size_t)K * dim);
    std::vector<uint64_t> global_counts((size_t)K);
    std::vector<float>    old_centroids((size_t)K * dim);

    int   iters_run          = 0;
    float final_displacement = F_INF;

    for (int iter = 0; iter < max_iter; iter++) {
        std::fill(partial_sums.begin(),   partial_sums.end(),   0.0);
        std::fill(partial_counts.begin(), partial_counts.end(), 0ULL);

        // Assignment + local accumulation
        for (int i = 0; i < local_n; i++) {
            int k = nearest_centroid(local_data + (size_t)i * dim,
                                     centroids.data(), K, dim, metric).first;
            const float* v = local_data + (size_t)i * dim;
            double* s      = partial_sums.data() + (size_t)k * dim;
            for (int d = 0; d < dim; d++) s[d] += v[d];
            partial_counts[k]++;
        }

        // Global reduce: sums and counts
        MPI_Allreduce(partial_sums.data(),   global_sums.data(),   K * dim, MPI_DOUBLE,   MPI_SUM, comm);
        MPI_Allreduce(partial_counts.data(), global_counts.data(), K,       MPI_UINT64_T, MPI_SUM, comm);

        // Update centroids (all ranks compute identically → no extra broadcast)
        old_centroids = centroids;
        for (int k = 0; k < K; k++) {
            if (global_counts[k] == 0) continue;   // empty cluster: keep centroid
            double inv = 1.0 / (double)global_counts[k];
            float* c   = centroids.data() + (size_t)k * dim;
            const double* gs = global_sums.data() + (size_t)k * dim;
            for (int d = 0; d < dim; d++) c[d] = (float)(gs[d] * inv);
            if (metric == Metric::IP) normalize_inplace(c, dim);
        }

        // Convergence: max centroid displacement (all ranks compute identically)
        float max_disp = 0.0f;
        for (int k = 0; k < K; k++)
            max_disp = std::max(max_disp,
                                l2(centroids.data() + (size_t)k * dim,
                                   old_centroids.data() + (size_t)k * dim,
                                   dim));

        iters_run          = iter + 1;
        final_displacement = max_disp;

        if (rank == 0) {
            printf("[rank 0]  k-means iter %2d/%d  max_disp=%.6f\n",
                   iters_run, max_iter, max_disp);
            fflush(stdout);
        }

        if (max_disp < tol) {
            if (rank == 0) printf("[rank 0] Converged.\n");
            break;
        }
    }

    return KMeansResult{std::move(centroids), K, dim, iters_run, final_displacement};
}

} // namespace dann
