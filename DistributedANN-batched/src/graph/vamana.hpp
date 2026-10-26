#pragma once

// Vamana graph construction — faithful to DiskANN (Subramanya et al., 2019)
// and the microsoft/DiskANN reference implementation.
//
// Algorithm (per the paper):
//   1. Initialise G as a random R-regular directed graph.
//   2. Compute the medoid s (point closest to the dataset centroid); it is the
//      fixed entry point for GreedySearch.
//   3. For two passes (alpha = 1.0 then alpha = user value), in a random order,
//      for each point p:
//        a. V <- GreedySearch(s, x_p, 1, L).visited
//        b. N_out(p) <- RobustPrune(p, V, alpha, R)
//        c. for each j in N_out(p): add edge j -> p, and if |N_out(j)| > R,
//           N_out(j) <- RobustPrune(j, N_out(j) U {p}, alpha, R)
//
// Storage during build: fixed-capacity adjacency (n * R) + per-node degree,
// which avoids reallocation so concurrent readers in the parallel build never
// crash (mirrors DiskANN's lock-per-node strategy). Neighbours are stored as
// LOCAL indices [0, n); callers translate to global ids at serialisation time.

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <unordered_set>

#include <omp.h>

#include "../common/distance.hpp"
#include "../common/metric.hpp"

namespace dann {

struct VamanaParams {
    int   R       = 72;     // max out-degree (paper default for DistributedANN)
    int   L       = 100;    // build-time search list size
    float alpha   = 1.2f;   // RobustPrune slack (DiskANN default)
    int   passes  = 2;      // pass 0 uses alpha=1.0, pass 1 uses alpha
    int   threads = 0;      // 0 -> use OpenMP default / OMP_NUM_THREADS
    unsigned seed = 42;
};

class Vamana {
public:
    // data: n * dim, row-major (cluster-local vectors). Not copied; must
    // outlive the build call.
    Vamana(const float* data, int n, int dim, const VamanaParams& p,
           Metric metric = Metric::L2)
        : data_(data), n_(n), dim_(dim), p_(p), metric_(metric),
          R_(std::min(p.R, std::max(1, n - 1))) {}

    // Build the graph. Returns nothing; results accessible via accessors.
    void build();

    uint32_t medoid()  const { return medoid_; }
    int      n()       const { return n_; }
    int      R()       const { return R_; }

    // Out-neighbours (local indices) of node u.
    const uint32_t* neighbors(uint32_t u) const { return &graph_[(size_t)u * R_]; }
    uint32_t        degree(uint32_t u)    const { return degree_[u]; }

private:
    const float* data_;
    int n_, dim_;
    VamanaParams p_;
    Metric metric_;
    int R_;

    std::vector<uint32_t> graph_;    // n * R_  (fixed capacity)
    std::vector<uint32_t> degree_;   // n
    std::vector<std::mutex> locks_;  // n
    uint32_t medoid_ = 0;

    const float* vec(uint32_t u) const { return data_ + (size_t)u * dim_; }
    float dist(uint32_t a, uint32_t b) const {
        return dist_score(metric_, vec(a), vec(b), dim_);
    }

    void     compute_medoid();
    void     random_init();
    std::vector<uint32_t> greedy_search(uint32_t query, int L) const;
    std::vector<uint32_t> robust_prune(uint32_t p,
                                       std::vector<uint32_t>& candidates,
                                       float alpha) const;
    void     run_pass(float alpha);
};

// ---------------------------------------------------------------------------
// Medoid: vector closest (L2) to the arithmetic mean of all vectors.
// ---------------------------------------------------------------------------
inline void Vamana::compute_medoid()
{
    std::vector<double> mean((size_t)dim_, 0.0);
    for (int i = 0; i < n_; i++) {
        const float* v = vec((uint32_t)i);
        for (int d = 0; d < dim_; d++) mean[d] += v[d];
    }
    for (int d = 0; d < dim_; d++) mean[d] /= (double)n_;

    std::vector<float> mean_f(mean.begin(), mean.end());
    float best = F_INF;
    uint32_t best_i = 0;
    if (metric_ == Metric::IP) normalize_inplace(mean_f.data(), dim_);
    for (int i = 0; i < n_; i++) {
        float dd = dist_score(metric_, vec((uint32_t)i), mean_f.data(), dim_);
        if (dd < best) { best = dd; best_i = (uint32_t)i; }
    }
    medoid_ = best_i;
}

// ---------------------------------------------------------------------------
// Random R-regular initialisation.
// ---------------------------------------------------------------------------
inline void Vamana::random_init()
{
    graph_.assign((size_t)n_ * R_, 0);
    degree_.assign((size_t)n_, 0);

    #pragma omp parallel
    {
        std::mt19937 rng(p_.seed + 1469u * (unsigned)omp_get_thread_num());
        std::uniform_int_distribution<int> pick(0, n_ - 1);

        #pragma omp for schedule(static)
        for (int u = 0; u < n_; u++) {
            std::unordered_set<uint32_t> chosen;
            chosen.reserve(R_ * 2);
            while ((int)chosen.size() < R_ && (int)chosen.size() < n_ - 1) {
                int v = pick(rng);
                if (v != u) chosen.insert((uint32_t)v);
            }
            uint32_t d = 0;
            for (uint32_t v : chosen) graph_[(size_t)u * R_ + d++] = v;
            degree_[u] = d;
        }
    }
}

// ---------------------------------------------------------------------------
// GreedySearch from the medoid towards `query`. Returns the set of *visited*
// (expanded) nodes — the candidate pool for RobustPrune.
// ---------------------------------------------------------------------------
inline std::vector<uint32_t> Vamana::greedy_search(uint32_t query, int L) const
{
    struct Cand { float d; uint32_t id; bool visited; };

    std::vector<Cand> pool;
    pool.reserve(L + 1);
    std::unordered_set<uint32_t> in_pool;
    in_pool.reserve(L * 4);

    std::vector<uint32_t> visited;
    visited.reserve(L * 2);

    auto insert_cand = [&](uint32_t id) {
        if (in_pool.count(id)) return;
        float d = dist(query, id);
        if ((int)pool.size() >= L && d >= pool.back().d) return;
        Cand c{d, id, false};
        auto it = std::lower_bound(pool.begin(), pool.end(), c,
                    [](const Cand& a, const Cand& b){ return a.d < b.d; });
        pool.insert(it, c);
        in_pool.insert(id);
        if ((int)pool.size() > L) {
            in_pool.erase(pool.back().id);
            pool.pop_back();
        }
    };

    insert_cand(medoid_);

    while (true) {
        // Find closest unvisited candidate.
        int idx = -1;
        for (int i = 0; i < (int)pool.size(); i++)
            if (!pool[i].visited) { idx = i; break; }
        if (idx < 0) break;

        pool[idx].visited = true;
        uint32_t cur = pool[idx].id;
        visited.push_back(cur);

        uint32_t d = degree_[cur];
        if (d > (uint32_t)R_) d = (uint32_t)R_;   // tolerate concurrent writes
        const uint32_t* nbrs = &graph_[(size_t)cur * R_];
        for (uint32_t k = 0; k < d; k++) {
            uint32_t nb = nbrs[k];
            if (nb != query) insert_cand(nb);
        }
    }
    return visited;
}

// ---------------------------------------------------------------------------
// RobustPrune(p, candidates, alpha): select <= R out-neighbours for p.
// ---------------------------------------------------------------------------
inline std::vector<uint32_t>
Vamana::robust_prune(uint32_t p, std::vector<uint32_t>& candidates, float alpha) const
{
    // Deduplicate and drop p itself.
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    candidates.erase(std::remove(candidates.begin(), candidates.end(), p),
                     candidates.end());

    // (dist to p, id), sorted ascending.
    std::vector<std::pair<float, uint32_t>> cand;
    cand.reserve(candidates.size());
    for (uint32_t c : candidates) cand.push_back({dist(p, c), c});
    std::sort(cand.begin(), cand.end());

    std::vector<uint32_t> result;
    result.reserve(R_);
    std::vector<char> removed(cand.size(), 0);

    for (size_t i = 0; i < cand.size(); i++) {
        if (removed[i]) continue;
        uint32_t pstar = cand[i].second;
        result.push_back(pstar);
        if ((int)result.size() >= R_) break;

        // Occlusion rule: remove p' dominated by p*.
        for (size_t j = i + 1; j < cand.size(); j++) {
            if (removed[j]) continue;
            float d_pstar_pp = dist(pstar, cand[j].second);
            if (alpha * d_pstar_pp <= cand[j].first)
                removed[j] = 1;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// One construction pass over all points in random order.
// ---------------------------------------------------------------------------
inline void Vamana::run_pass(float alpha)
{
    // Random visit order.
    std::vector<uint32_t> order((size_t)n_);
    for (int i = 0; i < n_; i++) order[i] = (uint32_t)i;
    std::shuffle(order.begin(), order.end(), std::mt19937(p_.seed + 7u));

    #pragma omp parallel for schedule(dynamic, 256)
    for (int oi = 0; oi < n_; oi++) {
        uint32_t p = order[oi];

        // (a) GreedySearch -> candidate pool
        std::vector<uint32_t> V = greedy_search(p, p_.L);

        // (b) RobustPrune p over (V U current neighbours)
        {
            std::lock_guard<std::mutex> lk(locks_[p]);
            uint32_t d = degree_[p];
            for (uint32_t k = 0; k < d; k++) V.push_back(graph_[(size_t)p * R_ + k]);
        }
        std::vector<uint32_t> newN = robust_prune(p, V, alpha);

        // Commit p's neighbours.
        {
            std::lock_guard<std::mutex> lk(locks_[p]);
            uint32_t d = 0;
            for (uint32_t nb : newN) graph_[(size_t)p * R_ + d++] = nb;
            degree_[p] = d;
        }

        // (c) Add back-edges j -> p, pruning j if it overflows.
        for (uint32_t j : newN) {
            std::lock_guard<std::mutex> lk(locks_[j]);
            uint32_t dj = degree_[j];

            // Skip if p already present.
            bool present = false;
            for (uint32_t k = 0; k < dj; k++)
                if (graph_[(size_t)j * R_ + k] == p) { present = true; break; }
            if (present) continue;

            if ((int)dj < R_) {
                graph_[(size_t)j * R_ + dj] = p;
                degree_[j] = dj + 1;
            } else {
                std::vector<uint32_t> cand;
                cand.reserve(dj + 1);
                for (uint32_t k = 0; k < dj; k++)
                    cand.push_back(graph_[(size_t)j * R_ + k]);
                cand.push_back(p);
                std::vector<uint32_t> pruned = robust_prune(j, cand, alpha);
                uint32_t d = 0;
                for (uint32_t nb : pruned) graph_[(size_t)j * R_ + d++] = nb;
                degree_[j] = d;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full build.
// ---------------------------------------------------------------------------
inline void Vamana::build()
{
    if (p_.threads > 0) omp_set_num_threads(p_.threads);

    locks_ = std::vector<std::mutex>((size_t)n_);

    if (n_ <= 1) {            // trivial graph
        graph_.assign((size_t)n_ * std::max(1, R_), 0);
        degree_.assign((size_t)n_, 0);
        medoid_ = 0;
        return;
    }

    compute_medoid();
    random_init();

    for (int pass = 0; pass < p_.passes; pass++) {
        float a = (pass == 0) ? 1.0f : p_.alpha;
        run_pass(a);
    }
}

} // namespace dann
