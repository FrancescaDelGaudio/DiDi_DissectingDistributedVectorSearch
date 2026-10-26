#pragma once

// Standalone RobustPrune (DiskANN) operating on global ids via a vector
// accessor. Used by Step 3 (stitching) to prune the union of neighbour lists
// of a vector that appears in several clusters.
//
// RobustPrune(p, V, alpha, R):
//   V <- (V U N_out(p)) \ {p}
//   N_out(p) <- {}
//   while V != empty:
//     p* <- argmin_{p' in V} dist(p, p')
//     N_out(p) <- N_out(p) U {p*}
//     if |N_out(p)| = R: break
//     for p' in V: if alpha * dist(p*, p') <= dist(p, p'): remove p' from V

#include <vector>
#include <algorithm>
#include <cstdint>

#include "../common/distance.hpp"
#include "../common/metric.hpp"

namespace dann {

// VecFn: uint32_t global_id -> const float* (vector of length dim).
template <class VecFn>
inline std::vector<uint32_t>
robust_prune_global(uint32_t p,
                    std::vector<uint32_t>& candidates,
                    VecFn vec, int dim,
                    int R, float alpha,
                    Metric metric = Metric::L2)
{
    // Dedup and drop p itself.
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    candidates.erase(std::remove(candidates.begin(), candidates.end(), p),
                     candidates.end());
    if (candidates.empty()) return {};

    const float* vp = vec(p);

    // (dist to p, id) ascending.
    std::vector<std::pair<float, uint32_t>> cand;
    cand.reserve(candidates.size());
    for (uint32_t c : candidates)
        cand.push_back({dist_score(metric, vp, vec(c), dim), c});
    std::sort(cand.begin(), cand.end());

    std::vector<uint32_t> result;
    result.reserve(R);
    std::vector<char> removed(cand.size(), 0);

    for (size_t i = 0; i < cand.size(); i++) {
        if (removed[i]) continue;
        uint32_t pstar = cand[i].second;
        result.push_back(pstar);
        if ((int)result.size() >= R) break;

        const float* vstar = vec(pstar);
        for (size_t j = i + 1; j < cand.size(); j++) {
            if (removed[j]) continue;
            float d_star_j = dist_score(metric, vstar, vec(cand[j].second), dim);
            if (alpha * d_star_j <= cand[j].first)
                removed[j] = 1;
        }
    }
    return result;
}

} // namespace dann
