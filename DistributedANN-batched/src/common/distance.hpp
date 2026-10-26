#pragma once

#include <cmath>
#include <limits>
#include <utility>
#include "types.hpp"
#include "metric.hpp"

namespace dann {

// Squared L2 distance between two vectors of length dim.
// Hot path: -O3 -march=native lets the compiler auto-vectorise this.
inline float l2_sq(const float* __restrict__ a,
                   const float* __restrict__ b,
                   int dim) noexcept
{
    float s = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// L2 distance (not squared).
inline float l2(const float* a, const float* b, int dim) noexcept {
    return std::sqrt(l2_sq(a, b, dim));
}

// Returns {index_of_nearest_centroid, score} where lower score is better.
// L2: squared distance; IP: -dot (vectors should be unit-normalised).
inline std::pair<int, float> nearest_centroid(const float* v,
                                               const float* centroids,
                                               int K,
                                               int dim,
                                               Metric metric = Metric::L2) noexcept
{
    float best_d = F_INF;
    int   best_k = 0;
    for (int k = 0; k < K; k++) {
        float d = dist_score(metric, v, centroids + (size_t)k * dim, dim);
        if (d < best_d) { best_d = d; best_k = k; }
    }
    return {best_k, best_d};
}

} // namespace dann
