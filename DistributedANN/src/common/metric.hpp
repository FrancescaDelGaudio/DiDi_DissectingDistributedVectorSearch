#pragma once

// Distance metric for indexing and query (L2 vs inner-product / MIPS).
//
// For MIPS we L2-normalise vectors at ingest. All graph/PQ scoring uses
// dist_score() where lower is better (IP -> -dot, compatible with min-heaps).

#include <cmath>
#include <cstring>
#include <string>
#include <stdexcept>

#include <omp.h>

#include "types.hpp"

namespace dann {

enum class Metric { L2, IP };

inline Metric parse_metric(const std::string& s) {
    if (s == "l2" || s == "L2") return Metric::L2;
    if (s == "ip" || s == "IP" || s == "mips" || s == "MIPS") return Metric::IP;
    throw std::runtime_error("Unknown metric: " + s + " (expected l2 or ip)");
}

inline const char* metric_name(Metric m) {
    return m == Metric::IP ? "ip" : "l2";
}

inline void normalize_inplace(float* v, int dim) {
    float norm_sq = 0.0f;
    for (int j = 0; j < dim; ++j) norm_sq += v[j] * v[j];
    if (norm_sq <= 0.0f) return;
    float inv = 1.0f / std::sqrt(norm_sq);
    for (int j = 0; j < dim; ++j) v[j] *= inv;
}

inline float dot(const float* a, const float* b, int dim) noexcept {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
}

// Squared L2 or negated dot — lower is better (min-heap / Vamana / PQ ADC).
inline float dist_score(Metric m, const float* a, const float* b, int dim) noexcept {
    if (m == Metric::L2) {
        float s = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }
    return -dot(a, b, dim);
}

inline void normalize_rows(float* data, int n, int dim) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        normalize_inplace(data + (size_t)i * dim, dim);
}

} // namespace dann
