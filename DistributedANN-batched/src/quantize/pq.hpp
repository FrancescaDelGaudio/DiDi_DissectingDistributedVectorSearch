#pragma once

// Product Quantization (Jegou et al., 2010) — global codebook.
//
// A d-dimensional vector is split into m contiguous subspaces. Each subspace
// has its own codebook of ksub = 2^nbits centroids (nbits = 8 -> 256). A vector
// is encoded as m bytes: the nearest centroid index per subspace.
//
// Asymmetric Distance Computation (ADC): given a query, precompute a table of
// squared distances from each query subvector to that subspace's centroids
// (m * ksub floats). The approximate squared L2 distance to an encoded vector
// is then the sum over subspaces of table[j * ksub + code[j]] — m table lookups.
//
// Subspace boundaries: if dim % m != 0, the first (dim % m) subspaces get one
// extra dimension. They are recomputed deterministically from (dim, m).

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <limits>
#include <stdexcept>
#include <algorithm>

#include "../common/distance.hpp"
#include "../common/metric.hpp"

namespace dann {

static constexpr uint32_t PQ_MAGIC   = 0x44515150u;   // 'PQQD'
static constexpr uint32_t PQ_VERSION = 1u;

class ProductQuantizer {
public:
    int dim   = 0;
    int m     = 0;       // number of subspaces
    int ksub  = 256;     // centroids per subspace (nbits = 8)
    Metric metric = Metric::L2;
    std::vector<int> sub_start;   // m+1 boundaries
    std::vector<int> cb_off;      // m offsets into centroids[]
    std::vector<float> centroids; // packed per subspace: ksub * sublen floats

    ProductQuantizer() = default;
    ProductQuantizer(int dim_, int m_, int ksub_ = 256) { init(dim_, m_, ksub_); }

    int sublen(int j) const { return sub_start[j + 1] - sub_start[j]; }

    float sub_score(const float* a, const float* b, int len) const {
        return dist_score(metric, a, b, len);
    }

    void init(int dim_, int m_, int ksub_) {
        dim = dim_; m = m_; ksub = ksub_;
        if (m > dim) throw std::runtime_error("PQ: m > dim");
        sub_start.assign(m + 1, 0);
        int base = dim / m, rem = dim % m;
        for (int j = 0; j < m; j++)
            sub_start[j + 1] = sub_start[j] + base + (j < rem ? 1 : 0);
        cb_off.assign(m, 0);
        int off = 0;
        for (int j = 0; j < m; j++) { cb_off[j] = off; off += ksub * sublen(j); }
        centroids.assign(off, 0.0f);
    }

    // ------------------------------------------------------------------
    // Train all subspaces (Lloyd's k-means) on a sample. OpenMP over subspaces.
    // sample: n * dim row-major.
    // ------------------------------------------------------------------
    void train(const float* sample, int n, int iters = 25, unsigned seed = 42);

    // Encode one vector into m bytes.
    void encode(const float* v, uint8_t* code) const {
        for (int j = 0; j < m; j++) {
            int s = sub_start[j], len = sublen(j);
            const float* cb = centroids.data() + cb_off[j];
            const float* x = v + s;
            float best = std::numeric_limits<float>::infinity();
            int   bk   = 0;
            for (int c = 0; c < ksub; c++) {
                float d = sub_score(x, cb + (size_t)c * len, len);
                if (d < best) { best = d; bk = c; }
            }
            code[j] = (uint8_t)bk;
        }
    }

    // Build the ADC table for a query: m * ksub squared distances.
    void adc_table(const float* q, std::vector<float>& table) const {
        table.assign((size_t)m * ksub, 0.0f);
        for (int j = 0; j < m; j++) {
            int s = sub_start[j], len = sublen(j);
            const float* cb = centroids.data() + cb_off[j];
            const float* x = q + s;
            float* t = table.data() + (size_t)j * ksub;
            for (int c = 0; c < ksub; c++)
                t[c] = sub_score(x, cb + (size_t)c * len, len);
        }
    }

    // Approximate squared L2 distance from a precomputed ADC table to a code.
    float adc_distance(const std::vector<float>& table, const uint8_t* code) const {
        float s = 0.0f;
        for (int j = 0; j < m; j++) s += table[(size_t)j * ksub + code[j]];
        return s;
    }

    // ------------------------------------------------------------------
    // Persistence.
    // ------------------------------------------------------------------
    void save(const std::string& path) const {
        FILE* fp = fopen(path.c_str(), "wb");
        if (!fp) throw std::runtime_error("Cannot create: " + path);
        uint32_t hdr[5] = { PQ_MAGIC, PQ_VERSION, (uint32_t)dim, (uint32_t)m, (uint32_t)ksub };
        fwrite(hdr, 4, 5, fp);
        uint32_t csize = (uint32_t)centroids.size();
        fwrite(&csize, 4, 1, fp);
        fwrite(centroids.data(), 4, centroids.size(), fp);
        fclose(fp);
    }

    void load(const std::string& path) {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) throw std::runtime_error("Cannot open: " + path);
        uint32_t hdr[5];
        if (fread(hdr, 4, 5, fp) != 5) throw std::runtime_error("Corrupt PQ header");
        if (hdr[0] != PQ_MAGIC) throw std::runtime_error("Bad PQ magic");
        init((int)hdr[2], (int)hdr[3], (int)hdr[4]);
        uint32_t csize;
        if (fread(&csize, 4, 1, fp) != 1 || csize != centroids.size())
            throw std::runtime_error("PQ size mismatch");
        if (fread(centroids.data(), 4, centroids.size(), fp) != centroids.size())
            throw std::runtime_error("Truncated PQ centroids");
        fclose(fp);
    }
};

// ---------------------------------------------------------------------------
// Training implementation.
// ---------------------------------------------------------------------------
inline void ProductQuantizer::train(const float* sample, int n, int iters, unsigned seed)
{
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < m; j++) {
        int s = sub_start[j], len = sublen(j);
        float* cb = centroids.data() + cb_off[j];

        std::mt19937 rng(seed + 100u * j);
        std::uniform_int_distribution<int> pick(0, n - 1);

        // Init: distinct random sample points (subvectors).
        for (int c = 0; c < ksub; c++) {
            const float* src = sample + (size_t)pick(rng) * dim + s;
            std::copy(src, src + len, cb + (size_t)c * len);
        }

        std::vector<int>    assign((size_t)n, 0);
        std::vector<double> sums((size_t)ksub * len);
        std::vector<int>    cnts((size_t)ksub);

        for (int it = 0; it < iters; it++) {
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(cnts.begin(), cnts.end(), 0);

            for (int i = 0; i < n; i++) {
                const float* x = sample + (size_t)i * dim + s;
                float best = std::numeric_limits<float>::infinity();
                int   bk   = 0;
                for (int c = 0; c < ksub; c++) {
                    float d = sub_score(x, cb + (size_t)c * len, len);
                    if (d < best) { best = d; bk = c; }
                }
                assign[i] = bk;
                double* acc = sums.data() + (size_t)bk * len;
                for (int t = 0; t < len; t++) acc[t] += x[t];
                cnts[bk]++;
            }

            for (int c = 0; c < ksub; c++) {
                if (cnts[c] == 0) {
                    // Reseed empty centroid to a random point.
                    const float* src = sample + (size_t)pick(rng) * dim + s;
                    std::copy(src, src + len, cb + (size_t)c * len);
                    continue;
                }
                double inv = 1.0 / cnts[c];
                float* cen = cb + (size_t)c * len;
                const double* acc = sums.data() + (size_t)c * len;
                for (int t = 0; t < len; t++) cen[t] = (float)(acc[t] * inv);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// codes.bin — all PQ codes, id-ordered.
//   uint32 magic = 0x45444F43 ('CODE'), version, N, m
//   uint8  codes[N * m]
// ---------------------------------------------------------------------------
static constexpr uint32_t CODES_MAGIC   = 0x45444F43u;
static constexpr uint32_t CODES_VERSION = 1u;
static constexpr uint64_t CODES_HDR_BYTES = 16u;

} // namespace dann
