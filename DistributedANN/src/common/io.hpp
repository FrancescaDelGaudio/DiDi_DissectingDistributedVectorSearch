#pragma once

// Large-file support: off_t and fseeko are 64-bit on 64-bit Linux by default,
// but define this to be safe on any 32-bit toolchain.
#ifndef _FILE_OFFSET_BITS
#  define _FILE_OFFSET_BITS 64
#endif

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <sys/types.h>

#include "types.hpp"

namespace dann {

// ---------------------------------------------------------------------------
// fvecs I/O
// Format per vector: [int32 dim][float32 x dim]
// All vectors in a file must have the same dimension.
// ---------------------------------------------------------------------------

// Returns the number of vectors in a .fvecs file.
inline idx_t fvecs_count(const std::string& path)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);
    int32_t d;
    if (fread(&d, 4, 1, fp) != 1) throw std::runtime_error("Cannot read dim from: " + path);
    fseeko(fp, 0, SEEK_END);
    off_t file_size = ftello(fp);
    fclose(fp);
    size_t bytes_per_vec = 4 + (size_t)d * 4;
    return (idx_t)(file_size / (off_t)bytes_per_vec);
}

// Returns the dimension of vectors in a .fvecs file.
inline int fvecs_dim(const std::string& path)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);
    int32_t d;
    if (fread(&d, 4, 1, fp) != 1) throw std::runtime_error("Cannot read dim from: " + path);
    fclose(fp);
    return (int)d;
}

// Read a contiguous slice [start_idx, start_idx + count) from a .fvecs file.
// dim is set to the vector dimension detected from the file header.
// Returns data in row-major order: data[i * dim + j].
inline std::vector<float> read_fvecs_slice(const std::string& path,
                                            idx_t start_idx,
                                            idx_t count,
                                            int& dim)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);

    int32_t d;
    if (fread(&d, 4, 1, fp) != 1) throw std::runtime_error("Cannot read dim from: " + path);
    dim = (int)d;

    size_t bytes_per_vec = 4 + (size_t)d * 4;
    off_t  seek_to       = (off_t)start_idx * (off_t)bytes_per_vec;
    if (fseeko(fp, seek_to, SEEK_SET) != 0)
        throw std::runtime_error("fseeko failed in: " + path);

    std::vector<float> data((size_t)count * (size_t)d);
    for (idx_t i = 0; i < count; i++) {
        int32_t check_d;
        if (fread(&check_d, 4, 1, fp) != 1)
            throw std::runtime_error("Unexpected EOF at vector " + std::to_string(start_idx + i));
        if (fread(data.data() + (size_t)i * (size_t)d, 4, (size_t)d, fp) != (size_t)d)
            throw std::runtime_error("Truncated vector " + std::to_string(start_idx + i));
    }
    fclose(fp);
    return data;
}

// Write n vectors of given dim as a .fvecs file.
inline void write_fvecs(const std::string& path, const float* data, int n, int dim)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Cannot create: " + path);
    int32_t d = dim;
    for (int i = 0; i < n; i++) {
        fwrite(&d, 4, 1, fp);
        fwrite(data + (size_t)i * (size_t)dim, 4, (size_t)dim, fp);
    }
    fclose(fp);
}

// ---------------------------------------------------------------------------
// assignments.bin  — CSR binary format
//
//   uint32  magic   = 0x444E4E41  ('A','N','N','D')
//   uint32  version = 1
//   uint32  N       (total vectors)
//   uint32  K       (total clusters)
//   uint32  total   (sum of all assignment list lengths)
//   uint32  offsets[N+1]   (CSR row-pointer; offsets[i] = start of vector i)
//   uint32  data[total]    (cluster ids, sorted nearest-first)
// ---------------------------------------------------------------------------

static constexpr uint32_t ASSIGN_MAGIC   = 0x444E4E41u;
static constexpr uint32_t ASSIGN_VERSION = 1u;

inline void write_assignments(const std::string& path,
                               const std::vector<std::vector<clust_t>>& assignments,
                               uint32_t K)
{
    uint32_t N     = (uint32_t)assignments.size();
    uint32_t total = 0;
    for (auto& a : assignments) total += (uint32_t)a.size();

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Cannot create: " + path);

    fwrite(&ASSIGN_MAGIC,   4, 1, fp);
    fwrite(&ASSIGN_VERSION, 4, 1, fp);
    fwrite(&N,              4, 1, fp);
    fwrite(&K,              4, 1, fp);
    fwrite(&total,          4, 1, fp);

    // CSR offsets (N+1 entries, last = sentinel = total)
    uint32_t off = 0;
    for (uint32_t i = 0; i < N; i++) {
        fwrite(&off, 4, 1, fp);
        off += (uint32_t)assignments[i].size();
    }
    fwrite(&off, 4, 1, fp);

    // Data
    for (auto& a : assignments)
        if (!a.empty())
            fwrite(a.data(), 4, a.size(), fp);

    fclose(fp);
}

// Read only the header of assignments.bin (cheap: no CSR payload).
inline void read_assignments_header(const std::string& path,
                                    uint32_t& N, uint32_t& K)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);
    uint32_t magic, version, total;
    if (fread(&magic,   4, 1, fp) != 1 ||
        fread(&version, 4, 1, fp) != 1 ||
        fread(&N,       4, 1, fp) != 1 ||
        fread(&K,       4, 1, fp) != 1 ||
        fread(&total,   4, 1, fp) != 1) {
        fclose(fp);
        throw std::runtime_error("Corrupt header in: " + path);
    }
    fclose(fp);
    if (magic != ASSIGN_MAGIC)
        throw std::runtime_error("Bad magic in: " + path);
}

// Read assignments.bin into CSR form.
//   offsets : N+1 entries (offsets[i]..offsets[i+1] = cluster list of vector i)
//   data    : total cluster ids
// Sets N and K from the header.
inline void read_assignments(const std::string& path,
                             std::vector<uint32_t>& offsets,
                             std::vector<uint32_t>& data,
                             uint32_t& N, uint32_t& K)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic, version, total;
    if (fread(&magic,   4, 1, fp) != 1 ||
        fread(&version, 4, 1, fp) != 1 ||
        fread(&N,       4, 1, fp) != 1 ||
        fread(&K,       4, 1, fp) != 1 ||
        fread(&total,   4, 1, fp) != 1)
        throw std::runtime_error("Corrupt header in: " + path);

    if (magic != ASSIGN_MAGIC)
        throw std::runtime_error("Bad magic in: " + path);

    offsets.resize((size_t)N + 1);
    if (fread(offsets.data(), 4, (size_t)N + 1, fp) != (size_t)N + 1)
        throw std::runtime_error("Truncated offsets in: " + path);

    data.resize(total);
    if (total > 0 && fread(data.data(), 4, total, fp) != total)
        throw std::runtime_error("Truncated data in: " + path);

    fclose(fp);
}

// ---------------------------------------------------------------------------
// .fbin I/O  (Big-ANN / DEEP format: int32 n, int32 d, float32[n*d])
// ---------------------------------------------------------------------------
inline std::vector<float> read_fbin(const std::string& path, idx_t& n, int& dim) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);
    int32_t n32 = 0, d32 = 0;
    if (fread(&n32, 4, 1, fp) != 1 || fread(&d32, 4, 1, fp) != 1)
        throw std::runtime_error("Bad fbin header: " + path);
    n = (idx_t)n32; dim = (int)d32;
    std::vector<float> data((size_t)n * (size_t)dim);
    if (n > 0 && fread(data.data(), 4, (size_t)n * (size_t)dim, fp) != (size_t)n * (size_t)dim)
        throw std::runtime_error("Truncated fbin: " + path);
    fclose(fp);
    return data;
}

inline idx_t fbin_count(const std::string& path) {
    idx_t n = 0; int d = 0;
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);
    int32_t n32 = 0, d32 = 0;
    fread(&n32, 4, 1, fp); fread(&d32, 4, 1, fp);
    fclose(fp);
    n = (idx_t)n32; (void)d;
    return n;
}

} // namespace dann
