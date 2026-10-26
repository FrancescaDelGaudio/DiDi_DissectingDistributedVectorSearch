#pragma once

// Unified (stitched) global graph — Step 3 output.
//
// File layout (global_graph.bin), little-endian:
//   offset 0   uint32 magic   = 0x47424C47  ('G','L','B','G')
//   offset 4   uint32 version = 1
//   offset 8   uint32 N            (number of nodes = dataset size)
//   offset 12  uint32 dim
//   offset 16  uint32 R            (max out-degree used for pruning)
//   offset 20  uint32 _pad
//   offset 24  uint64 total_edges
//   offset 32  uint64 offsets[N+1]                 (CSR row pointers)
//   offset 32+(N+1)*8  uint32 neighbors[total]     (neighbour global ids)
//
// Node id is implicit (the row index): node i's neighbours are
// neighbors[offsets[i] .. offsets[i+1]). Vectors are NOT stored here; they
// live in base.fvecs (and later in the KV store).

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

namespace dann {

static constexpr uint32_t GLOBAL_MAGIC    = 0x47424C47u;
static constexpr uint32_t GLOBAL_VERSION  = 1u;
static constexpr uint64_t GLOBAL_HDR_BYTES = 32u;

struct GlobalGraph {
    uint32_t N = 0, dim = 0, R = 0;
    uint64_t total_edges = 0;
    std::vector<uint64_t> offsets;    // N+1
    std::vector<uint32_t> neighbors;  // total_edges
};

// Full in-memory loader (fine for experimental scale; use mmap for huge N).
inline GlobalGraph read_global_graph(const std::string& path)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic, version, pad;
    GlobalGraph g;
    if (fread(&magic,        4, 1, fp) != 1 ||
        fread(&version,      4, 1, fp) != 1 ||
        fread(&g.N,          4, 1, fp) != 1 ||
        fread(&g.dim,        4, 1, fp) != 1 ||
        fread(&g.R,          4, 1, fp) != 1 ||
        fread(&pad,          4, 1, fp) != 1 ||
        fread(&g.total_edges,8, 1, fp) != 1)
        throw std::runtime_error("Corrupt header in: " + path);
    if (magic != GLOBAL_MAGIC)
        throw std::runtime_error("Bad magic in: " + path);

    g.offsets.resize((size_t)g.N + 1);
    if (fread(g.offsets.data(), 8, (size_t)g.N + 1, fp) != (size_t)g.N + 1)
        throw std::runtime_error("Truncated offsets in: " + path);

    g.neighbors.resize(g.total_edges);
    if (g.total_edges > 0 &&
        fread(g.neighbors.data(), 4, g.total_edges, fp) != g.total_edges)
        throw std::runtime_error("Truncated neighbors in: " + path);

    fclose(fp);
    return g;
}

} // namespace dann
