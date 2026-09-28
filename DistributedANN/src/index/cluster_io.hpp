#pragma once

// Per-cluster Vamana graph persistence.
//
// File layout (cluster_<c>.graph), little-endian:
//   uint32  magic   = 0x44414E47   ('G','N','A','D')
//   uint32  version = 1
//   uint32  cluster_id
//   uint32  n               (nodes in this cluster, incl. epsilon replicas)
//   uint32  dim
//   uint32  R               (max out-degree used at build)
//   uint32  medoid_global   (global id of the cluster medoid)
//   uint32  ids[n]                 global id of each node (local order)
//   float   vectors[n*dim]         full-precision vectors (local order)
//   uint32  offsets[n+1]           CSR adjacency row pointers
//   uint32  neighbors[offsets[n]]  neighbour GLOBAL ids
//
// Neighbours are stored as GLOBAL ids so Step 3 (stitching) can directly
// union the neighbour lists of a vector that appears in several clusters.

#ifndef _FILE_OFFSET_BITS
#  define _FILE_OFFSET_BITS 64
#endif

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <sys/types.h>

#include "../common/types.hpp"

namespace dann {

static constexpr uint32_t CLUSTER_MAGIC   = 0x44414E47u;
static constexpr uint32_t CLUSTER_VERSION = 1u;

// In-memory representation of a loaded cluster graph (used by Step 3).
struct ClusterGraph {
    uint32_t cluster_id = 0;
    uint32_t n          = 0;
    uint32_t dim        = 0;
    uint32_t R          = 0;
    uint32_t medoid_global = 0;
    std::vector<uint32_t> ids;        // n
    std::vector<float>    vectors;    // n * dim
    std::vector<uint32_t> offsets;    // n + 1
    std::vector<uint32_t> neighbors;  // offsets[n]  (global ids)
};

// Write a cluster graph.
//   ids        : n global ids (local order)
//   vectors    : n * dim
//   adj        : per-node neighbour lists as GLOBAL ids
inline void write_cluster_graph(const std::string& path,
                                uint32_t cluster_id,
                                uint32_t n, uint32_t dim, uint32_t R,
                                uint32_t medoid_global,
                                const uint32_t* ids,
                                const float* vectors,
                                const std::vector<std::vector<uint32_t>>& adj)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Cannot create: " + path);

    fwrite(&CLUSTER_MAGIC,   4, 1, fp);
    fwrite(&CLUSTER_VERSION, 4, 1, fp);
    fwrite(&cluster_id,      4, 1, fp);
    fwrite(&n,               4, 1, fp);
    fwrite(&dim,             4, 1, fp);
    fwrite(&R,               4, 1, fp);
    fwrite(&medoid_global,   4, 1, fp);

    fwrite(ids,     4, n,                 fp);
    fwrite(vectors, 4, (size_t)n * dim,   fp);

    // CSR offsets
    uint32_t off = 0;
    for (uint32_t i = 0; i < n; i++) {
        fwrite(&off, 4, 1, fp);
        off += (uint32_t)adj[i].size();
    }
    fwrite(&off, 4, 1, fp);   // sentinel

    // Neighbour data
    for (uint32_t i = 0; i < n; i++)
        if (!adj[i].empty())
            fwrite(adj[i].data(), 4, adj[i].size(), fp);

    fclose(fp);
}

// Lightweight reader for Step 3 stitching pass 1: ids + adjacency only,
// SKIPPING the stored full vectors (loaded selectively in pass 2 via
// read_cluster_graph when RobustPrune needs them).
// Returns false if the file does not exist (e.g. an empty cluster).
inline bool read_cluster_adjacency(const std::string& path,
                                   uint32_t& cluster_id,
                                   uint32_t& n, uint32_t& dim, uint32_t& R,
                                   uint32_t& medoid_global,
                                   std::vector<uint32_t>& ids,
                                   std::vector<uint32_t>& offsets,
                                   std::vector<uint32_t>& neighbors)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return false;

    uint32_t magic, version;
    if (fread(&magic,         4, 1, fp) != 1 ||
        fread(&version,       4, 1, fp) != 1 ||
        fread(&cluster_id,    4, 1, fp) != 1 ||
        fread(&n,             4, 1, fp) != 1 ||
        fread(&dim,           4, 1, fp) != 1 ||
        fread(&R,             4, 1, fp) != 1 ||
        fread(&medoid_global, 4, 1, fp) != 1) {
        fclose(fp);
        throw std::runtime_error("Corrupt header in: " + path);
    }
    if (magic != CLUSTER_MAGIC) {
        fclose(fp);
        throw std::runtime_error("Bad magic in: " + path);
    }

    ids.resize(n);
    if (fread(ids.data(), 4, n, fp) != n)
        throw std::runtime_error("Truncated ids in: " + path);

    // Skip the stored vectors (n * dim float32).
    if (fseeko(fp, (off_t)n * dim * 4, SEEK_CUR) != 0)
        throw std::runtime_error("Seek past vectors failed in: " + path);

    offsets.resize((size_t)n + 1);
    if (fread(offsets.data(), 4, (size_t)n + 1, fp) != (size_t)n + 1)
        throw std::runtime_error("Truncated offsets in: " + path);

    uint32_t total = offsets[n];
    neighbors.resize(total);
    if (total > 0 && fread(neighbors.data(), 4, total, fp) != total)
        throw std::runtime_error("Truncated neighbors in: " + path);

    fclose(fp);
    return true;
}

// Load a cluster graph (Step 3).
inline ClusterGraph read_cluster_graph(const std::string& path)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic, version;
    ClusterGraph g;
    if (fread(&magic,            4, 1, fp) != 1 ||
        fread(&version,          4, 1, fp) != 1 ||
        fread(&g.cluster_id,     4, 1, fp) != 1 ||
        fread(&g.n,              4, 1, fp) != 1 ||
        fread(&g.dim,            4, 1, fp) != 1 ||
        fread(&g.R,              4, 1, fp) != 1 ||
        fread(&g.medoid_global,  4, 1, fp) != 1)
        throw std::runtime_error("Corrupt header in: " + path);
    if (magic != CLUSTER_MAGIC)
        throw std::runtime_error("Bad magic in: " + path);

    g.ids.resize(g.n);
    g.vectors.resize((size_t)g.n * g.dim);
    g.offsets.resize((size_t)g.n + 1);

    if (fread(g.ids.data(),     4, g.n,                  fp) != g.n)
        throw std::runtime_error("Truncated ids in: " + path);
    if (fread(g.vectors.data(), 4, (size_t)g.n * g.dim,  fp) != (size_t)g.n * g.dim)
        throw std::runtime_error("Truncated vectors in: " + path);
    if (fread(g.offsets.data(), 4, (size_t)g.n + 1,      fp) != (size_t)g.n + 1)
        throw std::runtime_error("Truncated offsets in: " + path);

    uint32_t total = g.offsets[g.n];
    g.neighbors.resize(total);
    if (total > 0 && fread(g.neighbors.data(), 4, total, fp) != total)
        throw std::runtime_error("Truncated neighbors in: " + path);

    fclose(fp);
    return g;
}

} // namespace dann
