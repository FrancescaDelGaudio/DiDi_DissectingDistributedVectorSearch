#pragma once

// Distributed KV store node table — Step 6 output (paper §2.1/§2.4).
//
// Key = node global id (implicit row index). Value = {full vector, neighbour
// ids, neighbour PQ codes}. The neighbour codes are the "compressed vectors
// duplicated into graph nodes" (§2.2) consumed by the node-scoring service.
//
// File layout (nodes.bin), little-endian:
//   offset 0   uint32 magic = 0x4B565344  ('D','S','V','K')
//   offset 4   uint32 version = 1
//   offset 8   uint32 N
//   offset 12  uint32 dim
//   offset 16  uint32 m            (PQ code length in bytes)
//   offset 20  uint32 _pad
//   offset 24  uint64 total_edges
//   offset 32                      uint64 offsets[N+1]         (CSR row ptrs)
//   off_vec  = 32+(N+1)*8          float  full_vecs[N*dim]
//   off_ids  = off_vec+N*dim*4     uint32 nbr_ids[total]
//   off_code = off_ids+total*4     uint8  nbr_codes[total*m]
//
// Node i: vector at full_vecs[i*dim]; neighbours j in [offsets[i],offsets[i+1])
// with id nbr_ids[j] and PQ code nbr_codes[j*m .. j*m+m).

#include <cstdint>

namespace dann {

static constexpr uint32_t KV_MAGIC     = 0x4B565344u;
static constexpr uint32_t KV_VERSION   = 1u;
static constexpr uint64_t KV_HDR_BYTES = 32u;

// Byte offsets of the four sections, given N, dim, m, total_edges.
inline uint64_t kv_off_offsets()                       { return KV_HDR_BYTES; }
inline uint64_t kv_off_vecs(uint64_t N)                { return KV_HDR_BYTES + (N + 1) * 8; }
inline uint64_t kv_off_ids(uint64_t N, uint64_t dim)   { return kv_off_vecs(N) + N * dim * 4; }
inline uint64_t kv_off_codes(uint64_t N, uint64_t dim, uint64_t total) {
    return kv_off_ids(N, dim) + total * 4;
}

} // namespace dann
