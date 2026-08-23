#pragma once

// KV store shard + local Node Scoring Service (paper Algorithm 1, §2.3).
//
// At query time nodes.bin is sharded by hash(id) % W (deterministic pseudo-random
// sharding as discussed for the experimental setup). Each rank loads only the
// nodes it owns into memory for near-data scoring.

#ifndef _FILE_OFFSET_BITS
#  define _FILE_OFFSET_BITS 64
#endif

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "../common/distance.hpp"
#include "../common/metric.hpp"
#include "../index/kvstore_io.hpp"
#include "../quantize/pq.hpp"
#include "heaps.hpp"

namespace dann {

struct KVNode {
    std::vector<float>    vec;
    std::vector<uint32_t> nbr_ids;
    std::vector<uint8_t>  nbr_codes;   // flat: deg * m bytes
};

class KVShard {
public:
    int      rank_ = 0;
    int      W_    = 1;
    uint32_t N_    = 0;
    int      dim_  = 0;
    int      m_    = 0;
    std::unordered_map<uint32_t, KVNode> nodes_;

    static inline uint32_t owner(uint32_t id, int W) {
        return id % (uint32_t)W;
    }

    void load(const std::string& path, int rank, int W) {
        rank_ = rank;
        W_    = W;
        nodes_.clear();

        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open KV store: " + path);
        struct stat sb;
        if (fstat(fd, &sb) != 0) { close(fd); throw std::runtime_error("fstat failed"); }
        const char* map = (const char*)mmap(nullptr, (size_t)sb.st_size, PROT_READ,
                                              MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED) { close(fd); throw std::runtime_error("mmap failed"); }

        uint32_t magic, ver, pad;
        uint64_t total_edges;
        std::memcpy(&magic, map, 4);
        std::memcpy(&ver,   map + 4, 4);
        std::memcpy(&N_,    map + 8, 4);
        std::memcpy(&dim_,  map + 12, 4);
        std::memcpy(&m_,    map + 16, 4);
        std::memcpy(&pad,   map + 20, 4);
        std::memcpy(&total_edges, map + 24, 8);
        if (magic != KV_MAGIC) {
            munmap((void*)map, sb.st_size); close(fd);
            throw std::runtime_error("Bad KV magic");
        }

        const uint64_t off_off  = kv_off_offsets();
        const uint64_t off_vec  = kv_off_vecs(N_);
        const uint64_t off_ids  = kv_off_ids(N_, dim_);
        const uint64_t off_code = kv_off_codes(N_, dim_, total_edges);

        const uint64_t* offsets = (const uint64_t*)(map + off_off);
        const float*    vecs    = (const float*   )(map + off_vec);
        const uint32_t* ids     = (const uint32_t* )(map + off_ids);
        const uint8_t*  codes   = (const uint8_t*  )(map + off_code);

        size_t owned = 0;
        for (uint32_t id = 0; id < N_; id++) {
            if (owner(id, W_) != (uint32_t)rank_) continue;
            uint64_t a = offsets[id], b = offsets[id + 1];
            uint32_t deg = (uint32_t)(b - a);

            KVNode node;
            node.vec.assign(vecs + (size_t)id * dim_, vecs + (size_t)id * dim_ + dim_);
            node.nbr_ids.assign(ids + a, ids + b);
            node.nbr_codes.resize((size_t)deg * (size_t)m_);
            if (deg > 0)
                std::memcpy(node.nbr_codes.data(), codes + a * (size_t)m_,
                            (size_t)deg * (size_t)m_);

            nodes_.emplace(id, std::move(node));
            owned++;
        }

        munmap((void*)map, sb.st_size);
        close(fd);

        if (rank_ == 0) {
            printf("[rank 0] KV shard loaded: N=%u dim=%d m=%d W=%d owned=%zu\n",
                   N_, dim_, m_, W_, owned);
            fflush(stdout);
        }
    }

    // Algorithm 1 — score only keys owned by this rank.
    void score_local(const uint32_t* keys, int nkeys,
                     float threshold, int candidate_limit,
                     const float* q,
                     const std::vector<float>& adc_table,
                     const ProductQuantizer& pq,
                     Metric metric,
                     std::vector<IdDist>& out_R,
                     std::vector<IdDist>& out_C) const
    {
        out_R.clear();
        out_C.clear();
        out_R.reserve((size_t)nkeys);
        out_C.reserve((size_t)nkeys * 32);

        for (int ki = 0; ki < nkeys; ki++) {
            uint32_t id = keys[ki];
            auto it = nodes_.find(id);
            if (it == nodes_.end()) continue;

            const KVNode& node = it->second;
            float d_exact = dist_score(metric, q, node.vec.data(), dim_);
            out_R.push_back({id, d_exact});

            uint32_t deg = (uint32_t)node.nbr_ids.size();
            for (uint32_t e = 0; e < deg; e++) {
                float d_pq = pq.adc_distance(adc_table,
                                             node.nbr_codes.data() + (size_t)e * (size_t)m_);
                if (d_pq < threshold)
                    out_C.push_back({node.nbr_ids[e], d_pq});
            }
        }

        dedupe_best(out_R);
        dedupe_best(out_C);
        if ((int)out_C.size() > candidate_limit)
            out_C.resize((size_t)candidate_limit);
    }
};

} // namespace dann
