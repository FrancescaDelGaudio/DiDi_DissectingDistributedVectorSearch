#pragma once

// Orchestration Service (paper Algorithm 2, §2.4) + distributed Node Scoring (Alg. 1).
//
// Rank 0 runs the orchestrator (head search + H-hop beam search). All ranks
// participate in each NodeScoring call: keys are broadcast, each rank scores
// keys it owns in the local KV shard, partial (R,C) lists are gathered.

#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "../quantize/pq.hpp"
#include "kv_shard.hpp"
#include "heaps.hpp"
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_ip.h"

namespace dann {

struct QueryParams {
    int k       = 10;    // result count
    int L       = 100;   // candidate heap size
    int BW      = 64;    // beam width (keys per hop)
    int H       = 5;     // beam iterations (hops)
    int k_head  = 50;    // head index search width
    int ef_head = 100;   // HNSW efSearch for head index
};

inline void mpi_gather_iddist(MPI_Comm comm, int rank,
                              const std::vector<IdDist>& local,
                              std::vector<IdDist>& global)
{
    int local_n = (int)local.size();
    int nranks = 0;
    MPI_Comm_size(comm, &nranks);

    std::vector<int> counts((size_t)nranks), displs((size_t)nranks);
    std::vector<int> counts_b((size_t)nranks), displs_b((size_t)nranks);
    int total = 0;

    MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < nranks; r++) displs[r] = displs[r - 1] + counts[r - 1];
        for (int r = 0; r < nranks; r++) total += counts[r];
        global.resize((size_t)total);
        for (int r = 0; r < nranks; r++) {
            counts_b[r] = counts[r] * (int)sizeof(IdDist);
            displs_b[r] = displs[r] * (int)sizeof(IdDist);
        }
    }

    MPI_Gatherv((void*)local.data(), local_n * (int)sizeof(IdDist), MPI_BYTE,
                rank == 0 ? global.data() : nullptr,
                rank == 0 ? counts_b.data() : nullptr,
                rank == 0 ? displs_b.data() : nullptr,
                MPI_BYTE, 0, comm);
}

inline void distributed_node_scoring(MPI_Comm comm, int rank, int nranks,
                                     const KVShard& shard,
                                     const ProductQuantizer& pq,
                                     Metric metric,
                                     int dim,
                                     const uint32_t* keys, int nkeys,
                                     float threshold, int candidate_limit,
                                     const float* q,
                                     const std::vector<float>& adc_table,
                                     std::vector<IdDist>& out_R,
                                     std::vector<IdDist>& out_C)
{
    // Broadcast request metadata and query from rank 0.
    MPI_Bcast(&nkeys, 1, MPI_INT, 0, comm);
    if (nkeys <= 0) { out_R.clear(); out_C.clear(); return; }

    std::vector<uint32_t> keys_buf((size_t)nkeys);
    if (rank == 0) std::copy(keys, keys + nkeys, keys_buf.begin());
    MPI_Bcast(keys_buf.data(), nkeys, MPI_UINT32_T, 0, comm);
    MPI_Bcast(&threshold, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(&candidate_limit, 1, MPI_INT, 0, comm);

    std::vector<float> q_buf((size_t)dim);
    if (rank == 0) std::copy(q, q + dim, q_buf.begin());
    MPI_Bcast(q_buf.data(), dim, MPI_FLOAT, 0, comm);

    int adc_n = pq.m * pq.ksub;
    std::vector<float> adc_buf((size_t)adc_n);
    if (rank == 0) adc_buf = adc_table;
    MPI_Bcast(adc_buf.data(), adc_n, MPI_FLOAT, 0, comm);

    std::vector<IdDist> local_R, local_C;
    shard.score_local(keys_buf.data(), nkeys, threshold, candidate_limit,
                      q_buf.data(), adc_buf, pq, metric, local_R, local_C);

    mpi_gather_iddist(comm, rank, local_R, out_R);
    mpi_gather_iddist(comm, rank, local_C, out_C);
    if (rank == 0) {
        dedupe_best(out_R);
        dedupe_best(out_C);
        if ((int)out_C.size() > candidate_limit)
            out_C.resize((size_t)candidate_limit);
    }
}

class Orchestrator {
public:
    QueryParams params;
    Metric metric = Metric::L2;
    ProductQuantizer pq;
    KVShard       shard;
    hnswlib::SpaceInterface<float>* head_space = nullptr;
    hnswlib::HierarchicalNSW<float>* head = nullptr;

    ~Orchestrator() {
        delete head;
        delete head_space;
    }

    void load_head(const std::string& head_path, int dim, Metric metric_) {
        delete head;
        delete head_space;
        head_space = nullptr;
        metric = metric_;
        if (metric == Metric::IP)
            head_space = new hnswlib::InnerProductSpace((size_t)dim);
        else
            head_space = new hnswlib::L2Space((size_t)dim);
        head = new hnswlib::HierarchicalNSW<float>(head_space, head_path);
        head->setEf(params.ef_head);
    }

    // Algorithm 2 — returns top-k result ids on rank 0 (sorted by distance).
    std::vector<uint32_t> search(MPI_Comm comm, int rank, int nranks,
                                 const float* q,
                                 uint64_t* stats_nodes_scored = nullptr) const
    {
        std::vector<uint32_t> result;
        std::vector<IdDist> H_R, H_C;
        std::unordered_map<uint32_t, char> visited;
        uint64_t nodes_scored = 0;

        std::vector<float> q_local((size_t)pq.dim);
        if (rank == 0) std::copy(q, q + pq.dim, q_local.begin());
        MPI_Bcast(q_local.data(), pq.dim, MPI_FLOAT, 0, comm);

        std::vector<float> adc_table;
        if (rank == 0) pq.adc_table(q_local.data(), adc_table);

        if (rank == 0 && head) {
            auto res = head->searchKnn(q_local.data(), (size_t)params.k_head);
            while (!res.empty()) {
                auto top = res.top();
                res.pop();
                H_C.push_back({(uint32_t)top.second, top.first});
            }
            dedupe_best(H_C);
            if ((int)H_C.size() > params.L) H_C.resize((size_t)params.L);
        }

        std::vector<uint32_t> keys((size_t)params.BW);
        std::vector<IdDist> batch_R, batch_C;

        for (int hop = 0; hop < params.H; hop++) {
            int nkeys = 0;
            float t   = std::numeric_limits<float>::infinity();
            if (rank == 0) {
                if (!H_C.empty()) {
                    t = candidate_threshold(H_C, params.L);
                    nkeys = pick_beam(H_C, params.BW, visited, keys.data());
                }
            }
            MPI_Bcast(&nkeys, 1, MPI_INT, 0, comm);
            if (nkeys == 0) break;
            if (rank == 0) nodes_scored += (uint64_t)nkeys;

            distributed_node_scoring(comm, rank, nranks, shard, pq, metric, pq.dim,
                                     keys.data(), nkeys, t, params.L,
                                     q_local.data(), adc_table,
                                     batch_R, batch_C);

            if (rank == 0) {
                merge_cap(H_R, batch_R, params.k);
                merge_cap(H_C, batch_C, params.L);
            }
        }

        if (rank == 0) {
            if (stats_nodes_scored) *stats_nodes_scored = nodes_scored;
            result.reserve((size_t)params.k);
            for (auto& e : H_R) result.push_back(e.id);
            if ((int)result.size() > params.k) result.resize((size_t)params.k);
        }
        return result;
    }
};

} // namespace dann
