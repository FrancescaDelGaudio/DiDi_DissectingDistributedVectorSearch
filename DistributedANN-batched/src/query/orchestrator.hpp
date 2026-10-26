#pragma once

// Orchestration Service (paper Algorithm 2, §2.4) + distributed Node Scoring (Alg. 1).
//
// Rank 0 runs the orchestrator (head search + H-hop beam search). All ranks
// participate in each NodeScoring call: keys are broadcast, each rank scores
// keys it owns in the local KV shard, partial (R,C) lists are gathered.
//
// Optional QueryProfile*: times head / per-hop wall / score(max,mean) / merge /
// approx bytes. Shard score time is measured locally and gathered as 2 doubles
// (score_s, local_keys) so hop_comm = hop_wall - score_max needs no clock sync.

#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cstring>

#include "../quantize/pq.hpp"
#include "kv_shard.hpp"
#include "heaps.hpp"
#include "profile.hpp"
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
                              std::vector<IdDist>& global,
                              uint64_t* bytes_recv_root = nullptr)
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
        if (bytes_recv_root) {
            // counts gather + payload
            *bytes_recv_root += (uint64_t)nranks * sizeof(int)
                             + (uint64_t)total * sizeof(IdDist);
        }
    }

    MPI_Gatherv((void*)local.data(), local_n * (int)sizeof(IdDist), MPI_BYTE,
                rank == 0 ? global.data() : nullptr,
                rank == 0 ? counts_b.data() : nullptr,
                rank == 0 ? displs_b.data() : nullptr,
                MPI_BYTE, 0, comm);
}

// Optional hop timing filled on rank 0 when profile != nullptr.
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
                                     std::vector<IdDist>& out_C,
                                     HopSample* hop = nullptr)
{
    using Clock = std::chrono::steady_clock;
    const bool prof = (hop != nullptr);
    Clock::time_point t_wall0;
    if (prof && rank == 0) t_wall0 = Clock::now();

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

    uint64_t bytes_sent = 0;
    if (prof && rank == 0) {
        // Approximate root→all payload (logical bytes of Bcast data).
        bytes_sent += sizeof(int);                         // nkeys
        bytes_sent += (uint64_t)nkeys * sizeof(uint32_t); // keys
        bytes_sent += sizeof(float) + sizeof(int);         // threshold, limit
        bytes_sent += (uint64_t)dim * sizeof(float);       // query
        bytes_sent += (uint64_t)adc_n * sizeof(float);     // adc
    }

    std::vector<IdDist> local_R, local_C;
    double local_score_s = 0.0;
    double local_keys_d  = 0.0;
    {
        auto ts0 = Clock::now();
        shard.score_local(keys_buf.data(), nkeys, threshold, candidate_limit,
                          q_buf.data(), adc_buf, pq, metric, local_R, local_C);
        auto ts1 = Clock::now();
        local_score_s = std::chrono::duration<double>(ts1 - ts0).count();
        // Count keys this rank actually owns among the beam.
        int owned = 0;
        for (int i = 0; i < nkeys; i++) {
            if (KVShard::owner(keys_buf[(size_t)i], nranks) == (uint32_t)rank)
                owned++;
        }
        local_keys_d = (double)owned;
    }

    // Return shard timing as 2 doubles (no cross-node clock sync needed).
    double mine[2] = { local_score_s, local_keys_d };
    std::vector<double> all;
    if (prof) {
        all.resize((size_t)nranks * 2);
        MPI_Gather(mine, 2, MPI_DOUBLE,
                   rank == 0 ? all.data() : nullptr, 2, MPI_DOUBLE, 0, comm);
    }

    uint64_t bytes_recv = 0;
    if (prof && rank == 0)
        bytes_recv += (uint64_t)nranks * 2 * sizeof(double);

    mpi_gather_iddist(comm, rank, local_R, out_R, prof && rank == 0 ? &bytes_recv : nullptr);
    mpi_gather_iddist(comm, rank, local_C, out_C, prof && rank == 0 ? &bytes_recv : nullptr);

    // dedupe is part of "scoring result packaging" on rank 0 — keep outside merge;
    // user asked merge = heap merge. Dedupe stays here (small).
    if (rank == 0) {
        dedupe_best(out_R);
        dedupe_best(out_C);
        if ((int)out_C.size() > candidate_limit)
            out_C.resize((size_t)candidate_limit);
    }

    if (prof && rank == 0) {
        auto t_wall1 = Clock::now();
        hop->wall_s = std::chrono::duration<double>(t_wall1 - t_wall0).count();
        hop->nkeys = nkeys;
        hop->bytes_sent = bytes_sent;
        hop->bytes_recv = bytes_recv;

        double smax = -1.0, smin = std::numeric_limits<double>::infinity(), ssum = 0;
        int keys_at_max = 0;
        int rank_at_max = 0;
        double s0 = all[0];
        int keys0 = (int)all[1];
        double soth_max = 0, soth_sum = 0;
        int nother = 0;
        for (int r = 0; r < nranks; r++) {
            double s = all[(size_t)r * 2];
            double kloc = all[(size_t)r * 2 + 1];
            ssum += s;
            if (s > smax) {
                smax = s; keys_at_max = (int)kloc; rank_at_max = r;
            }
            if (s < smin) smin = s;
            if (r > 0) {
                nother++;
                soth_sum += s;
                if (s > soth_max) soth_max = s;
            }
        }
        if (smax < 0) smax = 0;
        hop->score_max_s = smax;
        hop->score_min_s = (smin == std::numeric_limits<double>::infinity()) ? 0.0 : smin;
        hop->score_mean_s = ssum / (double)nranks;
        hop->score_rank0_s = s0;
        hop->local_keys_rank0 = keys0;
        hop->score_others_max_s = soth_max;
        hop->score_others_mean_s = (nother > 0) ? soth_sum / (double)nother : 0.0;
        hop->local_keys_at_max = keys_at_max;
        hop->score_max_rank = rank_at_max;
        hop->comm_s = hop->wall_s - hop->score_max_s;
        if (hop->comm_s < 0) hop->comm_s = 0; // numerical / overlap
    }
}

// Empty hop: same Bcast payload size as a real hop with msg_nkeys keys, then
// Gather of empty R/C (+ 2-double meta like profiled path). No score_local.
// All ranks must call. Returns stats on rank 0 only (others get zeros).
inline EmptyHopFloor measure_empty_hop_floor(MPI_Comm comm, int rank, int nranks,
                                             int dim, int adc_n, int msg_nkeys,
                                             int n_iters, int n_warmup = 20)
{
    using Clock = std::chrono::steady_clock;
    EmptyHopFloor out;
    out.n_iters = n_iters;
    out.msg_nkeys = msg_nkeys;

    std::vector<uint32_t> keys((size_t)std::max(msg_nkeys, 0), 0);
    std::vector<float> q((size_t)dim, 0.f);
    std::vector<float> adc((size_t)adc_n, 0.f);
    float threshold = 0.f;
    int candidate_limit = 0;
    int nkeys = msg_nkeys;

    out.bytes_bcast = sizeof(int) + (uint64_t)msg_nkeys * sizeof(uint32_t)
                    + sizeof(float) + sizeof(int)
                    + (uint64_t)dim * sizeof(float)
                    + (uint64_t)adc_n * sizeof(float);

    auto one_hop = [&]() {
        MPI_Bcast(&nkeys, 1, MPI_INT, 0, comm);
        MPI_Bcast(keys.data(), msg_nkeys, MPI_UINT32_T, 0, comm);
        MPI_Bcast(&threshold, 1, MPI_FLOAT, 0, comm);
        MPI_Bcast(&candidate_limit, 1, MPI_INT, 0, comm);
        MPI_Bcast(q.data(), dim, MPI_FLOAT, 0, comm);
        MPI_Bcast(adc.data(), adc_n, MPI_FLOAT, 0, comm);
        double meta[2] = {0.0, 0.0};
        std::vector<double> all;
        if (rank == 0) all.resize((size_t)nranks * 2);
        MPI_Gather(meta, 2, MPI_DOUBLE,
                   rank == 0 ? all.data() : nullptr, 2, MPI_DOUBLE, 0, comm);
        std::vector<IdDist> empty, out_R, out_C;
        mpi_gather_iddist(comm, rank, empty, out_R, nullptr);
        mpi_gather_iddist(comm, rank, empty, out_C, nullptr);
    };

    for (int i = 0; i < n_warmup; i++) one_hop();
    MPI_Barrier(comm);

    std::vector<double> samples_us;
    if (rank == 0) samples_us.reserve((size_t)n_iters);
    for (int i = 0; i < n_iters; i++) {
        Clock::time_point t0;
        if (rank == 0) t0 = Clock::now();
        one_hop();
        if (rank == 0) {
            double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            samples_us.push_back(us);
        }
    }
    MPI_Barrier(comm);

    if (rank == 0 && !samples_us.empty()) {
        double sum = 0;
        for (double x : samples_us) sum += x;
        out.mean_us = sum / (double)samples_us.size();
        out.p50_us = percentile_sorted(samples_us, 0.50);
        out.p99_us = percentile_sorted(samples_us, 0.99);
        out.min_us = *std::min_element(samples_us.begin(), samples_us.end());
        out.max_us = *std::max_element(samples_us.begin(), samples_us.end());
    }
    return out;
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
    // If profile != nullptr and profile->enabled, fill timing (rank 0 aggregates).
    std::vector<uint32_t> search(MPI_Comm comm, int rank, int nranks,
                                 const float* q,
                                 uint64_t* stats_nodes_scored = nullptr,
                                 QueryProfile* profile = nullptr) const
    {
        using Clock = std::chrono::steady_clock;
        const bool prof = (profile != nullptr && profile->enabled);

        std::vector<uint32_t> result;
        std::vector<IdDist> H_R, H_C;
        std::unordered_map<uint32_t, char> visited;
        uint64_t nodes_scored = 0;

        if (prof && rank == 0) profile->begin_query();

        std::vector<float> q_local((size_t)pq.dim);
        if (rank == 0) std::copy(q, q + pq.dim, q_local.begin());
        MPI_Bcast(q_local.data(), pq.dim, MPI_FLOAT, 0, comm);

        std::vector<float> adc_table;
        if (rank == 0) {
            auto t0 = Clock::now();
            pq.adc_table(q_local.data(), adc_table);
            if (prof)
                profile->current.adc_s =
                    std::chrono::duration<double>(Clock::now() - t0).count();
        }

        if (rank == 0 && head) {
            auto t0 = Clock::now();
            auto res = head->searchKnn(q_local.data(), (size_t)params.k_head);
            while (!res.empty()) {
                auto top = res.top();
                res.pop();
                H_C.push_back({(uint32_t)top.second, top.first});
            }
            dedupe_best(H_C);
            if ((int)H_C.size() > params.L) H_C.resize((size_t)params.L);
            if (prof)
                profile->current.head_s =
                    std::chrono::duration<double>(Clock::now() - t0).count();
        }

        std::vector<uint32_t> keys((size_t)params.BW);
        std::vector<IdDist> batch_R, batch_C;

        // Tell non-root whether profiling is on (so they participate in Gather of doubles).
        int prof_i = prof ? 1 : 0;
        MPI_Bcast(&prof_i, 1, MPI_INT, 0, comm);
        const bool prof_all = (prof_i != 0);

        for (int hop = 0; hop < params.H; hop++) {
            int nkeys = 0;
            float t   = std::numeric_limits<float>::infinity();
            if (rank == 0) {
                if (!H_C.empty()) {
                    t = candidate_threshold(H_C, params.L);
                    nkeys = pick_beam(H_C, params.BW, visited, keys.data());
                }
            }

            // Hop wall starts at the first Bcast of this hop (outer nkeys).
            Clock::time_point hop_t0;
            HopSample hop_sample;
            if (prof_all && rank == 0) hop_t0 = Clock::now();

            MPI_Bcast(&nkeys, 1, MPI_INT, 0, comm);
            if (nkeys == 0) break;
            if (rank == 0) nodes_scored += (uint64_t)nkeys;

            // Non-root: dummy HopSample only to opt into Gather of score doubles.
            HopSample hop_dummy;
            HopSample* score_hop = nullptr;
            if (prof_all) score_hop = (rank == 0) ? &hop_sample : &hop_dummy;

            distributed_node_scoring(comm, rank, nranks, shard, pq, metric, pq.dim,
                                     keys.data(), nkeys, t, params.L,
                                     q_local.data(), adc_table,
                                     batch_R, batch_C,
                                     score_hop);

            if (rank == 0) {
                // Overwrite wall so it includes the outer nkeys Bcast (scoring
                // function also re-Bcasts nkeys — same as before).
                if (prof) {
                    hop_sample.wall_s =
                        std::chrono::duration<double>(Clock::now() - hop_t0).count();
                    hop_sample.comm_s = hop_sample.wall_s - hop_sample.score_max_s;
                    if (hop_sample.comm_s < 0) hop_sample.comm_s = 0;
                    hop_sample.bytes_sent += sizeof(int); // outer nkeys
                }

                auto tm0 = Clock::now();
                merge_cap(H_R, batch_R, params.k);
                merge_cap(H_C, batch_C, params.L);
                auto tm1 = Clock::now();
                if (prof) {
                    hop_sample.merge_s =
                        std::chrono::duration<double>(tm1 - tm0).count();

                    profile->current.hops_wall_s += hop_sample.wall_s;
                    profile->current.hops_comm_s += hop_sample.comm_s;
                    profile->current.hops_score_max_s += hop_sample.score_max_s;
                    profile->current.hops_score_mean_s += hop_sample.score_mean_s;
                    profile->current.hops_score_rank0_s += hop_sample.score_rank0_s;
                    profile->current.hops_score_others_max_s += hop_sample.score_others_max_s;
                    profile->current.hops_merge_s += hop_sample.merge_s;
                    profile->current.bytes_sent += hop_sample.bytes_sent;
                    profile->current.bytes_recv += hop_sample.bytes_recv;
                    profile->current.nhops += 1;
                    if (profile->store_hops)
                        profile->current.hops.push_back(hop_sample);
                }
            }
        }

        if (rank == 0) {
            if (stats_nodes_scored) *stats_nodes_scored = nodes_scored;
            if (prof) {
                profile->current.nodes_scored = nodes_scored;
                profile->end_query();
            }
            result.reserve((size_t)params.k);
            for (auto& e : H_R) result.push_back(e.id);
            if ((int)result.size() > params.k) result.resize((size_t)params.k);
        }
        return result;
    }
};

} // namespace dann
