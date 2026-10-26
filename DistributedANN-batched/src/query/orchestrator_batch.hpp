#pragma once

// Batched / concurrent query engine (paper §2.4, §4.1).
//
// Unlike Orchestrator (single rank-0 orchestrator, one query at a time, every
// hop broadcast to all ranks), here every rank is both:
//   - an orchestrator for its own queries (qi % W == rank), with a replicated
//     head index, and
//   - a KV shard serving node-scoring requests from all orchestrators.
//
// All queries of a chunk are in flight together. Each round advances every
// active query by one hop:
//   1. orchestrator: threshold + pick beam, group keys by owner  (OpenMP / query)
//   2. Alltoallv of requests — only to owners of the keys (targeted fan-out);
//      the query vector travels once per (query, owner), on first contact
//   3. shard: ADC table built once per (origin rank, qid) and cached for the
//      chunk; score_local per request                           (OpenMP / request)
//   4. Alltoallv of responses (R, C per request)
//   5. orchestrator: dedupe + merge into H_R / H_C                (OpenMP / query)
// Per-query semantics match Orchestrator::search (same beam, threshold,
// dedupe_best and merge_cap), so result ids match run_query exactly.

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <climits>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <chrono>
#include <string>

#include "../quantize/pq.hpp"
#include "kv_shard.hpp"
#include "heaps.hpp"
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_ip.h"

namespace dann {

struct BatchParams {
    int k       = 10;
    int L       = 100;
    int BW      = 64;
    int H       = 5;
    int k_head  = 50;
    int ef_head = 100;
    int threads = 1;
    int chunk   = 0;   // queries in flight per rank; 0 = all local queries
    // Communication profiling: MPI_Barrier before each exchange, so that the
    // skew to the slowest rank lands in wait_*_s and comm_*_s is transfer only;
    // per-round message sizes are recorded for replay_floor().
    bool profile_comm = false;
};

// Per-rank accumulators. *_s are this rank's wall-time sums over all rounds.
// score_max_sum / score_mean_sum are cross-rank values (identical on all ranks).
struct BatchStats {
    int      chunks = 0;
    int      rounds = 0;
    double   head_s = 0, orch_s = 0, comm_req_s = 0, score_s = 0,
             pack_resp_s = 0, comm_resp_s = 0, merge_s = 0;
    double   wait_req_s = 0, wait_resp_s = 0;   // profile_comm only
    double   score_max_sum = 0, score_mean_sum = 0;
    double   imbalance_sum = 0;   // sum over rounds of score_max / score_mean
    int      imbalance_n = 0;
    uint64_t req_bytes_remote = 0, resp_bytes_remote = 0;
    uint64_t nodes_scored = 0;    // keys requested by this rank's queries
    uint64_t q_sent = 0;          // query vectors shipped (first contacts)
    uint64_t adc_built = 0;       // ADC tables built on this shard
    uint64_t requests_served = 0;
    // Per exchange, max(remote bytes sent, remote bytes received) on this rank:
    // what a single link must carry, for the wire-time lower bound.
    uint64_t bneck_req_bytes = 0, bneck_resp_bytes = 0;
    // profile_comm only: send counts of every exchange, for replay_floor().
    std::vector<std::vector<size_t>> rec_req, rec_resp;
};

// Empty-exchange floor: seconds per full pass over all recorded rounds.
struct FloorStats {
    int    iters = 0;
    double req_mean = 0, req_min = 0, resp_mean = 0, resp_min = 0;
};

namespace batch_detail {

struct ReqHdr  { uint32_t qid; int32_t nkeys; float thr; int32_t has_q; };
struct RespHdr { uint32_t qid; int32_t nR; int32_t nC; int32_t pad; };

struct QState {
    uint32_t qid = 0;
    const float* q = nullptr;
    std::vector<IdDist> HR, HC;
    std::unordered_map<uint32_t, char> visited;
    std::vector<uint8_t> q_sent;           // per owner: query vector already shipped
    // Current round.
    bool done = false;
    int  nkeys = 0;
    float thr = 0.f;
    std::vector<uint32_t> keys;            // beam, then regrouped by owner
    std::vector<int> dest_cnt, dest_first; // per owner: #keys, start in keys
    std::vector<uint8_t> dest_has_q;
    std::vector<size_t> dest_bytes, dest_off;
    // Responses of the current round: pointers into the receive buffer.
    std::vector<const uint8_t*> resp;
};

struct ServeReq {
    int src = 0;
    uint32_t qid = 0;
    int nkeys = 0;
    float thr = 0.f;
    const uint32_t* keys = nullptr;
    int slot = -1;
    std::vector<IdDist> R, C;
    size_t out_bytes = 0, out_off = 0;
};

inline double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline int to_int_or_abort(size_t v, const char* what) {
    if (v > (size_t)INT_MAX) {
        fprintf(stderr, "FATAL: %s = %zu bytes exceeds MPI int limit; lower --chunk\n",
                what, v);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    return (int)v;
}

// Alltoallv of byte buffers. send_counts[d] bytes to rank d, packed in send.
inline void exchange(MPI_Comm comm, int W,
                     const std::vector<size_t>& send_counts,
                     const std::vector<uint8_t>& send,
                     std::vector<int>& recv_counts, std::vector<int>& recv_displs,
                     std::vector<uint8_t>& recv)
{
    std::vector<int> sc((size_t)W), sd((size_t)W);
    size_t off = 0;
    for (int d = 0; d < W; d++) {
        sc[d] = to_int_or_abort(send_counts[d], "send count");
        sd[d] = to_int_or_abort(off, "send displacement");
        off += send_counts[d];
    }
    recv_counts.assign((size_t)W, 0);
    MPI_Alltoall(sc.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    recv_displs.assign((size_t)W, 0);
    size_t roff = 0;
    for (int s = 0; s < W; s++) {
        recv_displs[s] = to_int_or_abort(roff, "recv displacement");
        roff += (size_t)recv_counts[s];
    }
    recv.resize(roff);
    MPI_Alltoallv(send.data(), sc.data(), sd.data(), MPI_BYTE,
                  recv.data(), recv_counts.data(), recv_displs.data(), MPI_BYTE, comm);
}

inline uint64_t bottleneck_bytes(int rank, int W,
                                 const std::vector<size_t>& send_counts,
                                 const std::vector<int>& recv_counts)
{
    uint64_t out = 0, in = 0;
    for (int r = 0; r < W; r++) {
        if (r == rank) continue;
        out += (uint64_t)send_counts[(size_t)r];
        in  += (uint64_t)recv_counts[(size_t)r];
    }
    return std::max(out, in);
}

} // namespace batch_detail

class BatchEngine {
public:
    BatchParams params;
    Metric metric = Metric::L2;
    ProductQuantizer pq;
    KVShard shard;
    hnswlib::SpaceInterface<float>* head_space = nullptr;
    hnswlib::HierarchicalNSW<float>* head = nullptr;

    ~BatchEngine() {
        delete head;
        delete head_space;
    }

    void load_head(const std::string& head_path, int dim, Metric metric_) {
        delete head;
        delete head_space;
        metric = metric_;
        if (metric == Metric::IP)
            head_space = new hnswlib::InnerProductSpace((size_t)dim);
        else
            head_space = new hnswlib::L2Space((size_t)dim);
        head = new hnswlib::HierarchicalNSW<float>(head_space, head_path);
        head->setEf(params.ef_head);
    }

    // Collective over comm. qids = this rank's queries (global ids), rows of
    // `queries` (row-major, dim floats). results[i] = top-k ids of qids[i].
    void run(MPI_Comm comm, int rank, int W,
             const std::vector<uint32_t>& qids, const float* queries, int dim,
             std::vector<std::vector<uint32_t>>& results, BatchStats& st)
    {
        results.assign(qids.size(), {});
        long long nloc = (long long)qids.size(), nmax = 0;
        MPI_Allreduce(&nloc, &nmax, 1, MPI_LONG_LONG, MPI_MAX, comm);
        const long long chunk = params.chunk > 0 ? params.chunk : std::max(nmax, 1LL);
        const long long nchunks = (nmax + chunk - 1) / chunk;

        for (long long c = 0; c < nchunks; c++) {
            size_t b = (size_t)std::min(nloc, c * chunk);
            size_t e = (size_t)std::min(nloc, (c + 1) * chunk);
            run_chunk(comm, rank, W, qids, b, e, queries, dim, results, st);
            st.chunks++;
        }
    }

    // Collective. Replays every recorded exchange (same per-destination sizes,
    // zero-filled payload, no compute) `iters` times, each preceded by a Barrier.
    // This is the MPI cost of moving those bytes on this cluster: the floor
    // that a hand-written RDMA layer would have to beat.
    static FloorStats replay_floor(MPI_Comm comm, int W, const BatchStats& st, int iters)
    {
        using namespace batch_detail;
        FloorStats f;
        f.iters = iters;
        const size_t nr = std::min(st.rec_req.size(), st.rec_resp.size());
        size_t maxb = 0;
        auto total = [](const std::vector<size_t>& v) {
            size_t t = 0;
            for (size_t x : v) t += x;
            return t;
        };
        for (size_t r = 0; r < nr; r++)
            maxb = std::max(maxb, std::max(total(st.rec_req[r]), total(st.rec_resp[r])));
        std::vector<uint8_t> send(maxb, 0), recv;
        std::vector<int> rc, rd;
        double rsum = 0, ssum = 0;
        double rmin = std::numeric_limits<double>::infinity(), smin = rmin;
        for (int it = 0; it < iters; it++) {
            double a = 0, b = 0;
            for (size_t r = 0; r < nr; r++) {
                MPI_Barrier(comm);
                double t = now_s();
                exchange(comm, W, st.rec_req[r], send, rc, rd, recv);
                a += now_s() - t;
                MPI_Barrier(comm);
                t = now_s();
                exchange(comm, W, st.rec_resp[r], send, rc, rd, recv);
                b += now_s() - t;
            }
            rsum += a; ssum += b;
            rmin = std::min(rmin, a); smin = std::min(smin, b);
        }
        if (iters > 0) {
            f.req_mean = rsum / iters;  f.req_min = rmin;
            f.resp_mean = ssum / iters; f.resp_min = smin;
        }
        return f;
    }

    // Writes the recorded exchanges (profile_comm only) as a transport-replay trace for
    // exp_eval/comm_replay/transport_replay: per round the request and response Alltoallv
    // matrices (bytes, row = sender) and the Allreduce of active queries. Rank 0 writes.
    static void dump_sizes(MPI_Comm comm, int rank, int W, const BatchStats& st, const std::string& path)
    {
        const unsigned long long nr = std::min(st.rec_req.size(), st.rec_resp.size());
        unsigned long long nr_min = 0, nr_max = 0;
        MPI_Allreduce(&nr, &nr_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
        MPI_Allreduce(&nr, &nr_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
        if (nr_min != nr_max) {
            if (rank == 0) fprintf(stderr, "[dump_sizes] rounds differ across ranks (%llu..%llu), not written\n", nr_min, nr_max);
            return;
        }
        const size_t per_rank = 2 * (size_t)nr * (size_t)W;
        std::vector<unsigned long long> mine(per_rank), all;
        for (size_t r = 0; r < nr; r++)
            for (int d = 0; d < W; d++) {
                mine[(2 * r) * W + d] = st.rec_req[r][(size_t)d];
                mine[(2 * r + 1) * W + d] = st.rec_resp[r][(size_t)d];
            }
        if (rank == 0) all.resize(per_rank * (size_t)W);
        MPI_Gather(mine.data(), (int)per_rank, MPI_UNSIGNED_LONG_LONG,
                   all.data(), (int)per_rank, MPI_UNSIGNED_LONG_LONG, 0, comm);
        if (rank != 0) return;
        FILE* f = fopen(path.c_str(), "w");
        if (!f) { fprintf(stderr, "[dump_sizes] cannot write %s\n", path.c_str()); return; }
        fprintf(f, "# DA-batch exchanges, %llu rounds\nranks %d\n", nr, W);
        for (size_t r = 0; r < nr; r++) {
            for (int x = 0; x < 2; x++) {
                fprintf(f, "barrier %s\na2a\n", x == 0 ? "req" : "resp");
                for (int s = 0; s < W; s++) {
                    for (int d = 0; d < W; d++)
                        fprintf(f, "%s%llu", d ? " " : "", all[(size_t)s * per_rank + (2 * r + x) * W + d]);
                    fputc('\n', f);
                }
            }
            fprintf(f, "barrier sync\nallreduce 8\n");
        }
        fclose(f);
        printf("[dump_sizes] %llu rounds written to %s\n", nr, path.c_str());
    }

private:
    void run_chunk(MPI_Comm comm, int rank, int W,
                   const std::vector<uint32_t>& qids, size_t b, size_t e,
                   const float* queries, int dim,
                   std::vector<std::vector<uint32_t>>& results, BatchStats& st)
    {
        using namespace batch_detail;
        const int T = std::max(1, params.threads);
        const int nq = (int)(e - b);

        // ---- Orchestrator state + head search -----------------------------
        std::vector<QState> qs((size_t)nq);
        std::unordered_map<uint32_t, int> qidx;
        qidx.reserve((size_t)nq * 2);
        for (int i = 0; i < nq; i++) {
            QState& s = qs[(size_t)i];
            s.qid = qids[b + (size_t)i];
            s.q = queries + (size_t)s.qid * (size_t)dim;
            s.q_sent.assign((size_t)W, 0);
            s.keys.resize((size_t)params.BW);
            s.dest_cnt.assign((size_t)W, 0);
            s.dest_first.assign((size_t)W, 0);
            s.dest_has_q.assign((size_t)W, 0);
            s.dest_bytes.assign((size_t)W, 0);
            s.dest_off.assign((size_t)W, 0);
            qidx[s.qid] = i;
        }

        double t = now_s();
        #pragma omp parallel for schedule(dynamic, 4) num_threads(T)
        for (int i = 0; i < nq; i++) {
            QState& s = qs[(size_t)i];
            auto res = head->searchKnn(s.q, (size_t)params.k_head);
            while (!res.empty()) {
                auto top = res.top();
                res.pop();
                s.HC.push_back({(uint32_t)top.second, top.first});
            }
            dedupe_best(s.HC);
            if ((int)s.HC.size() > params.L) s.HC.resize((size_t)params.L);
        }
        st.head_s += now_s() - t;

        // ---- Shard-side cache: (origin rank, qid) -> q + ADC table ---------
        std::unordered_map<uint64_t, int> slot_of;
        std::vector<std::vector<float>> cache_q, cache_adc;

        std::vector<uint8_t> send, recv, rsend, rrecv;
        std::vector<int> rc, rd, rrc, rrd;

        for (int hop = 0; hop < params.H; hop++) {
            // ---- 1. Orchestrator: beam + group by owner --------------------
            t = now_s();
            #pragma omp parallel for schedule(dynamic, 4) num_threads(T)
            for (int i = 0; i < nq; i++) {
                QState& s = qs[(size_t)i];
                s.resp.clear();
                std::fill(s.dest_bytes.begin(), s.dest_bytes.end(), 0);
                if (s.done) { s.nkeys = 0; continue; }
                s.nkeys = 0;
                s.thr = std::numeric_limits<float>::infinity();
                if (!s.HC.empty()) {
                    s.thr = candidate_threshold(s.HC, params.L);
                    s.nkeys = pick_beam(s.HC, params.BW, s.visited, s.keys.data());
                }
                if (s.nkeys == 0) { s.done = true; continue; }

                // Stable regroup of the beam by owner (keeps beam order per owner).
                std::fill(s.dest_cnt.begin(), s.dest_cnt.end(), 0);
                for (int j = 0; j < s.nkeys; j++)
                    s.dest_cnt[KVShard::owner(s.keys[(size_t)j], W)]++;
                int acc = 0;
                for (int d = 0; d < W; d++) { s.dest_first[d] = acc; acc += s.dest_cnt[d]; }
                std::vector<int> fill(s.dest_first);
                std::vector<uint32_t> grouped((size_t)s.nkeys);
                for (int j = 0; j < s.nkeys; j++) {
                    uint32_t id = s.keys[(size_t)j];
                    grouped[(size_t)fill[KVShard::owner(id, W)]++] = id;
                }
                std::copy(grouped.begin(), grouped.end(), s.keys.begin());

                for (int d = 0; d < W; d++) {
                    if (s.dest_cnt[d] == 0) continue;
                    s.dest_has_q[d] = s.q_sent[d] ? 0 : 1;
                    s.dest_bytes[d] = sizeof(ReqHdr)
                                    + (size_t)s.dest_cnt[d] * sizeof(uint32_t)
                                    + (s.dest_has_q[d] ? (size_t)dim * sizeof(float) : 0);
                }
            }

            // Termination: stop the chunk when no rank has an active query.
            long long active = 0, active_all = 0;
            for (auto& s : qs) {
                active += s.nkeys;
                st.nodes_scored += (uint64_t)s.nkeys;
            }
            MPI_Allreduce(&active, &active_all, 1, MPI_LONG_LONG, MPI_SUM, comm);
            if (active_all == 0) { st.orch_s += now_s() - t; break; }

            // Pack requests: for each owner, queries in local order.
            std::vector<size_t> send_counts((size_t)W, 0);
            for (int d = 0; d < W; d++) {
                size_t off = 0;
                for (auto& s : qs) { s.dest_off[d] = off; off += s.dest_bytes[d]; }
                send_counts[d] = off;
            }
            std::vector<size_t> base((size_t)W, 0);
            for (int d = 1; d < W; d++) base[d] = base[d - 1] + send_counts[d - 1];
            send.resize(base[(size_t)W - 1] + send_counts[(size_t)W - 1]);

            #pragma omp parallel for schedule(dynamic, 16) num_threads(T)
            for (int i = 0; i < nq; i++) {
                QState& s = qs[(size_t)i];
                for (int d = 0; d < W; d++) {
                    if (s.dest_bytes[d] == 0) continue;
                    uint8_t* p = send.data() + base[d] + s.dest_off[d];
                    ReqHdr h{s.qid, s.dest_cnt[d], s.thr, s.dest_has_q[d]};
                    std::memcpy(p, &h, sizeof(h)); p += sizeof(h);
                    std::memcpy(p, s.keys.data() + s.dest_first[d],
                                (size_t)s.dest_cnt[d] * sizeof(uint32_t));
                    p += (size_t)s.dest_cnt[d] * sizeof(uint32_t);
                    if (s.dest_has_q[d]) {
                        std::memcpy(p, s.q, (size_t)dim * sizeof(float));
                        s.q_sent[d] = 1;
                    }
                }
            }
            for (int d = 0; d < W; d++) {
                if (d != rank) st.req_bytes_remote += send_counts[d];
                for (auto& s : qs) if (s.dest_bytes[d] && s.dest_has_q[d]) st.q_sent++;
            }
            st.orch_s += now_s() - t;

            // ---- 2. Request exchange ---------------------------------------
            if (params.profile_comm) {
                t = now_s();
                MPI_Barrier(comm);
                st.wait_req_s += now_s() - t;
                st.rec_req.push_back(send_counts);
            }
            t = now_s();
            exchange(comm, W, send_counts, send, rc, rd, recv);
            st.comm_req_s += now_s() - t;
            st.bneck_req_bytes += bottleneck_bytes(rank, W, send_counts, rc);

            // ---- 3. Shard: parse, ADC cache, score -------------------------
            t = now_s();
            std::vector<ServeReq> reqs;
            std::vector<int> new_slots;
            for (int src = 0; src < W; src++) {
                const uint8_t* p = recv.data() + rd[src];
                const uint8_t* end = p + rc[src];
                while (p < end) {
                    ReqHdr h;
                    std::memcpy(&h, p, sizeof(h)); p += sizeof(h);
                    ServeReq r;
                    r.src = src;
                    r.qid = h.qid;
                    r.nkeys = h.nkeys;
                    r.thr = h.thr;
                    r.keys = reinterpret_cast<const uint32_t*>(p);
                    p += (size_t)h.nkeys * sizeof(uint32_t);
                    uint64_t key = ((uint64_t)(uint32_t)src << 32) | h.qid;
                    if (h.has_q) {
                        int slot = (int)cache_q.size();
                        cache_q.emplace_back((const float*)p, (const float*)p + dim);
                        cache_adc.emplace_back();
                        slot_of[key] = slot;
                        new_slots.push_back(slot);
                        p += (size_t)dim * sizeof(float);
                        r.slot = slot;
                    } else {
                        r.slot = slot_of.at(key);
                    }
                    reqs.push_back(std::move(r));
                }
            }
            const int nnew = (int)new_slots.size();
            #pragma omp parallel for schedule(static) num_threads(T)
            for (int j = 0; j < nnew; j++) {
                int slot = new_slots[(size_t)j];
                pq.adc_table(cache_q[(size_t)slot].data(), cache_adc[(size_t)slot]);
            }
            st.adc_built += (uint64_t)nnew;

            const int nreq = (int)reqs.size();
            #pragma omp parallel for schedule(dynamic, 4) num_threads(T)
            for (int j = 0; j < nreq; j++) {
                ServeReq& r = reqs[(size_t)j];
                shard.score_local(r.keys, r.nkeys, r.thr, params.L,
                                  cache_q[(size_t)r.slot].data(),
                                  cache_adc[(size_t)r.slot], pq, metric, r.R, r.C);
                r.out_bytes = sizeof(RespHdr) + (r.R.size() + r.C.size()) * sizeof(IdDist);
            }
            st.requests_served += (uint64_t)nreq;
            double score_local_s = now_s() - t;
            st.score_s += score_local_s;

            // ---- 4. Response exchange (reqs are grouped by src already) ----
            t = now_s();
            std::vector<size_t> rsend_counts((size_t)W, 0);
            {
                size_t off = 0;
                for (auto& r : reqs) {
                    r.out_off = off;
                    off += r.out_bytes;
                    rsend_counts[(size_t)r.src] += r.out_bytes;
                }
                rsend.resize(off);
            }
            #pragma omp parallel for schedule(static) num_threads(T)
            for (int j = 0; j < nreq; j++) {
                ServeReq& r = reqs[(size_t)j];
                uint8_t* p = rsend.data() + r.out_off;
                RespHdr h{r.qid, (int32_t)r.R.size(), (int32_t)r.C.size(), 0};
                std::memcpy(p, &h, sizeof(h)); p += sizeof(h);
                std::memcpy(p, r.R.data(), r.R.size() * sizeof(IdDist));
                p += r.R.size() * sizeof(IdDist);
                std::memcpy(p, r.C.data(), r.C.size() * sizeof(IdDist));
            }
            for (int d = 0; d < W; d++)
                if (d != rank) st.resp_bytes_remote += rsend_counts[d];
            st.pack_resp_s += now_s() - t;
            if (params.profile_comm) {
                t = now_s();
                MPI_Barrier(comm);
                st.wait_resp_s += now_s() - t;
                st.rec_resp.push_back(rsend_counts);
            }
            t = now_s();
            exchange(comm, W, rsend_counts, rsend, rrc, rrd, rrecv);
            st.comm_resp_s += now_s() - t;
            st.bneck_resp_bytes += bottleneck_bytes(rank, W, rsend_counts, rrc);

            // ---- 5. Orchestrator: merge -------------------------------------
            t = now_s();
            for (int src = 0; src < W; src++) {
                const uint8_t* p = rrecv.data() + rrd[src];
                const uint8_t* end = p + rrc[src];
                while (p < end) {
                    RespHdr h;
                    std::memcpy(&h, p, sizeof(h));
                    qs[(size_t)qidx.at(h.qid)].resp.push_back(p);
                    p += sizeof(h) + (size_t)(h.nR + h.nC) * sizeof(IdDist);
                }
            }
            #pragma omp parallel for schedule(dynamic, 4) num_threads(T)
            for (int i = 0; i < nq; i++) {
                QState& s = qs[(size_t)i];
                if (s.resp.empty()) continue;
                std::vector<IdDist> bR, bC;
                for (const uint8_t* p : s.resp) {
                    RespHdr h;
                    std::memcpy(&h, p, sizeof(h));
                    const uint8_t* d = p + sizeof(h);
                    size_t oR = bR.size(), oC = bC.size();
                    bR.resize(oR + (size_t)h.nR);
                    bC.resize(oC + (size_t)h.nC);
                    std::memcpy(bR.data() + oR, d, (size_t)h.nR * sizeof(IdDist));
                    std::memcpy(bC.data() + oC, d + (size_t)h.nR * sizeof(IdDist),
                                (size_t)h.nC * sizeof(IdDist));
                }
                // Same post-gather steps as distributed_node_scoring on rank 0.
                dedupe_best(bR);
                dedupe_best(bC);
                if ((int)bC.size() > params.L) bC.resize((size_t)params.L);
                merge_cap(s.HR, bR, params.k);
                merge_cap(s.HC, bC, params.L);
            }
            st.merge_s += now_s() - t;

            // Cross-rank shard imbalance for this round.
            double mx = 0, sm = 0;
            MPI_Allreduce(&score_local_s, &mx, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(&score_local_s, &sm, 1, MPI_DOUBLE, MPI_SUM, comm);
            double mean = sm / (double)W;
            st.score_max_sum += mx;
            st.score_mean_sum += mean;
            if (mean > 0) { st.imbalance_sum += mx / mean; st.imbalance_n++; }
            st.rounds++;
        }

        for (int i = 0; i < nq; i++) {
            QState& s = qs[(size_t)i];
            std::vector<uint32_t>& out = results[b + (size_t)i];
            out.reserve((size_t)params.k);
            for (auto& x : s.HR) out.push_back(x.id);
            if ((int)out.size() > params.k) out.resize((size_t)params.k);
        }
    }
};

} // namespace dann
