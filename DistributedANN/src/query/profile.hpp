#pragma once

// Lightweight per-query / per-hop timing for communication vs scoring analysis.
// Enabled only when a QueryProfile* is passed into Orchestrator::search.
// No per-hop printf — callers accumulate and print a summary at the end.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

namespace dann {

struct HopSample {
    double wall_s       = 0;  // Bcast start → Gather(R,C) done (rank-0 clock)
    double score_max_s  = 0;  // max local score_local among ALL ranks (incl. rank 0)
    double score_mean_s = 0;  // mean local score_local among all ranks
    double score_min_s  = 0;
    double score_rank0_s = 0; // rank 0's own score_local (rank 0 is also a shard)
    double score_others_max_s = 0;  // max among ranks 1..W-1 (0 if W==1)
    double score_others_mean_s = 0;
    double comm_s       = 0;  // wall_s - score_max_s  (UPPER BOUND on "non-scoring")
    double merge_s      = 0;  // rank-0 heap merge after scoring
    uint64_t bytes_sent = 0;  // approx payload from rank 0 (Bcasts)
    uint64_t bytes_recv = 0;  // approx payload into rank 0 (Gathers)
    int      nkeys      = 0;
    int      local_keys_at_max = 0;  // keys scored on the slowest shard
    int      local_keys_rank0  = 0;
    int      score_max_rank    = -1; // which rank had score_max
};

struct QuerySample {
    double head_s = 0;          // HNSW head search (rank 0)
    double adc_s  = 0;          // ADC table build (rank 0)
    double hops_wall_s = 0;     // sum over hops
    double hops_comm_s = 0;
    double hops_score_max_s = 0;
    double hops_score_mean_s = 0;
    double hops_score_rank0_s = 0;
    double hops_score_others_max_s = 0;
    double hops_merge_s = 0;
    uint64_t bytes_sent = 0;
    uint64_t bytes_recv = 0;
    int      nhops = 0;
    uint64_t nodes_scored = 0;
    std::vector<HopSample> hops;  // optional detail; kept if store_hops
};

// Empty-hop MPI floor: same Bcast/Gather shape as a real hop with `msg_nkeys`
// keys, but zero scoring work (no score_local). Measures software+wire MPI cost.
struct EmptyHopFloor {
    int n_iters = 0;
    int msg_nkeys = 0;
    uint64_t bytes_bcast = 0;
    double mean_us = 0;
    double p50_us = 0;
    double p99_us = 0;
    double min_us = 0;
    double max_us = 0;
};

struct QueryProfile {
    bool enabled = true;
    bool store_hops = false;   // keep per-hop vectors (heavier); default off
    std::vector<QuerySample> samples;

    // Filled during search for the current query (rank 0).
    QuerySample current;

    void begin_query() {
        current = QuerySample{};
    }

    void end_query() {
        if (!enabled) return;
        if (!store_hops) current.hops.clear();
        samples.push_back(std::move(current));
        current = QuerySample{};
    }
};

inline double percentile_sorted(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    double idx = p * (double)(v.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

inline void print_ms_stats(const char* name, const std::vector<double>& ms) {
    if (ms.empty()) {
        printf("  %-22s : (no samples)\n", name);
        return;
    }
    double sum = std::accumulate(ms.begin(), ms.end(), 0.0);
    double mean = sum / (double)ms.size();
    double p50 = percentile_sorted(ms, 0.50);
    double p99 = percentile_sorted(ms, 0.99);
    double mx  = *std::max_element(ms.begin(), ms.end());
    printf("  %-22s : mean=%.3f  p50=%.3f  p99=%.3f  max=%.3f   ms/query\n",
           name, mean, p50, p99, mx);
}

inline void print_us_stats(const char* name, const std::vector<double>& us) {
    if (us.empty()) {
        printf("  %-22s : (no samples)\n", name);
        return;
    }
    double sum = std::accumulate(us.begin(), us.end(), 0.0);
    double mean = sum / (double)us.size();
    printf("  %-22s : mean=%.1f  p50=%.1f  p99=%.1f  min=%.1f  max=%.1f   µs/hop\n",
           name, mean, percentile_sorted(us, 0.50), percentile_sorted(us, 0.99),
           *std::min_element(us.begin(), us.end()),
           *std::max_element(us.begin(), us.end()));
}

inline void print_empty_hop_floor(const EmptyHopFloor& f) {
    printf("\n[profile] === Empty-hop MPI floor (no scoring; same Bcast/Gather shape) ===\n");
    printf("[profile] msg_nkeys=%d  iters=%d  bcast_payload≈%llu B/hop\n",
           f.msg_nkeys, f.n_iters, (unsigned long long)f.bytes_bcast);
    printf("  %-22s : mean=%.1f  p50=%.1f  p99=%.1f  min=%.1f  max=%.1f   µs/hop\n",
           "mpi_empty_hop", f.mean_us, f.p50_us, f.p99_us, f.min_us, f.max_us);
    printf("[profile] Compare to typical verbs SEND/WRITE RTT ~2–5 µs (one-sided WRITE even lower).\n");
    printf("[profile] If mpi_empty_hop ≫ 5 µs, a large fraction of hop_comm is MPI software/collective\n");
    printf("[profile] cost (and/or sync skew) — an upper-bound on what RDMA can remove is hop_comm,\n");
    printf("[profile] a lower-bound floor for 'must pay something' is closer to verbs latency;\n");
    printf("[profile] the empty-hop number is the practical MPI floor on THIS cluster/fabric.\n");
}

inline void print_profile_summary(const QueryProfile& prof, int H_param, int W) {
    const auto& S = prof.samples;
    if (S.empty()) {
        printf("[profile] no samples\n");
        return;
    }
    const size_t n = S.size();

    std::vector<double> head, adc, wall, comm, smax, smean, srank0, sothmax, merge;
    std::vector<double> bsent, brecv, nhops, nscored;
    std::vector<double> comm_frac, score_frac;
    head.reserve(n); adc.reserve(n); wall.reserve(n); comm.reserve(n);
    smax.reserve(n); smean.reserve(n); srank0.reserve(n); sothmax.reserve(n); merge.reserve(n);

    double sum_bytes_sent = 0, sum_bytes_recv = 0;
    for (const auto& q : S) {
        head.push_back(q.head_s * 1e3);
        adc.push_back(q.adc_s * 1e3);
        wall.push_back(q.hops_wall_s * 1e3);
        comm.push_back(q.hops_comm_s * 1e3);
        smax.push_back(q.hops_score_max_s * 1e3);
        smean.push_back(q.hops_score_mean_s * 1e3);
        srank0.push_back(q.hops_score_rank0_s * 1e3);
        sothmax.push_back(q.hops_score_others_max_s * 1e3);
        merge.push_back(q.hops_merge_s * 1e3);
        bsent.push_back((double)q.bytes_sent);
        brecv.push_back((double)q.bytes_recv);
        nhops.push_back((double)q.nhops);
        nscored.push_back((double)q.nodes_scored);
        sum_bytes_sent += (double)q.bytes_sent;
        sum_bytes_recv += (double)q.bytes_recv;
        if (q.hops_wall_s > 0) {
            comm_frac.push_back(100.0 * q.hops_comm_s / q.hops_wall_s);
            score_frac.push_back(100.0 * q.hops_score_max_s / q.hops_wall_s);
        }
    }

    printf("\n[profile] === Communication / scoring breakdown (n=%zu queries, W=%d, H_param=%d) ===\n",
           n, W, H_param);
    printf("[profile] hop_wall = first Bcast .. Gather(R,C) done (rank-0 clock).\n");
    printf("[profile] hop_comm = hop_wall - score_max   *** UPPER BOUND on non-scoring time ***\n");
    printf("[profile]   includes: MPI Bcast/Gather software, packing/unpacking on rank 0,\n");
    printf("[profile]   wait if a rank enters late, and true wire time. NOT 'network only'.\n");
    printf("[profile]   RDMA can at most remove a fraction of hop_comm (never more).\n");
    printf("[profile] Rank 0 is ALSO a KV shard: its score_local is in score_max / score_rank0.\n");
    printf("[profile] Shard times gathered as 2 doubles (score_s, local_keys). No barrier in hops.\n");
    print_ms_stats("head_search", head);
    print_ms_stats("adc_table", adc);
    print_ms_stats("hops_wall (sum)", wall);
    print_ms_stats("hops_comm (sum)", comm);
    print_ms_stats("hops_score_max (sum)", smax);
    print_ms_stats("hops_score_mean (sum)", smean);
    print_ms_stats("hops_score_rank0 (sum)", srank0);
    print_ms_stats("hops_score_oth_max (sum)", sothmax);
    print_ms_stats("hops_merge (sum)", merge);

    auto mean = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
    };
    printf("  %-22s : mean=%.2f  p99=%.2f  %% of hops_wall\n",
           "comm_fraction", mean(comm_frac), percentile_sorted(comm_frac, 0.99));
    printf("  %-22s : mean=%.2f  p99=%.2f  %% of hops_wall (score_max)\n",
           "score_max_fraction", mean(score_frac), percentile_sorted(score_frac, 0.99));
    printf("  %-22s : mean=%.1f  p99=%.1f\n",
           "nhops_used", mean(nhops), percentile_sorted(nhops, 0.99));
    printf("  %-22s : mean=%.1f  p99=%.1f  (total keys scored / query)\n",
           "nodes_scored", mean(nscored), percentile_sorted(nscored, 0.99));
    printf("  %-22s : mean=%.1f  p99=%.1f  KiB/query\n",
           "bytes_sent", mean(bsent) / 1024.0, percentile_sorted(bsent, 0.99) / 1024.0);
    printf("  %-22s : mean=%.1f  p99=%.1f  KiB/query\n",
           "bytes_recv", mean(brecv) / 1024.0, percentile_sorted(brecv, 0.99) / 1024.0);
    printf("  %-22s : total_sent=%.2f  total_recv=%.2f  MiB over all queries\n",
           "bytes_totals", sum_bytes_sent / (1024.0 * 1024.0), sum_bytes_recv / (1024.0 * 1024.0));

    std::vector<double> hop_wall, hop_comm, hop_smax, hop_smean, hop_smin, hop_merge;
    std::vector<double> hop_s0, hop_soth, hop_bs, hop_br, hop_imb;
    int n_max_is_rank0 = 0, n_hop_samples = 0;
    for (const auto& q : S) {
        if (!q.hops.empty()) {
            for (const auto& h : q.hops) {
                hop_wall.push_back(h.wall_s * 1e3);
                hop_comm.push_back(h.comm_s * 1e3);
                hop_smax.push_back(h.score_max_s * 1e3);
                hop_smean.push_back(h.score_mean_s * 1e3);
                hop_smin.push_back(h.score_min_s * 1e3);
                hop_s0.push_back(h.score_rank0_s * 1e3);
                hop_soth.push_back(h.score_others_max_s * 1e3);
                hop_merge.push_back(h.merge_s * 1e3);
                hop_bs.push_back((double)h.bytes_sent);
                hop_br.push_back((double)h.bytes_recv);
                if (h.score_mean_s > 0)
                    hop_imb.push_back(h.score_max_s / h.score_mean_s);
                n_hop_samples++;
                if (h.score_max_rank == 0) n_max_is_rank0++;
            }
        } else if (q.nhops > 0) {
            double inv = 1.0 / (double)q.nhops;
            hop_wall.push_back(q.hops_wall_s * inv * 1e3);
            hop_comm.push_back(q.hops_comm_s * inv * 1e3);
            hop_smax.push_back(q.hops_score_max_s * inv * 1e3);
            hop_smean.push_back(q.hops_score_mean_s * inv * 1e3);
            hop_s0.push_back(q.hops_score_rank0_s * inv * 1e3);
            hop_soth.push_back(q.hops_score_others_max_s * inv * 1e3);
            hop_merge.push_back(q.hops_merge_s * inv * 1e3);
            hop_bs.push_back((double)q.bytes_sent * inv);
            hop_br.push_back((double)q.bytes_recv * inv);
        }
    }
    printf("[profile] --- Per-hop (ms/hop; averaged over used hops) ---\n");
    print_ms_stats("hop_wall", hop_wall);
    print_ms_stats("hop_comm (upper)", hop_comm);
    print_ms_stats("hop_score_max", hop_smax);
    print_ms_stats("hop_score_mean", hop_smean);
    print_ms_stats("hop_score_rank0", hop_s0);
    print_ms_stats("hop_score_oth_max", hop_soth);
    if (!hop_smin.empty()) print_ms_stats("hop_score_min", hop_smin);
    print_ms_stats("hop_merge", hop_merge);
    if (!hop_imb.empty()) {
        printf("  %-22s : mean=%.3f  p99=%.3f  (score_max/score_mean)\n",
               "score_imbalance", mean(hop_imb), percentile_sorted(hop_imb, 0.99));
    }
    if (n_hop_samples > 0) {
        printf("  %-22s : %d / %d hops (%.1f%%)  — rank0 was the slowest scorer\n",
               "score_max_is_rank0", n_max_is_rank0, n_hop_samples,
               100.0 * (double)n_max_is_rank0 / (double)n_hop_samples);
    }
    printf("  %-22s : mean=%.1f  p99=%.1f  B/hop\n",
           "hop_bytes_sent", mean(hop_bs), percentile_sorted(hop_bs, 0.99));
    printf("  %-22s : mean=%.1f  p99=%.1f  B/hop\n",
           "hop_bytes_recv", mean(hop_br), percentile_sorted(hop_br, 0.99));

    double cf = mean(comm_frac), sf = mean(score_frac);
    printf("[profile] --- RDMA hint (using hop_comm UPPER BOUND) ---\n");
    if (sf >= 70.0)
        printf("  score_max ≈ %.0f%% of hop wall → bottleneck is compute/imbalance; RDMA unlikely to help much.\n", sf);
    else if (cf >= 40.0)
        printf("  hop_comm ≈ %.0f%% of hop wall → non-scoring time is material; check empty-hop floor vs verbs.\n", cf);
    else
        printf("  mixed: hop_comm≈%.0f%% score_max≈%.0f%% of hop wall.\n", cf, sf);
    printf("[profile] === End profile ===\n");
}

} // namespace dann
