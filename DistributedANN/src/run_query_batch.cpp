// DistributedANN — batched query answering (paper §2.4, §4.1).
//
// Same index artifacts and per-query algorithm as run_query, but:
//   - every rank loads the head index and orchestrates queries qi % W == rank,
//   - all queries of a chunk are in flight together (H synchronous rounds),
//   - node-scoring requests go only to the owners of the keys (Alltoallv),
//   - optional OpenMP over queries/requests (--threads).
// See query/orchestrator_batch.hpp.
//
//   mpirun -np W ./run_query_batch
//       --index_dir <dir> --query_path <fvecs|fbin>
//       [--gt_path <gt>] [--k 10] [--L 100] [--BW 64] [--H 5]
//       [--k_head 50] [--ef_head 100] [--warmup 10] [--num_queries 0]
//       [--threads 1] [--chunk 0] [--print_results 0]
//
//   threads  OpenMP threads per rank (head search, scoring, merge)
//   chunk    queries in flight per rank (0 = all local queries = nq/W)
//   QPS      = nq / max over ranks of query-phase wall (load excluded)

#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#include "common/io.hpp"
#include "common/bench.hpp"
#include "common/metric_io.hpp"
#include "common/metric.hpp"
#include "query/orchestrator_batch.hpp"

using namespace dann;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string index_dir;
    std::string query_path;
    std::string gt_path;
    int k       = 10;
    int L       = 100;
    int BW      = 64;
    int H       = 5;
    int k_head  = 50;
    int ef_head = 100;
    int warmup  = 10;
    int num_queries = 0;   // 0 = all
    int threads = 1;
    int chunk   = 0;       // 0 = all local queries in flight
    int print_results = 0;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc - 1; i++) {
        std::string key = argv[i], val = argv[i + 1];
        if      (key == "--index_dir")    { a.index_dir = val;           i++; }
        else if (key == "--query_path")   { a.query_path = val;          i++; }
        else if (key == "--gt_path")      { a.gt_path = val;             i++; }
        else if (key == "--k")            { a.k = std::stoi(val);        i++; }
        else if (key == "--L")            { a.L = std::stoi(val);        i++; }
        else if (key == "--BW")           { a.BW = std::stoi(val);       i++; }
        else if (key == "--H")            { a.H = std::stoi(val);        i++; }
        else if (key == "--k_head")       { a.k_head = std::stoi(val);   i++; }
        else if (key == "--ef_head")      { a.ef_head = std::stoi(val);  i++; }
        else if (key == "--warmup")       { a.warmup = std::stoi(val);   i++; }
        else if (key == "--num_queries")  { a.num_queries = std::stoi(val); i++; }
        else if (key == "--threads")      { a.threads = std::stoi(val);  i++; }
        else if (key == "--chunk")        { a.chunk = std::stoi(val);    i++; }
        else if (key == "--print_results"){ a.print_results = std::stoi(val); i++; }
    }
    return a;
}

static std::vector<float> load_queries(const std::string& path, idx_t& n, int& dim) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".fbin")
        return read_fbin(path, n, dim);
    n = fvecs_count(path);
    return read_fvecs_slice(path, 0, n, dim);
}

int main(int argc, char* argv[])
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, W;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &W);

    Args args = parse_args(argc, argv);
    if (args.index_dir.empty() || args.query_path.empty()) {
        if (rank == 0)
            fprintf(stderr,
                "Usage: mpirun -np W %s --index_dir <dir> --query_path <fvecs|fbin>\n"
                "       [--gt_path <gt>] [--k 10] [--L 100] [--BW 64] [--H 5]\n"
                "       [--k_head 50] [--ef_head 100] [--warmup 10] [--num_queries 0]\n"
                "       [--threads 1] [--chunk 0] [--print_results 0]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED && rank == 0)
        fprintf(stderr, "WARNING: MPI_THREAD_FUNNELED not provided (got %d)\n", provided);
    omp_set_num_threads(std::max(1, args.threads));

    auto t_load0 = Clock::now();
    BatchEngine eng;
    eng.params.k       = args.k;
    eng.params.L       = std::max(args.L, std::max(args.k, args.BW));
    eng.params.BW      = args.BW;
    eng.params.H       = args.H;
    eng.params.k_head  = args.k_head;
    eng.params.ef_head = args.ef_head;
    eng.params.threads = std::max(1, args.threads);
    eng.params.chunk   = args.chunk;

    std::string pq_path   = args.index_dir + "/pq_codebook.bin";
    std::string kv_path   = args.index_dir + "/nodes.bin";
    std::string head_path = args.index_dir + "/head/head.hnsw";

    Metric metric = Metric::L2;
    if (rank == 0) metric = read_index_metric(args.index_dir);
    int metric_i = (int)metric;
    MPI_Bcast(&metric_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    metric = (Metric)metric_i;

    eng.pq.load(pq_path);
    eng.pq.metric = metric;
    eng.metric = metric;
    eng.shard.load(kv_path, rank, W);
    const int dim = eng.pq.dim;
    eng.load_head(head_path, dim, metric);   // replicated on every rank

    // Every rank reads the (small) query file; rank 0 alone reads ground truth.
    idx_t nq_total = 0;
    int qdim = 0;
    std::vector<float> queries = load_queries(args.query_path, nq_total, qdim);
    if (qdim != dim) {
        if (rank == 0) fprintf(stderr, "Query dim=%d != index dim=%d\n", qdim, dim);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (metric == Metric::IP)
        normalize_rows(queries.data(), (int)nq_total, dim);

    GroundTruth gt;
    bool have_gt = false;
    if (rank == 0 && !args.gt_path.empty()) {
        gt.load(args.gt_path);
        have_gt = true;
    }

    idx_t nq = nq_total;
    if (args.num_queries > 0) nq = std::min(nq_total, (idx_t)args.num_queries);

    std::vector<uint32_t> my_qids;
    for (idx_t qi = (idx_t)rank; qi < nq; qi += (idx_t)W) my_qids.push_back((uint32_t)qi);

    MPI_Barrier(MPI_COMM_WORLD);
    double load_s = std::chrono::duration<double>(Clock::now() - t_load0).count();

    if (rank == 0) {
        printf("[rank 0] Batch query answering: W=%d  N=%u  dim=%d  metric=%s\n",
               W, eng.shard.N_, dim, metric_name(metric));
        printf("[rank 0] params: k=%d L=%d BW=%d H=%d k_head=%d ef_head=%d threads=%d chunk=%d (0=all, local=%zu)\n",
               args.k, eng.params.L, args.BW, args.H, args.k_head, args.ef_head,
               eng.params.threads, args.chunk, my_qids.size());
        printf("[rank 0] queries=%u  gt=%s  load_s=%.1f  mpi_thread=%d\n", (unsigned)nq,
               have_gt ? args.gt_path.c_str() : "(none)", load_s, provided);
        fflush(stdout);
    }

    // Warmup: first `warmup` queries overall, same engine path, not timed.
    if (args.warmup > 0) {
        std::vector<uint32_t> wq;
        for (uint32_t qi : my_qids) if ((int)qi < args.warmup) wq.push_back(qi);
        std::vector<std::vector<uint32_t>> wres;
        BatchStats wst;
        eng.run(MPI_COMM_WORLD, rank, W, wq, queries.data(), dim, wres, wst);
    }

    // Timed run.
    std::vector<std::vector<uint32_t>> res;
    BatchStats st;
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = Clock::now();
    eng.run(MPI_COMM_WORLD, rank, W, my_qids, queries.data(), dim, res, st);
    double local_s = std::chrono::duration<double>(Clock::now() - t0).count();
    MPI_Barrier(MPI_COMM_WORLD);

    double elapsed = 0;
    MPI_Reduce(&local_s, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to rank 0: per query [qid, id_0 .. id_{k-1}], padded.
    const int rec = 1 + args.k;
    std::vector<uint32_t> sendbuf(my_qids.size() * (size_t)rec, UINT32_MAX);
    for (size_t i = 0; i < my_qids.size(); i++) {
        sendbuf[i * rec] = my_qids[i];
        for (size_t j = 0; j < res[i].size() && (int)j < args.k; j++)
            sendbuf[i * rec + 1 + j] = res[i][j];
    }
    int scount = (int)sendbuf.size();
    std::vector<int> rcounts((size_t)W), rdispl((size_t)W);
    MPI_Gather(&scount, 1, MPI_INT, rcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<uint32_t> allbuf;
    if (rank == 0) {
        int off = 0;
        for (int r = 0; r < W; r++) { rdispl[r] = off; off += rcounts[r]; }
        allbuf.resize((size_t)off);
    }
    MPI_Gatherv(sendbuf.data(), scount, MPI_UINT32_T,
                allbuf.data(), rcounts.data(), rdispl.data(), MPI_UINT32_T,
                0, MPI_COMM_WORLD);

    // Cross-rank aggregates of the breakdown.
    double loc_t[6] = { st.head_s, st.orch_s, st.comm_req_s, st.score_s,
                        st.comm_resp_s, st.merge_s };
    double max_t[6], sum_t[6];
    MPI_Reduce(loc_t, max_t, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(loc_t, sum_t, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    unsigned long long loc_c[6] = { st.nodes_scored, st.req_bytes_remote,
                                    st.resp_bytes_remote, st.q_sent,
                                    st.adc_built, st.requests_served };
    unsigned long long sum_c[6];
    MPI_Reduce(loc_c, sum_c, 6, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<std::vector<uint32_t>> by_q((size_t)nq);
        for (size_t p = 0; p + (size_t)rec <= allbuf.size(); p += (size_t)rec) {
            uint32_t qid = allbuf[p];
            auto& v = by_q[qid];
            for (int j = 0; j < args.k; j++)
                if (allbuf[p + 1 + j] != UINT32_MAX) v.push_back(allbuf[p + 1 + j]);
        }
        double total_recall = 0;
        for (idx_t qi = 0; qi < nq; qi++) {
            if (have_gt && qi < (idx_t)gt.num_queries)
                total_recall += recall_at_k(by_q[qi], gt.gt[(size_t)qi], args.k);
            if (args.print_results) {
                printf("query %u ->", (unsigned)qi);
                for (auto id : by_q[qi]) printf(" %u", id);
                printf("\n");
            }
        }

        double qps = (nq > 0 && elapsed > 0) ? (double)nq / elapsed : 0.0;
        double recall = (have_gt && nq > 0) ? total_recall / nq : -1.0;
        double dnq = nq > 0 ? (double)nq : 1.0;
        printf("\n[rank 0] === Batch query results ===\n");
        printf("  queries     : %u\n", (unsigned)nq);
        printf("  W / threads / chunk : %d / %d / %d\n", W, eng.params.threads, args.chunk);
        printf("  elapsed_s   : %.2f  (max rank, query phase)\n", elapsed);
        printf("  QPS         : %.1f\n", qps);
        printf("  nodes_scored_avg: %.1f  (total keys / query)\n", (double)sum_c[0] / dnq);
        if (have_gt)
            printf("  recall@%d   : %.4f\n", args.k, recall);

        const char* names[6] = { "head_search", "orch_pick_pack", "comm_requests",
                                 "shard_score", "comm_responses", "orch_merge" };
        printf("[batch] --- Per-phase wall (s, summed over rounds): max rank | mean rank ---\n");
        for (int i = 0; i < 6; i++)
            printf("  %-16s : %8.3f | %8.3f\n", names[i], max_t[i], sum_t[i] / W);
        printf("[batch] rounds=%d  chunks=%d  score_imbalance(mean max/mean per round)=%.3f\n",
               st.rounds, st.chunks,
               st.imbalance_n > 0 ? st.imbalance_sum / st.imbalance_n : 0.0);
        printf("[batch] per-round shard score: max=%.3f ms  mean=%.3f ms\n",
               st.rounds > 0 ? 1e3 * st.score_max_sum / st.rounds : 0.0,
               st.rounds > 0 ? 1e3 * st.score_mean_sum / st.rounds : 0.0);
        printf("[batch] network bytes/query (remote only): requests=%.1f KiB  responses=%.1f KiB\n",
               (double)sum_c[1] / dnq / 1024.0, (double)sum_c[2] / dnq / 1024.0);
        printf("[batch] q vectors sent/query=%.2f  ADC tables built/query=%.2f  requests/query=%.2f\n",
               (double)sum_c[3] / dnq, (double)sum_c[4] / dnq, (double)sum_c[5] / dnq);
        if (have_gt)
            printf("SWEEP_SUMMARY k=%d BW=%d H=%d L=%d nq=%u recall@%d=%.4f elapsed_s=%.2f qps=%.1f mode=batch W=%d threads=%d chunk=%d\n",
                   args.k, args.BW, args.H, eng.params.L, (unsigned)nq, args.k, recall,
                   elapsed, qps, W, eng.params.threads, args.chunk);
        printf("[rank 0] === Done ===\n");
    }

    MPI_Finalize();
    return 0;
}
