// DistributedANN — query answering (paper §2.3–§2.4, Algorithms 1 & 2).
//
// Loads persistent index artifacts and runs distributed beam search:
//   head index -> H hops of node scoring on the sharded KV store.
//
// Run with mpirun (see script/run_query.sbatch):
//   mpirun -np W ./run_query
//       --index_dir <dir> --query_path <fvecs>
//       [--gt_path <lshbox>] [--k 10] [--L 100] [--BW 64] [--H 5]
//       [--k_head 50] [--ef_head 100] [--warmup 10] [--num_queries 0]
//       [--profile_comm 0|1] [--store_hops 0|1]
//
// Tunable parameters (recall/QPS trade-off, paper Table 1 uses H=5,BW=128,L=200):
//   k       final results returned / recall@k
//   L       candidate heap size
//   BW      beam width (nodes scored per hop)
//   H       graph hops (beam iterations)
//   k_head  head index search width
//   ef_head HNSW efSearch for head index
//   profile_comm  if 1, time head / hop wall / score / merge / bytes (mean+p99)

#include <mpi.h>
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
#include "query/orchestrator.hpp"

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
    int print_results = 0;
    int profile_comm = 0;  // 1 = enable hop/comm timing
    int store_hops = 1;    // keep per-hop samples when profiling
    int empty_hop_iters = 0;  // MPI floor microbench (0 = skip; set e.g. 2000)
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
        else if (key == "--print_results"){ a.print_results = std::stoi(val); i++; }
        else if (key == "--profile_comm") { a.profile_comm = std::stoi(val); i++; }
        else if (key == "--store_hops")   { a.store_hops = std::stoi(val); i++; }
        else if (key == "--empty_hop_iters") { a.empty_hop_iters = std::stoi(val); i++; }
    }
    return a;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    Args args = parse_args(argc, argv);
    if (args.index_dir.empty() || args.query_path.empty()) {
        if (rank == 0)
            fprintf(stderr,
                "Usage: mpirun -np W %s --index_dir <dir> --query_path <fvecs>\n"
                "       [--gt_path <lshbox>] [--k 10] [--L 100] [--BW 64] [--H 5]\n"
                "       [--k_head 50] [--ef_head 100] [--warmup 10] [--num_queries 0]\n"
                "       [--profile_comm 0|1] [--store_hops 0|1] [--empty_hop_iters N]\n",
                argv[0]);
        MPI_Finalize();
        return 1;
    }

    Orchestrator orch;
    orch.params.k       = args.k;
    orch.params.L       = std::max(args.L, std::max(args.k, args.BW));
    orch.params.BW      = args.BW;
    orch.params.H       = args.H;
    orch.params.k_head  = args.k_head;
    orch.params.ef_head = args.ef_head;

    std::string pq_path   = args.index_dir + "/pq_codebook.bin";
    std::string kv_path   = args.index_dir + "/nodes.bin";
    std::string head_path = args.index_dir + "/head/head.hnsw";

    Metric metric = Metric::L2;
    if (rank == 0) metric = read_index_metric(args.index_dir);
    int metric_i = (int)metric;
    MPI_Bcast(&metric_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    metric = (Metric)metric_i;

    orch.pq.load(pq_path);
    orch.pq.metric = metric;
    orch.metric = metric;
    orch.shard.load(kv_path, rank, nranks);

    int dim = orch.pq.dim;
    if (rank == 0) orch.load_head(head_path, dim, metric);

    // Rank 0 loads queries (+ optional ground truth).
    idx_t nq_total = 0;
    int qdim = 0;
    std::vector<float> queries;
    GroundTruth gt;
    bool have_gt = false;

    if (rank == 0) {
        if (args.query_path.size() >= 5 &&
            args.query_path.substr(args.query_path.size() - 5) == ".fbin") {
            idx_t n = 0;
            queries = read_fbin(args.query_path, n, qdim);
            nq_total = n;
        } else {
            nq_total = fvecs_count(args.query_path);
            queries = read_fvecs_slice(args.query_path, 0, nq_total, qdim);
        }
        if (qdim != dim) {
            fprintf(stderr, "Query dim=%d != index dim=%d\n", qdim, dim);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (metric == Metric::IP)
            normalize_rows(queries.data(), (int)nq_total, dim);
        if (!args.gt_path.empty()) {
            gt.load(args.gt_path);
            have_gt = true;
        }
        printf("[rank 0] Query answering: W=%d  N=%u  dim=%d  metric=%s\n",
               nranks, orch.shard.N_, dim, metric_name(metric));
        printf("[rank 0] params: k=%d L=%d BW=%d H=%d k_head=%d ef_head=%d profile_comm=%d\n",
               args.k, orch.params.L, args.BW, args.H, args.k_head, args.ef_head,
               args.profile_comm);
        printf("[rank 0] queries=%u  gt=%s\n", (unsigned)nq_total,
               have_gt ? args.gt_path.c_str() : "(none)");
        fflush(stdout);
    }

    idx_t nq = nq_total;
    int num_q = args.num_queries;
    MPI_Bcast(&num_q, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0 && num_q > 0) nq = std::min(nq_total, (idx_t)num_q);
    uint32_t nq_bcast = (uint32_t)nq;
    MPI_Bcast(&nq_bcast, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    nq = (idx_t)nq_bcast;

    // Warmup (synchronised). Profiling off during warmup.
    for (int w = 0; w < args.warmup; w++) {
        if (rank == 0) {
            orch.search(MPI_COMM_WORLD, rank, nranks,
                        queries.data() + (size_t)(w % (int)nq_total) * dim,
                        nullptr, nullptr);
        } else {
            orch.search(MPI_COMM_WORLD, rank, nranks, nullptr, nullptr, nullptr);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Optional empty-hop MPI floor (same message shape as BW keys, zero scoring).
    if (args.empty_hop_iters > 0) {
        int adc_n = orch.pq.m * orch.pq.ksub;
        EmptyHopFloor floor = measure_empty_hop_floor(
            MPI_COMM_WORLD, rank, nranks, dim, adc_n, args.BW,
            args.empty_hop_iters, /*warmup=*/50);
        if (rank == 0) print_empty_hop_floor(floor);
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        // Keep ranks aligned if some jobs skip: nothing to do.
    }

    QueryProfile profile;
    profile.enabled = (args.profile_comm != 0);
    profile.store_hops = (args.store_hops != 0);
    QueryProfile* prof_ptr = profile.enabled ? &profile : nullptr;

    // Timed queries.
    double total_recall = 0.0;
    uint64_t total_scored = 0;
    auto t0 = Clock::now();

    if (rank == 0) {
        for (idx_t qi = 0; qi < nq; qi++) {
            uint64_t scored = 0;
            auto res = orch.search(MPI_COMM_WORLD, rank, nranks,
                                   queries.data() + (size_t)qi * dim,
                                   &scored, prof_ptr);
            total_scored += scored;

            if (have_gt && qi < (idx_t)gt.num_queries)
                total_recall += recall_at_k(res, gt.gt[(size_t)qi], args.k);

            if (args.print_results) {
                printf("query %u ->", (unsigned)qi);
                for (auto id : res) printf(" %u", id);
                printf("\n");
            }
        }
    } else {
        for (idx_t qi = 0; qi < nq; qi++)
            orch.search(MPI_COMM_WORLD, rank, nranks, nullptr, nullptr, prof_ptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = Clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    if (rank == 0) {
        double qps = (nq > 0 && elapsed > 0) ? (double)nq / elapsed : 0.0;
        double recall = (have_gt && nq > 0) ? total_recall / nq : -1.0;
        printf("\n[rank 0] === Query results ===\n");
        printf("  queries     : %u\n", (unsigned)nq);
        printf("  elapsed_s   : %.2f\n", elapsed);
        printf("  latency avg : %.3f ms\n", (nq > 0 ? elapsed / nq * 1000.0 : 0.0));
        printf("  QPS         : %.1f\n", qps);
        printf("  nodes_scored_avg: %.1f  (total keys / query; label was formerly nodes/hop)\n",
               nq > 0 ? (double)total_scored / nq : 0.0);
        if (have_gt)
            printf("  recall@%d   : %.4f\n", args.k, recall);
        if (have_gt)
            printf("SWEEP_SUMMARY k=%d BW=%d H=%d L=%d nq=%u recall@%.0f=%.4f elapsed_s=%.2f qps=%.1f profile_comm=%d\n",
                   args.k, args.BW, args.H, orch.params.L, (unsigned)nq, (double)args.k, recall, elapsed, qps,
                   args.profile_comm);
        if (profile.enabled)
            print_profile_summary(profile, args.H, nranks);
        printf("[rank 0] === Done ===\n");
    }

    MPI_Finalize();
    return 0;
}
