// DistributedANN — index construction (Step 2: per-cluster Vamana graphs)
//
// Prerequisite: Step 1 outputs in <index_dir> (centroids.fvecs, assignments.bin).
//
// Run with mpirun:
//   mpirun -n W ./build_graph
//       --base_path <fvecs>  --index_dir <dir>  [--R 72 --L 100 --alpha 1.2]
//
// Strategy B (shared GPFS read, no MPI vector shuffle):
//   - Every rank loads assignments.bin (the global cluster membership).
//   - Rank r owns clusters { c : c % W == r }.
//   - Each rank streams base.fvecs and keeps the vectors its clusters need.
//   - For each owned cluster, build a Vamana graph (global ids) and persist it.
//
// Outputs (one file per cluster):
//   <index_dir>/clusters/cluster_<c>.graph
//
// Each node stores {global id, full vector, neighbour global ids} so Step 3
// can merge the neighbour lists of vectors replicated across clusters.

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

#include "common/types.hpp"
#include "common/io.hpp"
#include "common/metric_io.hpp"
#include "common/metric.hpp"
#include "graph/vamana.hpp"
#include "index/cluster_io.hpp"

using namespace dann;

struct Args {
    std::string base_path;
    std::string index_dir;
    int   R       = 72;
    int   L       = 100;
    float alpha   = 1.2f;
    int   passes  = 2;
    int   threads = 0;        // 0 -> OpenMP default
    int   chunk   = 200000;   // streaming read chunk (vectors)
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: mpirun -n W %s \\\n"
        "    --base_path <fvecs>  --index_dir <dir>\n"
        "    [--R 72] [--L 100] [--alpha 1.2] [--passes 2] [--threads 0]\n",
        prog);
}

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc - 1; i++) {
        std::string k = argv[i], v = argv[i + 1];
        if      (k == "--base_path") { a.base_path = v;            i++; }
        else if (k == "--index_dir") { a.index_dir = v;            i++; }
        else if (k == "--R")         { a.R       = std::stoi(v);   i++; }
        else if (k == "--L")         { a.L       = std::stoi(v);   i++; }
        else if (k == "--alpha")     { a.alpha   = std::stof(v);   i++; }
        else if (k == "--passes")    { a.passes  = std::stoi(v);   i++; }
        else if (k == "--threads")   { a.threads = std::stoi(v);   i++; }
        else if (k == "--chunk")     { a.chunk   = std::stoi(v);   i++; }
    }
    return a;
}

static void mkdirp(const std::string& path) {
    std::string cmd = "mkdir -p \"" + path + "\"";
    if (system(cmd.c_str()) != 0)
        throw std::runtime_error("Failed to create directory: " + path);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    Args args = parse_args(argc, argv);
    if (args.base_path.empty() || args.index_dir.empty()) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    // ------------------------------------------------------------------
    // Load global cluster membership (every rank reads the full CSR).
    // ------------------------------------------------------------------
    std::vector<uint32_t> assign_off, assign_data;
    uint32_t N = 0, K = 0;
    try {
        read_assignments(args.index_dir + "/assignments.bin",
                         assign_off, assign_data, N, K);
    } catch (const std::exception& e) {
        if (rank == 0) fprintf(stderr, "Error: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dim = fvecs_dim(args.base_path);
    Metric metric = read_index_metric(args.index_dir);

    if (rank == 0) {
        printf("[rank 0] Step 2: N=%u  K=%u  dim=%d  W=%d  metric=%s\n",
               N, K, dim, nranks, metric_name(metric));
        printf("[rank 0] Vamana params: R=%d L=%d alpha=%.2f passes=%d\n",
               args.R, args.L, args.alpha, args.passes);
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Determine owned clusters (round-robin) and their member id lists.
    // ------------------------------------------------------------------
    std::vector<uint32_t> owned;
    for (uint32_t c = (uint32_t)rank; c < K; c += (uint32_t)nranks)
        owned.push_back(c);

    // cluster id -> list of global vector ids
    std::unordered_map<uint32_t, std::vector<idx_t>> members;
    members.reserve(owned.size() * 2);
    for (uint32_t c : owned) members[c];   // create empty lists

    std::vector<char> needed((size_t)N, 0);
    for (idx_t i = 0; i < N; i++) {
        for (uint32_t j = assign_off[i]; j < assign_off[i + 1]; j++) {
            uint32_t c = assign_data[j];
            auto it = members.find(c);
            if (it != members.end()) {
                it->second.push_back(i);
                needed[i] = 1;
            }
        }
    }

    size_t n_needed = 0;
    for (idx_t i = 0; i < N; i++) n_needed += needed[i];

    printf("[rank %d] owns %zu clusters, needs %zu vectors\n",
           rank, owned.size(), n_needed);
    fflush(stdout);

    // ------------------------------------------------------------------
    // Stream base.fvecs from GPFS; keep only needed vectors.
    // gid -> local storage row.
    // ------------------------------------------------------------------
    std::unordered_map<idx_t, uint32_t> gid2row;
    gid2row.reserve(n_needed * 2);
    std::vector<float> store((size_t)n_needed * dim);

    uint32_t row = 0;
    for (idx_t start = 0; start < N; start += (idx_t)args.chunk) {
        idx_t cnt = std::min((idx_t)args.chunk, N - start);
        int rd = 0;
        std::vector<float> buf = read_fvecs_slice(args.base_path, start, cnt, rd);
        for (idx_t i = 0; i < cnt; i++) {
            idx_t gid = start + i;
            if (!needed[gid]) continue;
            std::copy(buf.data() + (size_t)i * dim,
                      buf.data() + (size_t)i * dim + dim,
                      store.data() + (size_t)row * dim);
            gid2row[gid] = row;
            row++;
        }
    }

    if (metric == Metric::IP && row > 0)
        normalize_rows(store.data(), (int)row, dim);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { printf("[rank 0] All ranks loaded cluster vectors.\n"); fflush(stdout); }

    // ------------------------------------------------------------------
    // Build a Vamana graph per owned cluster and persist it.
    // ------------------------------------------------------------------
    std::string clusters_dir = args.index_dir + "/clusters";
    if (rank == 0) mkdirp(clusters_dir);
    MPI_Barrier(MPI_COMM_WORLD);

    VamanaParams vp;
    vp.R = args.R; vp.L = args.L; vp.alpha = args.alpha;
    vp.passes = args.passes; vp.threads = args.threads;

    for (uint32_t c : owned) {
        const std::vector<idx_t>& mem = members[c];
        uint32_t nc = (uint32_t)mem.size();
        if (nc == 0) {
            printf("[rank %d] cluster %u empty, skipping\n", rank, c);
            continue;
        }

        // Build contiguous local arrays for this cluster.
        std::vector<float>    cdata((size_t)nc * dim);
        std::vector<uint32_t> cids(nc);
        for (uint32_t i = 0; i < nc; i++) {
            idx_t gid = mem[i];
            cids[i] = (uint32_t)gid;
            uint32_t r = gid2row[gid];
            std::copy(store.data() + (size_t)r * dim,
                      store.data() + (size_t)r * dim + dim,
                      cdata.data() + (size_t)i * dim);
        }

        // Vamana build (local indices).
        Vamana vam(cdata.data(), (int)nc, dim, vp, metric);
        vam.build();

        // Translate adjacency (local -> global) for serialisation.
        std::vector<std::vector<uint32_t>> adj(nc);
        for (uint32_t u = 0; u < nc; u++) {
            uint32_t d = vam.degree(u);
            const uint32_t* nb = vam.neighbors(u);
            adj[u].reserve(d);
            for (uint32_t k = 0; k < d; k++)
                adj[u].push_back(cids[nb[k]]);
        }

        uint32_t medoid_global = cids[vam.medoid()];

        char fname[64];
        snprintf(fname, sizeof(fname), "/cluster_%u.graph", c);
        write_cluster_graph(clusters_dir + fname,
                            c, nc, (uint32_t)dim, (uint32_t)vam.R(),
                            medoid_global, cids.data(), cdata.data(), adj);

        printf("[rank %d] built cluster %u: n=%u medoid_gid=%u -> %s\n",
               rank, c, nc, medoid_global, (std::string("cluster_") +
               std::to_string(c) + ".graph").c_str());
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("\n[rank 0] === Step 2 complete ===\n");

    MPI_Finalize();
    return 0;
}
