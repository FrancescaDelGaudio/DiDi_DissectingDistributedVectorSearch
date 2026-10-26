// DistributedANN — index construction (Step 5: in-memory head index)
//
// Prerequisite: Step 2 per-cluster graphs in <index_dir>/clusters/.
//
// Run with mpirun:
//   mpirun -n W ./build_head
//       --base_path <fvecs> --index_dir <dir>
//       [--head_frac 0.05] [--M 16] [--efc 200] [--method random|bfs]
//
// Paper §3: the head index is built from "the union of the top layers of EACH
// partition's graph" (not the stitched graph) so the entire graph is reachable.
// Each Vamana partition is single-layer, so we approximate "top layers" by a
// per-cluster sample of representatives:
//   - random : uniform sample per cluster (best coverage; default)
//   - bfs    : BFS from the cluster medoid (central blob)
// Representatives (proportional to cluster size, summing to ~head_frac*N) are
// gathered on rank 0, which builds an HNSW (hnswlib) with labels = global ids
// and serialises it. At query time the head search returns global ids directly,
// used as entry points for beam search on the global graph.
//
// Output: <index_dir>/head/head.hnsw  and  <index_dir>/head/meta.json

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>
#include <queue>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

#include "common/types.hpp"
#include "common/io.hpp"
#include "common/metric_io.hpp"
#include "common/metric.hpp"
#include "index/cluster_io.hpp"

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_ip.h"

using namespace dann;

struct Args {
    std::string base_path;
    std::string index_dir;
    float head_frac = 0.05f;     // C / N  (paper: 2.5B / 50B = 5%)
    int   M         = 16;
    int   efc       = 200;
    std::string method = "random";   // "random" | "bfs"
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc - 1; i++) {
        std::string k = argv[i], v = argv[i + 1];
        if      (k == "--base_path") { a.base_path = v;            i++; }
        else if (k == "--index_dir") { a.index_dir = v;            i++; }
        else if (k == "--head_frac") { a.head_frac = std::stof(v); i++; }
        else if (k == "--M")         { a.M   = std::stoi(v);       i++; }
        else if (k == "--efc")       { a.efc = std::stoi(v);       i++; }
        else if (k == "--method")    { a.method = v;               i++; }
    }
    return a;
}

static void mkdirp(const std::string& path) {
    std::string cmd = "mkdir -p \"" + path + "\"";
    if (system(cmd.c_str()) != 0)
        throw std::runtime_error("Failed to create directory: " + path);
}

// Select up to `quota` local node indices from a cluster graph.
static std::vector<uint32_t> select_reps(const ClusterGraph& g,
                                         uint32_t quota,
                                         const std::string& method,
                                         std::mt19937& rng)
{
    std::vector<uint32_t> sel;
    if (quota >= g.n) {
        sel.resize(g.n);
        for (uint32_t i = 0; i < g.n; i++) sel[i] = i;
        return sel;
    }

    if (method == "bfs") {
        // BFS from the medoid over the cluster-local graph.
        std::unordered_map<uint32_t, uint32_t> g2l;
        g2l.reserve(g.n * 2);
        for (uint32_t i = 0; i < g.n; i++) g2l[g.ids[i]] = i;

        uint32_t start = 0;
        auto it = g2l.find(g.medoid_global);
        if (it != g2l.end()) start = it->second;

        std::vector<char> seen(g.n, 0);
        std::queue<uint32_t> q;
        q.push(start); seen[start] = 1;
        while (!q.empty() && sel.size() < quota) {
            uint32_t u = q.front(); q.pop();
            sel.push_back(u);
            for (uint32_t e = g.offsets[u]; e < g.offsets[u + 1]; e++) {
                auto jt = g2l.find(g.neighbors[e]);
                if (jt != g2l.end() && !seen[jt->second]) {
                    seen[jt->second] = 1;
                    q.push(jt->second);
                }
            }
        }
    } else {
        // Uniform random sample without replacement (reservoir-free: shuffle).
        std::vector<uint32_t> idx(g.n);
        for (uint32_t i = 0; i < g.n; i++) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), rng);
        sel.assign(idx.begin(), idx.begin() + quota);
    }
    return sel;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    Args args = parse_args(argc, argv);
    if (args.base_path.empty() || args.index_dir.empty()) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -n W %s --base_path <fvecs> "
                            "--index_dir <dir> [--head_frac 0.05] [--M 16] "
                            "[--efc 200] [--method random|bfs]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    uint32_t N = 0, K = 0;
    try {
        read_assignments_header(args.index_dir + "/assignments.bin", N, K);
    } catch (const std::exception& e) {
        if (rank == 0) fprintf(stderr, "Error: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dim = fvecs_dim(args.base_path);
    Metric metric = read_index_metric(args.index_dir);
    uint32_t C = (uint32_t)((double)args.head_frac * N);

    if (rank == 0) {
        printf("[rank 0] Step 5 (head index): N=%u K=%u dim=%d W=%d metric=%s\n",
               N, K, dim, nranks, metric_name(metric));
        printf("[rank 0] head_frac=%.3f -> C~=%u  method=%s  M=%d efC=%d\n",
               args.head_frac, C, args.method.c_str(), args.M, args.efc);
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Collect representatives from owned clusters.
    // ------------------------------------------------------------------
    std::vector<uint32_t> local_ids;
    std::vector<float>    local_vecs;
    std::mt19937 rng(1234u + (unsigned)rank);

    for (uint32_t c = (uint32_t)rank; c < K; c += (uint32_t)nranks) {
        std::string path = args.index_dir + "/clusters/cluster_" + std::to_string(c) + ".graph";
        ClusterGraph g;
        try { g = read_cluster_graph(path); }
        catch (...) { continue; }     // missing/empty cluster
        if (g.n == 0) continue;

        long long q_ll = std::llround((double)C * g.n / N);
        uint32_t quota = (uint32_t)std::max<long long>(1, q_ll);
        std::vector<uint32_t> sel = select_reps(g, quota, args.method, rng);

        for (uint32_t u : sel) {
            local_ids.push_back(g.ids[u]);
            const float* v = g.vectors.data() + (size_t)u * dim;
            local_vecs.insert(local_vecs.end(), v, v + dim);
        }
    }

    int local_count = (int)local_ids.size();
    printf("[rank %d] collected %d head representatives\n", rank, local_count);
    fflush(stdout);

    // ------------------------------------------------------------------
    // Gather all representatives on rank 0.
    // ------------------------------------------------------------------
    std::vector<int> counts(nranks), displs_id(nranks), displs_vec(nranks);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total = 0;
    std::vector<uint32_t> all_ids;
    std::vector<float>    all_vecs;
    if (rank == 0) {
        for (int r = 0; r < nranks; r++) total += counts[r];
        displs_id[0] = 0; displs_vec[0] = 0;
        for (int r = 1; r < nranks; r++) {
            displs_id[r]  = displs_id[r-1]  + counts[r-1];
            displs_vec[r] = displs_vec[r-1] + counts[r-1] * dim;
        }
        all_ids.resize(total);
        all_vecs.resize((size_t)total * dim);
    }

    std::vector<int> counts_vec(nranks);
    if (rank == 0) for (int r = 0; r < nranks; r++) counts_vec[r] = counts[r] * dim;

    MPI_Gatherv(local_ids.data(), local_count, MPI_UINT32_T,
                all_ids.data(), counts.data(), displs_id.data(), MPI_UINT32_T,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vecs.data(), local_count * dim, MPI_FLOAT,
                all_vecs.data(), counts_vec.data(), displs_vec.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Rank 0 builds and serialises the HNSW head index.
    // ------------------------------------------------------------------
    if (rank == 0) {
        printf("[rank 0] Building HNSW head index over %d vectors...\n", total);
        fflush(stdout);

        hnswlib::SpaceInterface<float>* space = nullptr;
        if (metric == Metric::IP)
            space = new hnswlib::InnerProductSpace((size_t)dim);
        else
            space = new hnswlib::L2Space((size_t)dim);
        {
            hnswlib::HierarchicalNSW<float> index(space, (size_t)total,
                                                  (size_t)args.M, (size_t)args.efc);

            #pragma omp parallel for schedule(dynamic, 256)
            for (int i = 0; i < total; i++)
                index.addPoint(all_vecs.data() + (size_t)i * dim,
                               (hnswlib::labeltype)all_ids[i]);

            mkdirp(args.index_dir + "/head");
            std::string hpath = args.index_dir + "/head/head.hnsw";
            index.saveIndex(hpath);
            printf("[rank 0] Written %s (%d vectors)\n", hpath.c_str(), total);

            FILE* fp = fopen((args.index_dir + "/head/meta.json").c_str(), "w");
            if (fp) {
                fprintf(fp,
                    "{\n  \"C\": %d,\n  \"dim\": %d,\n  \"M\": %d,\n"
                    "  \"efConstruction\": %d,\n  \"method\": \"%s\",\n"
                    "  \"metric\": \"%s\",\n"
                    "  \"head_frac\": %g\n}\n",
                    total, dim, args.M, args.efc, args.method.c_str(),
                    metric_name(metric), (double)args.head_frac);
                fclose(fp);
            }
        }
        delete space;

        printf("\n[rank 0] === Step 5 complete ===\n");
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
