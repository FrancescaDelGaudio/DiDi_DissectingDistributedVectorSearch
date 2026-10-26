// DistributedANN — index construction (Step 1: k-means + ε-replication)
//
// Run with mpirun:
//   mpirun -n W ./build_index
//       --base_path <fvecs>  --index_dir <dir>  [options]
//
// Outputs written to <index_dir>/:
//   meta.json          — index metadata (parameters, N, dim)
//   centroids.fvecs    — K cluster centroids (standard fvecs)
//   assignments.bin    — CSR mapping: for each vector, its cluster list
//
// All outputs are persistent and reloaded by subsequent indexing steps
// without rerunning k-means.

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>
#include <sys/types.h>

#include "common/types.hpp"
#include "common/io.hpp"
#include "common/distance.hpp"
#include "common/metric.hpp"
#include "clustering/kmeans_mpi.hpp"
#include "clustering/replication.hpp"

using namespace dann;

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
struct Args {
    std::string base_path;          // input: base vectors (.fvecs)
    std::string index_dir;          // output: persistent index directory
    int   K            = 32;        // number of k-means clusters
    int   kmeans_iter  = 20;        // max Lloyd's iterations
    float kmeans_tol   = 1e-4f;     // centroid displacement convergence threshold
    float epsilon      = 0.2f;      // SPANN ε-replication threshold (L2 / fallback)
    int   max_replicas = 3;         // max clusters per vector (epsilon mode)
    int   replica_r    = 0;         // Pyramid top-r per centroid (IP); 0 = use epsilon
    std::string metric = "l2";      // l2 | ip (MIPS: normalise vectors, spherical k-means)
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: mpirun -n W %s \\\n"
        "    --base_path <fvecs>  --index_dir <dir>\n"
        "    [--K 32] [--kmeans_iter 20] [--kmeans_tol 1e-4]\n"
        "    [--epsilon 0.2] [--max_replicas 3] [--metric l2|ip]\n"
        "    [--replica_r 0]  (IP: Pyramid top-r per centroid, e.g. 300)\n",
        prog);
}

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc - 1; i++) {
        std::string k = argv[i];
        std::string v = argv[i + 1];
        if      (k == "--base_path")    { a.base_path    = v;             i++; }
        else if (k == "--index_dir")    { a.index_dir    = v;             i++; }
        else if (k == "--K")            { a.K            = std::stoi(v);  i++; }
        else if (k == "--kmeans_iter")  { a.kmeans_iter  = std::stoi(v);  i++; }
        else if (k == "--kmeans_tol")   { a.kmeans_tol   = std::stof(v);  i++; }
        else if (k == "--epsilon")      { a.epsilon       = std::stof(v); i++; }
        else if (k == "--max_replicas") { a.max_replicas  = std::stoi(v); i++; }
        else if (k == "--replica_r")    { a.replica_r     = std::stoi(v); i++; }
        else if (k == "--metric")       { a.metric        = v;          i++; }
    }
    return a;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
// Create a directory hierarchy (equivalent to mkdir -p).
static void mkdirp(const std::string& path) {
    std::string cmd = "mkdir -p \"" + path + "\"";
    if (system(cmd.c_str()) != 0)
        throw std::runtime_error("Failed to create directory: " + path);
}

// Write a minimal JSON metadata file (no external library needed).
static void write_meta_json(const std::string& path,
                             const Args& a,
                             idx_t N, int dim)
{
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) throw std::runtime_error("Cannot create: " + path);
    fprintf(fp,
        "{\n"
        "  \"N\": %u,\n"
        "  \"dim\": %d,\n"
        "  \"K\": %d,\n"
        "  \"epsilon\": %g,\n"
        "  \"max_replicas\": %d,\n"
        "  \"replica_r\": %d,\n"
        "  \"kmeans_iter\": %d,\n"
        "  \"kmeans_tol\": %g,\n"
        "  \"metric\": \"%s\",\n"
        "  \"base_path\": \"%s\",\n"
        "  \"index_dir\": \"%s\",\n"
        "  \"step_complete\": 1\n"
        "}\n",
        N, dim,
        a.K, a.epsilon, a.max_replicas, a.replica_r,
        a.kmeans_iter, (double)a.kmeans_tol,
        metric_name(parse_metric(a.metric)),
        a.base_path.c_str(), a.index_dir.c_str());
    fclose(fp);
}

// ---------------------------------------------------------------------------
// Compute rank's slice boundaries (ceiling-division partitioning).
// Ranks 0 .. (N%nranks)-1 own one extra vector to cover the remainder.
// ---------------------------------------------------------------------------
static void slice_bounds(idx_t N, int nranks, int rank,
                         idx_t& start, idx_t& count)
{
    idx_t base  = N / (idx_t)nranks;
    idx_t extra = N % (idx_t)nranks;
    if ((idx_t)rank < extra) {
        count = base + 1;
        start = (idx_t)rank * (base + 1);
    } else {
        count = base;
        start = extra * (base + 1) + ((idx_t)rank - extra) * base;
    }
}

// ---------------------------------------------------------------------------
// Gather per-rank assignment CSR arrays on rank 0 and write assignments.bin.
//
// Each rank sends:
//   - local_offsets[0 .. local_n-1]  (relative offsets within rank's data)
//   - local_data_flat[]              (cluster ids)
// Rank 0 reconstructs the global CSR and writes it.
// ---------------------------------------------------------------------------
static void gather_and_write_assignments(
    const std::string&                   out_path,
    const std::vector<std::vector<clust_t>>& local_assign,
    idx_t N, uint32_t K,
    int rank, int nranks,
    MPI_Comm comm)
{
    idx_t local_n = (idx_t)local_assign.size();

    // Build local CSR
    std::vector<uint32_t> local_offsets(local_n);
    std::vector<uint32_t> local_data_flat;
    uint32_t running = 0;
    for (idx_t i = 0; i < local_n; i++) {
        local_offsets[i] = running;
        for (clust_t c : local_assign[i]) local_data_flat.push_back(c);
        running += (uint32_t)local_assign[i].size();
    }
    uint32_t local_total = running;

    // --- Gather counts so rank 0 can allocate ---
    int iln  = (int)local_n;
    int ilt  = (int)local_total;
    std::vector<int> all_n(nranks), all_total(nranks);
    MPI_Gather(&iln,  1, MPI_INT, all_n.data(),     1, MPI_INT, 0, comm);
    MPI_Gather(&ilt,  1, MPI_INT, all_total.data(), 1, MPI_INT, 0, comm);

    if (rank == 0) {
        uint32_t global_total = 0;
        for (int r = 0; r < nranks; r++) global_total += (uint32_t)all_total[r];

        std::vector<uint32_t> global_offsets(N);   // one entry per vector (sentinel added later)
        std::vector<uint32_t> global_data(global_total);

        // Build Gatherv displacements
        std::vector<int> displs_off(nranks), displs_dat(nranks);
        displs_off[0] = 0; displs_dat[0] = 0;
        for (int r = 1; r < nranks; r++) {
            displs_off[r] = displs_off[r-1] + all_n[r-1];
            displs_dat[r] = displs_dat[r-1] + all_total[r-1];
        }

        // Gather offsets (local_n elements per rank — no sentinel sent)
        MPI_Gatherv(local_offsets.data(),  (int)local_n, MPI_UINT32_T,
                    global_offsets.data(), all_n.data(), displs_off.data(),
                    MPI_UINT32_T, 0, comm);

        // Gather data
        MPI_Gatherv(local_data_flat.data(), ilt, MPI_UINT32_T,
                    global_data.data(), all_total.data(), displs_dat.data(),
                    MPI_UINT32_T, 0, comm);

        // Fix up offsets: each rank's entries are relative to its own data
        // start; shift by the cumulative data offset of previous ranks.
        uint32_t data_off = 0, vec_off = 0;
        for (int r = 0; r < nranks; r++) {
            for (int i = 0; i < all_n[r]; i++)
                global_offsets[vec_off + i] += data_off;
            data_off += (uint32_t)all_total[r];
            vec_off  += (uint32_t)all_n[r];
        }

        // Write assignments.bin
        FILE* fp = fopen(out_path.c_str(), "wb");
        if (!fp) throw std::runtime_error("Cannot create: " + out_path);

        uint32_t magic   = ASSIGN_MAGIC;
        uint32_t version = ASSIGN_VERSION;
        uint32_t N32     = (uint32_t)N;
        fwrite(&magic,        4, 1,           fp);
        fwrite(&version,      4, 1,           fp);
        fwrite(&N32,          4, 1,           fp);
        fwrite(&K,            4, 1,           fp);
        fwrite(&global_total, 4, 1,           fp);
        fwrite(global_offsets.data(), 4, N,   fp);   // N offsets
        fwrite(&global_total,         4, 1,   fp);   // sentinel (offset[N])
        fwrite(global_data.data(),    4, global_total, fp);
        fclose(fp);

        printf("[rank 0] Written %s  (N=%u  total_assignments=%u  avg=%.2f)\n",
               out_path.c_str(), N32, global_total,
               (double)global_total / N32);
        fflush(stdout);
    } else {
        // Non-root ranks: just participate in Gatherv calls
        MPI_Gatherv(local_offsets.data(),  (int)local_n, MPI_UINT32_T,
                    nullptr, nullptr, nullptr, MPI_UINT32_T, 0, comm);
        MPI_Gatherv(local_data_flat.data(), ilt, MPI_UINT32_T,
                    nullptr, nullptr, nullptr, MPI_UINT32_T, 0, comm);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    Args args = parse_args(argc, argv);
    Metric metric;
    try { metric = parse_metric(args.metric); }
    catch (const std::exception& e) {
        if (rank == 0) fprintf(stderr, "Error: %s\n", e.what());
        MPI_Finalize();
        return 1;
    }

    if (args.base_path.empty() || args.index_dir.empty()) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    // ------------------------------------------------------------------
    // Broadcast N and dim from rank 0 (avoids all ranks opening the file
    // simultaneously just for the header).
    // ------------------------------------------------------------------
    idx_t N   = 0;
    int   dim = 0;
    if (rank == 0) {
        N   = fvecs_count(args.base_path);
        dim = fvecs_dim(args.base_path);
        printf("[rank 0] Dataset: N=%u  dim=%d  K=%d  nranks=%d  metric=%s\n",
               N, dim, args.K, nranks, metric_name(metric));
        printf("[rank 0] Params: epsilon=%.3f  max_replicas=%d  replica_r=%d  "
               "kmeans_iter=%d  tol=%g\n",
               args.epsilon, args.max_replicas, args.replica_r,
               args.kmeans_iter, (double)args.kmeans_tol);
        fflush(stdout);
    }
    MPI_Bcast(&N,   1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_INT,      0, MPI_COMM_WORLD);

    if (args.K > (int)N) {
        if (rank == 0) fprintf(stderr, "Error: K=%d > N=%u\n", args.K, N);
        MPI_Finalize();
        return 1;
    }

    // ------------------------------------------------------------------
    // Load this rank's slice of the dataset.
    // ------------------------------------------------------------------
    idx_t rank_start, local_n;
    slice_bounds(N, nranks, rank, rank_start, local_n);

    printf("[rank %d] Loading %u vectors [%u, %u)\n",
           rank, local_n, rank_start, rank_start + local_n);
    fflush(stdout);

    int read_dim = 0;
    std::vector<float> local_data =
        read_fvecs_slice(args.base_path, rank_start, local_n, read_dim);

    if (read_dim != dim) {
        fprintf(stderr, "[rank %d] Dimension mismatch: expected %d got %d\n",
                rank, dim, read_dim);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (metric == Metric::IP)
        normalize_rows(local_data.data(), (int)local_n, dim);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("[rank 0] All ranks loaded data.\n\n");

    // ------------------------------------------------------------------
    // Step 1a: Distributed k-means
    // ------------------------------------------------------------------
    auto km = kmeans_mpi(local_data.data(), (int)local_n, dim,
                          args.K, args.kmeans_iter, args.kmeans_tol,
                          MPI_COMM_WORLD, metric);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[rank 0] k-means done: %d iters  final_disp=%.6f\n\n",
               km.iters_run, km.final_displacement);
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Step 1b: replication (epsilon for L2; Pyramid top-r for IP when replica_r > 0)
    std::vector<std::vector<clust_t>> local_assign;
    if (metric == Metric::IP && args.replica_r > 0) {
        if (rank == 0) {
            printf("[rank 0] Running Pyramid top-r MIPS replication (r=%d per centroid)...\n",
                   args.replica_r);
            fflush(stdout);
        }
        local_assign = top_r_mips_replicate_mpi(
            local_data.data(), (int)local_n, rank_start, dim,
            km.centroids.data(), args.K, args.replica_r, MPI_COMM_WORLD);
    } else {
        if (rank == 0) {
            printf("[rank 0] Running epsilon-replication (eps=%.3f max_rep=%d)...\n",
                   args.epsilon, args.max_replicas);
            fflush(stdout);
        }
        local_assign = epsilon_replicate(local_data.data(), (int)local_n, dim,
                                         km.centroids.data(), args.K,
                                         args.epsilon, args.max_replicas,
                                         metric);
    }

    // Print per-rank stats
    uint64_t local_total = 0;
    for (auto& a : local_assign) local_total += a.size();
    uint64_t global_total_rep = 0;
    MPI_Reduce(&local_total, &global_total_rep, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[rank 0] Replication done: total=%lu  avg replicas/vec=%.2f\n\n",
               global_total_rep, (double)global_total_rep / N);
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Step 1c: Write persistent outputs
    // ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) mkdirp(args.index_dir);
    MPI_Barrier(MPI_COMM_WORLD);

    // centroids.fvecs — rank 0 only
    if (rank == 0) {
        std::string c_path = args.index_dir + "/centroids.fvecs";
        write_fvecs(c_path, km.centroids.data(), args.K, dim);
        printf("[rank 0] Written %s\n", c_path.c_str());
        fflush(stdout);
    }

    // assignments.bin — gathered on rank 0 and written there
    gather_and_write_assignments(
        args.index_dir + "/assignments.bin",
        local_assign, N, (uint32_t)args.K,
        rank, nranks, MPI_COMM_WORLD);

    // meta.json — rank 0 only (written last to signal step completion)
    if (rank == 0) {
        write_meta_json(args.index_dir + "/meta.json", args, N, dim);
        printf("[rank 0] Written %s\n", (args.index_dir + "/meta.json").c_str());
        printf("\n[rank 0] === Step 1 complete ===\n");
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
