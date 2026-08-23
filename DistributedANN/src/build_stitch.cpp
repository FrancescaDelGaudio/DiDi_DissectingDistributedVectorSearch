// DistributedANN — index construction (Step 3: global graph stitching)
//
// Prerequisite: Step 2 per-cluster graphs in <index_dir>/clusters/.
//
// Run with mpirun:
//   mpirun -n W ./build_stitch
//       --base_path <fvecs>  --index_dir <dir>  [--R 72 --alpha 1.2]
//
// A vector replicated across several clusters has one neighbour list per
// cluster it belongs to. Step 3 takes the UNION of those lists and applies
// RobustPrune to at most R, producing a single unified graph (paper §3, Fig 2).
//
// Distributed plan (contiguous id ownership, no MPI shuffle):
//   - Rank r owns the global id range [start_r, start_r + count_r).
//   - base.fvecs is mmap'd so any rank can read any vector by id.
//   - Each rank scans ALL cluster files; for every node whose global id it
//     owns, it appends that cluster's neighbour list to the node's union.
//   - Each owned node is RobustPruned (vectors via mmap) to <= R neighbours.
//   - The CSR graph is written collectively with MPI-IO.
//
// Output: <index_dir>/global_graph.bin

#include <mpi.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include "common/types.hpp"
#include "common/io.hpp"
#include "common/metric_io.hpp"
#include "common/metric.hpp"
#include "graph/prune.hpp"
#include "index/cluster_io.hpp"
#include "index/global_graph.hpp"

using namespace dann;

struct Args {
    std::string base_path;
    std::string index_dir;
    int   R     = 72;
    float alpha = 1.2f;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc - 1; i++) {
        std::string k = argv[i], v = argv[i + 1];
        if      (k == "--base_path") { a.base_path = v;          i++; }
        else if (k == "--index_dir") { a.index_dir = v;          i++; }
        else if (k == "--R")         { a.R     = std::stoi(v);   i++; }
        else if (k == "--alpha")     { a.alpha = std::stof(v);   i++; }
    }
    return a;
}

static void slice_bounds(uint32_t N, int nranks, int rank,
                         uint32_t& start, uint32_t& count) {
    uint32_t base = N / (uint32_t)nranks, extra = N % (uint32_t)nranks;
    if ((uint32_t)rank < extra) { count = base + 1; start = (uint32_t)rank * (base + 1); }
    else { count = base; start = extra * (base + 1) + ((uint32_t)rank - extra) * base; }
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
                            "--index_dir <dir> [--R 72] [--alpha 1.2]\n", argv[0]);
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

    if (rank == 0) {
        printf("[rank 0] Step 3 (stitching): N=%u K=%u dim=%d W=%d R=%d alpha=%.2f metric=%s\n",
               N, K, dim, nranks, args.R, args.alpha, metric_name(metric));
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // mmap base.fvecs for random vector access by global id.
    // ------------------------------------------------------------------
    int fd = open(args.base_path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[rank %d] cannot open base\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    struct stat st;
    fstat(fd, &st);
    void* base_map = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base_map == MAP_FAILED) { fprintf(stderr, "[rank %d] mmap failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    const char* base_bytes = (const char*)base_map;
    const size_t bytes_per_vec = 4 + (size_t)dim * 4;
    auto vec_raw = [&](uint32_t id) -> const float* {
        return (const float*)(base_bytes + (size_t)id * bytes_per_vec + 4);
    };
    std::vector<float> norm_buf((size_t)dim);
    auto vec = [&](uint32_t id) -> const float* {
        if (metric == Metric::L2) return vec_raw(id);
        std::copy(vec_raw(id), vec_raw(id) + dim, norm_buf.data());
        normalize_inplace(norm_buf.data(), dim);
        return norm_buf.data();
    };

    // ------------------------------------------------------------------
    // Ownership of a contiguous global-id range.
    // ------------------------------------------------------------------
    uint32_t my_start, my_count;
    slice_bounds(N, nranks, rank, my_start, my_count);

    std::vector<std::vector<uint32_t>> unions((size_t)my_count);
    std::vector<uint16_t> ncontrib((size_t)my_count, 0);

    // ------------------------------------------------------------------
    // Scan every cluster file; accumulate neighbour lists for owned ids.
    // ------------------------------------------------------------------
    uint32_t clusters_seen = 0;
    for (uint32_t c = 0; c < K; c++) {
        std::string path = args.index_dir + "/clusters/cluster_" + std::to_string(c) + ".graph";
        uint32_t cid, n, cdim, cR, med;
        std::vector<uint32_t> ids, offs, nbrs;
        bool ok = read_cluster_adjacency(path, cid, n, cdim, cR, med, ids, offs, nbrs);
        if (!ok) continue;            // empty/missing cluster
        clusters_seen++;

        for (uint32_t u = 0; u < n; u++) {
            uint32_t g = ids[u];
            if (g < my_start || g >= my_start + my_count) continue;
            uint32_t li = g - my_start;
            auto& dst = unions[li];
            for (uint32_t e = offs[u]; e < offs[u + 1]; e++)
                dst.push_back(nbrs[e]);
            ncontrib[li]++;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { printf("[rank 0] All ranks accumulated neighbour lists.\n"); fflush(stdout); }

    // ------------------------------------------------------------------
    // RobustPrune each owned node; build local CSR.
    // ------------------------------------------------------------------
    std::vector<uint64_t> local_off((size_t)my_count);  // global edge offsets
    std::vector<uint32_t> local_nbr;                    // flattened neighbours
    local_nbr.reserve((size_t)my_count * args.R / 2);

    uint64_t replicated = 0;
    uint64_t running = 0;
    std::vector<uint64_t> deg_prefix((size_t)my_count);

    for (uint32_t i = 0; i < my_count; i++) {
        uint32_t g = my_start + i;
        if (ncontrib[i] > 1) replicated++;

        // Deduplicate the union and drop self-loops.
        auto& u = unions[i];
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        u.erase(std::remove(u.begin(), u.end(), g), u.end());

        // Paper §3: take the union, then prune ONLY if it exceeds R.
        // Nodes in a single cluster (union <= R) keep their Vamana neighbours.
        std::vector<uint32_t> pruned;
        if ((int)u.size() <= args.R)
            pruned = u;
        else
            pruned = robust_prune_global(g, u, vec, dim, args.R, args.alpha, metric);
        std::vector<uint32_t>().swap(unions[i]);   // free early

        deg_prefix[i] = running;
        for (uint32_t nb : pruned) local_nbr.push_back(nb);
        running += pruned.size();
    }
    uint64_t local_edges = running;

    // Global edge offset (exclusive prefix sum) and grand total.
    uint64_t edge_base = 0;
    MPI_Exscan(&local_edges, &edge_base, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) edge_base = 0;
    uint64_t total_edges = 0;
    MPI_Allreduce(&local_edges, &total_edges, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    for (uint32_t i = 0; i < my_count; i++)
        local_off[i] = edge_base + deg_prefix[i];

    uint64_t global_replicated = 0;
    MPI_Reduce(&replicated, &global_replicated, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[rank 0] clusters read=%u  total_edges=%lu  avg_degree=%.1f\n",
               clusters_seen, (unsigned long)total_edges, (double)total_edges / N);
        printf("[rank 0] replicated nodes (in >1 cluster, merged): %lu (%.2f%%)\n",
               (unsigned long)global_replicated, 100.0 * global_replicated / N);
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Collective write of global_graph.bin (MPI-IO).
    // ------------------------------------------------------------------
    std::string out = args.index_dir + "/global_graph.bin";
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, out.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    const MPI_Offset off_start = (MPI_Offset)GLOBAL_HDR_BYTES;                 // offsets[] start
    const MPI_Offset nbr_start = off_start + (MPI_Offset)((uint64_t)N + 1) * 8; // neighbors[] start

    // Header (rank 0).
    if (rank == 0) {
        uint32_t hdr32[6] = { GLOBAL_MAGIC, GLOBAL_VERSION, N, (uint32_t)dim,
                              (uint32_t)args.R, 0u };
        MPI_File_write_at(fh, 0, hdr32, 6 * 4, MPI_BYTE, MPI_STATUS_IGNORE);
        MPI_File_write_at(fh, 24, &total_edges, 8, MPI_BYTE, MPI_STATUS_IGNORE);
    }

    // Offsets for owned nodes.
    MPI_File_write_at(fh, off_start + (MPI_Offset)my_start * 8,
                      local_off.data(), (int)(my_count * 8), MPI_BYTE,
                      MPI_STATUS_IGNORE);
    // Sentinel offsets[N] written by the last rank.
    if (rank == nranks - 1)
        MPI_File_write_at(fh, off_start + (MPI_Offset)N * 8,
                          &total_edges, 8, MPI_BYTE, MPI_STATUS_IGNORE);

    // Neighbour block.
    if (local_edges > 0)
        MPI_File_write_at(fh, nbr_start + (MPI_Offset)edge_base * 4,
                          local_nbr.data(), (int)(local_edges * 4), MPI_BYTE,
                          MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    munmap(base_map, st.st_size);
    close(fd);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[rank 0] Written %s\n", out.c_str());
        printf("\n[rank 0] === Step 3 complete ===\n");
    }
    MPI_Finalize();
    return 0;
}
