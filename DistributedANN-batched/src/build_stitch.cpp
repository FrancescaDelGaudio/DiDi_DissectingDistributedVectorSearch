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
// Distributed plan (contiguous id ownership, no MPI vector shuffle):
//   - Rank r owns the global id range [start_r, start_r + count_r).
//   - Pass 1: scan ALL cluster files (adjacency only); accumulate neighbour
//     unions for owned ids.
//   - Collect the global ids whose full vectors are needed for RobustPrune
//     (owned node + candidates only when |union| > R).
//   - Pass 2: scan cluster files again and load ONLY those vectors from the
//     per-cluster payloads (no mmap of base.fvecs — avoids random GPFS I/O).
//   - RobustPrune owned nodes; write CSR with MPI-IO.
//
// --base_path is used only to read the fvecs dimension header (and as a
// sanity check against cluster dim). Vectors for prune come from clusters.
//
// Output: <index_dir>/global_graph.bin

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

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

// MPI_File_write_at takes an `int` count. At 100M scale a single write of
// neighbour bytes can exceed INT_MAX — chunk without changing graph logic.
static void mpi_file_write_at_bytes(MPI_File fh, MPI_Offset offset,
                                    const void* data, uint64_t nbytes,
                                    int rank, const char* label)
{
    const uint64_t kChunk = 256ull << 20;  // 256 MiB per call (< INT_MAX)
    const char* p = (const char*)data;
    uint64_t done = 0;
    int chunk_i = 0;
    if (nbytes > kChunk) {
        printf("[rank %d] MPI-IO %s: writing %lu bytes in <=%lu MiB chunks...\n",
               rank, label, (unsigned long)nbytes,
               (unsigned long)(kChunk >> 20));
        fflush(stdout);
    }
    while (done < nbytes) {
        uint64_t n = nbytes - done;
        if (n > kChunk) n = kChunk;
        MPI_File_write_at(fh, offset + (MPI_Offset)done,
                          (void*)(p + done), (int)n, MPI_BYTE, MPI_STATUS_IGNORE);
        done += n;
        chunk_i++;
        if (nbytes > kChunk) {
            printf("[rank %d] MPI-IO %s: chunk %d done (%lu / %lu bytes)\n",
                   rank, label, chunk_i, (unsigned long)done, (unsigned long)nbytes);
            fflush(stdout);
        }
    }
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

    int dim = 0;
    try {
        dim = fvecs_dim(args.base_path);  // header only — no full-file mmap
    } catch (const std::exception& e) {
        if (rank == 0) fprintf(stderr, "Error reading base dim: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Metric metric = read_index_metric(args.index_dir);

    if (rank == 0) {
        printf("[rank 0] Step 3 (stitching): N=%u K=%u dim=%d W=%d R=%d alpha=%.2f "
               "metric=%s  (vectors from cluster graphs, no base mmap)\n",
               N, K, dim, nranks, args.R, args.alpha, metric_name(metric));
        fflush(stdout);
    }

    uint32_t my_start, my_count;
    slice_bounds(N, nranks, rank, my_start, my_count);

    std::vector<std::vector<uint32_t>> unions((size_t)my_count);
    std::vector<uint16_t> ncontrib((size_t)my_count, 0);

    // ------------------------------------------------------------------
    // Pass 1: adjacency only — accumulate neighbour unions for owned ids.
    // ------------------------------------------------------------------
    uint32_t clusters_seen = 0;
    for (uint32_t c = 0; c < K; c++) {
        std::string path = args.index_dir + "/clusters/cluster_" + std::to_string(c) + ".graph";
        uint32_t cid, n, cdim, cR, med;
        std::vector<uint32_t> ids, offs, nbrs;
        bool ok = false;
        try {
            ok = read_cluster_adjacency(path, cid, n, cdim, cR, med, ids, offs, nbrs);
        } catch (const std::exception& e) {
            fprintf(stderr, "[rank %d] %s\n", rank, e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!ok) continue;
        if ((int)cdim != dim) {
            fprintf(stderr, "[rank %d] dim mismatch in %s: cluster=%u base=%d\n",
                    rank, path.c_str(), cdim, dim);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
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
    if (rank == 0) {
        printf("[rank 0] Pass 1 done: neighbour lists accumulated (clusters=%u).\n",
               clusters_seen);
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Dedup unions and collect global ids needed for RobustPrune.
    // ------------------------------------------------------------------
    std::unordered_set<uint32_t> need_vec;
    need_vec.reserve((size_t)my_count / 8);
    uint64_t nodes_need_prune = 0;

    for (uint32_t i = 0; i < my_count; i++) {
        uint32_t g = my_start + i;
        auto& u = unions[i];
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        u.erase(std::remove(u.begin(), u.end(), g), u.end());

        if ((int)u.size() > args.R) {
            nodes_need_prune++;
            need_vec.insert(g);
            for (uint32_t nb : u) need_vec.insert(nb);
        }
    }

    printf("[rank %d] owned_nodes=%u  need_RobustPrune=%lu  unique_vectors_to_load=%zu\n",
           rank, my_count, (unsigned long)nodes_need_prune, need_vec.size());
    fflush(stdout);

    // ------------------------------------------------------------------
    // Pass 2: load ONLY needed vectors from cluster graph payloads.
    // ------------------------------------------------------------------
    std::unordered_map<uint32_t, uint32_t> id_to_slot;
    id_to_slot.reserve(need_vec.size() * 2 + 1);
    std::vector<float> vec_store;
    vec_store.reserve(need_vec.size() * (size_t)dim);

    auto cache_put = [&](uint32_t gid, const float* src) {
        if (need_vec.find(gid) == need_vec.end()) return;
        if (id_to_slot.find(gid) != id_to_slot.end()) return;
        uint32_t slot = (uint32_t)id_to_slot.size();
        id_to_slot.emplace(gid, slot);
        vec_store.insert(vec_store.end(), src, src + dim);
    };

    if (!need_vec.empty()) {
        for (uint32_t c = 0; c < K; c++) {
            if (id_to_slot.size() >= need_vec.size()) break;  // all found
            std::string path = args.index_dir + "/clusters/cluster_" +
                               std::to_string(c) + ".graph";
            ClusterGraph cg;
            try {
                // Missing empty cluster: skip
                FILE* probe = fopen(path.c_str(), "rb");
                if (!probe) continue;
                fclose(probe);
                cg = read_cluster_graph(path);
            } catch (const std::exception& e) {
                fprintf(stderr, "[rank %d] pass2 %s\n", rank, e.what());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if ((int)cg.dim != dim) {
                fprintf(stderr, "[rank %d] dim mismatch pass2 in %s\n", rank, path.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            for (uint32_t u = 0; u < cg.n; u++)
                cache_put(cg.ids[u], cg.vectors.data() + (size_t)u * (size_t)dim);
        }
    }

    if (id_to_slot.size() != need_vec.size()) {
        fprintf(stderr,
                "[rank %d] FATAL: vector cache incomplete: have %zu / need %zu\n",
                rank, id_to_slot.size(), need_vec.size());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[rank 0] Pass 2 done: cluster vectors loaded for RobustPrune.\n");
        fflush(stdout);
    }
    printf("[rank %d] starting RobustPrune/CSR build...\n", rank);
    fflush(stdout);

    std::vector<float> norm_buf((size_t)dim);
    auto vec = [&](uint32_t id) -> const float* {
        auto it = id_to_slot.find(id);
        if (it == id_to_slot.end()) {
            fprintf(stderr, "[rank %d] FATAL: missing vector for id=%u\n", rank, id);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const float* raw = vec_store.data() + (size_t)it->second * (size_t)dim;
        if (metric == Metric::L2) return raw;
        std::copy(raw, raw + dim, norm_buf.data());
        normalize_inplace(norm_buf.data(), dim);
        return norm_buf.data();
    };

    // ------------------------------------------------------------------
    // RobustPrune each owned node; build local CSR.
    // ------------------------------------------------------------------
    std::vector<uint64_t> local_off((size_t)my_count);
    std::vector<uint32_t> local_nbr;
    local_nbr.reserve((size_t)my_count * (size_t)args.R / 2);

    uint64_t replicated = 0;
    uint64_t running = 0;
    uint64_t pruned_count = 0;
    std::vector<uint64_t> deg_prefix((size_t)my_count);

    const uint32_t progress_every =
        std::max<uint32_t>(1u, my_count / 20u);

    for (uint32_t i = 0; i < my_count; i++) {
        uint32_t g = my_start + i;
        if (ncontrib[i] > 1) replicated++;

        auto& u = unions[i];  // already deduped / self-removed

        std::vector<uint32_t> pruned;
        if ((int)u.size() <= args.R)
            pruned = u;
        else {
            pruned = robust_prune_global(g, u, vec, dim, args.R, args.alpha, metric);
            pruned_count++;
        }
        std::vector<uint32_t>().swap(unions[i]);

        deg_prefix[i] = running;
        for (uint32_t nb : pruned) local_nbr.push_back(nb);
        running += pruned.size();

        if ((i + 1) % progress_every == 0 || (i + 1) == my_count) {
            printf("[rank %d] prune progress %u/%u (%.1f%%)  edges_so_far=%lu  "
                   "nodes_pruned=%lu\n",
                   rank, i + 1, my_count,
                   100.0 * (double)(i + 1) / (double)my_count,
                   (unsigned long)running, (unsigned long)pruned_count);
            fflush(stdout);
        }
    }
    uint64_t local_edges = running;

    // Free vector cache before the (potentially large) MPI-IO phase.
    std::unordered_map<uint32_t, uint32_t>().swap(id_to_slot);
    std::vector<float>().swap(vec_store);
    std::unordered_set<uint32_t>().swap(need_vec);

    printf("[rank %d] prune done: local_edges=%lu  entering MPI_Exscan...\n",
           rank, (unsigned long)local_edges);
    fflush(stdout);

    uint64_t edge_base = 0;
    MPI_Exscan(&local_edges, &edge_base, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) edge_base = 0;
    printf("[rank %d] MPI_Exscan done (edge_base=%lu); entering MPI_Allreduce...\n",
           rank, (unsigned long)edge_base);
    fflush(stdout);
    uint64_t total_edges = 0;
    MPI_Allreduce(&local_edges, &total_edges, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    printf("[rank %d] MPI_Allreduce done.\n", rank);
    fflush(stdout);

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

    printf("[rank %d] writing global_graph.bin (local_edges=%lu edge_base=%lu)...\n",
           rank, (unsigned long)local_edges, (unsigned long)edge_base);
    fflush(stdout);

    std::string out = args.index_dir + "/global_graph.bin";
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, out.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    const MPI_Offset off_start = (MPI_Offset)GLOBAL_HDR_BYTES;
    const MPI_Offset nbr_start = off_start + (MPI_Offset)((uint64_t)N + 1) * 8;

    if (rank == 0) {
        uint32_t hdr32[6] = { GLOBAL_MAGIC, GLOBAL_VERSION, N, (uint32_t)dim,
                              (uint32_t)args.R, 0u };
        MPI_File_write_at(fh, 0, hdr32, 6 * 4, MPI_BYTE, MPI_STATUS_IGNORE);
        MPI_File_write_at(fh, 24, &total_edges, 8, MPI_BYTE, MPI_STATUS_IGNORE);
    }

    mpi_file_write_at_bytes(fh, off_start + (MPI_Offset)my_start * 8,
                            local_off.data(), (uint64_t)my_count * 8ull,
                            rank, "offsets");
    if (rank == nranks - 1)
        MPI_File_write_at(fh, off_start + (MPI_Offset)N * 8,
                          &total_edges, 8, MPI_BYTE, MPI_STATUS_IGNORE);

    if (local_edges > 0)
        mpi_file_write_at_bytes(fh, nbr_start + (MPI_Offset)edge_base * 4,
                                local_nbr.data(), local_edges * 4ull,
                                rank, "neighbors");

    MPI_File_close(&fh);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[rank 0] Written %s\n", out.c_str());
        printf("\n[rank 0] === Step 3 complete ===\n");
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
