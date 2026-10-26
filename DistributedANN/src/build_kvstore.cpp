// DistributedANN — index construction (Step 6: distributed KV store)
//
// Prerequisites: global_graph.bin (Step 3), codes.bin + pq_codebook.bin (Step 4).
//
// Run with mpirun:
//   mpirun -n W ./build_kvstore --base_path <fvecs> --index_dir <dir>
//
// Assembles the node table where each node stores {full vector, neighbour ids,
// neighbour PQ codes}. base.fvecs and codes.bin are mmap'd for random access by
// id; the unified graph adjacency is read in contiguous slices. Output is
// written id-ordered with MPI-IO so the query phase can shard it by hash(id)%W.
//
// Output: <index_dir>/nodes.bin

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
#include "quantize/pq.hpp"
#include "index/global_graph.hpp"
#include "index/kvstore_io.hpp"

using namespace dann;

static void slice_bounds(uint32_t N, int nranks, int rank,
                         uint32_t& start, uint32_t& count) {
    uint32_t base = N / (uint32_t)nranks, extra = N % (uint32_t)nranks;
    if ((uint32_t)rank < extra) { count = base + 1; start = (uint32_t)rank * (base + 1); }
    else { count = base; start = extra * (base + 1) + ((uint32_t)rank - extra) * base; }
}

// MPI_File_write_at with 64-bit length (chunked to stay within int counts).
static void write_at_large(MPI_File fh, MPI_Offset off, const void* buf, uint64_t bytes) {
    const char* p = (const char*)buf;
    const uint64_t CHUNK = 1ull << 30;   // 1 GiB
    uint64_t done = 0;
    while (done < bytes) {
        uint64_t n = std::min(CHUNK, bytes - done);
        MPI_File_write_at(fh, off + (MPI_Offset)done, p + done, (int)n, MPI_BYTE,
                          MPI_STATUS_IGNORE);
        done += n;
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    std::string base_path, index_dir;
    for (int i = 1; i < argc - 1; i++) {
        std::string k = argv[i], v = argv[i + 1];
        if      (k == "--base_path") { base_path = v; i++; }
        else if (k == "--index_dir") { index_dir = v; i++; }
    }
    if (base_path.empty() || index_dir.empty()) {
        if (rank == 0) fprintf(stderr, "Usage: mpirun -n W %s --base_path <fvecs> --index_dir <dir>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    Metric metric = Metric::L2;
    if (rank == 0) metric = read_index_metric(index_dir);
    int metric_i = (int)metric;
    MPI_Bcast(&metric_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    metric = (Metric)metric_i;

    // ------------------------------------------------------------------
    // Read global graph header + PQ params.
    // ------------------------------------------------------------------
    uint32_t N, dim, R, mcode;
    uint64_t total_edges;
    {
        FILE* fp = fopen((index_dir + "/global_graph.bin").c_str(), "rb");
        if (!fp) { if (rank==0) fprintf(stderr,"missing global_graph.bin\n"); MPI_Abort(MPI_COMM_WORLD,1); }
        uint32_t magic, ver, pad;
        if (fread(&magic,4,1,fp)!=1||fread(&ver,4,1,fp)!=1||fread(&N,4,1,fp)!=1||
            fread(&dim,4,1,fp)!=1||fread(&R,4,1,fp)!=1||fread(&pad,4,1,fp)!=1||
            fread(&total_edges,8,1,fp)!=1) { fclose(fp); MPI_Abort(MPI_COMM_WORLD,1); }
        fclose(fp);
        if (magic != GLOBAL_MAGIC) { if(rank==0) fprintf(stderr,"bad global magic\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    // PQ code length m from codes.bin header.
    {
        FILE* fp = fopen((index_dir + "/codes.bin").c_str(), "rb");
        if (!fp) { if (rank==0) fprintf(stderr,"missing codes.bin\n"); MPI_Abort(MPI_COMM_WORLD,1); }
        uint32_t hdr[4];
        if (fread(hdr,4,4,fp)!=4) { fclose(fp); MPI_Abort(MPI_COMM_WORLD,1); }
        fclose(fp);
        mcode = hdr[3];
    }

    if (rank == 0) {
        printf("[rank 0] Step 6 (KV store): N=%u dim=%u R=%u m=%u total_edges=%lu\n",
               N, dim, R, mcode, (unsigned long)total_edges);
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // mmap base.fvecs and codes.bin for random access by id.
    // ------------------------------------------------------------------
    int fd_base = open(base_path.c_str(), O_RDONLY);
    struct stat sb; fstat(fd_base, &sb);
    const char* base_map = (const char*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd_base, 0);
    const size_t bpv = 4 + (size_t)dim * 4;
    auto vec = [&](uint32_t id){ return (const float*)(base_map + (size_t)id * bpv + 4); };

    int fd_codes = open((index_dir + "/codes.bin").c_str(), O_RDONLY);
    struct stat sc; fstat(fd_codes, &sc);
    const char* codes_map = (const char*)mmap(nullptr, sc.st_size, PROT_READ, MAP_PRIVATE, fd_codes, 0);
    auto code = [&](uint32_t id){ return (const uint8_t*)(codes_map + CODES_HDR_BYTES + (size_t)id * mcode); };

    // ------------------------------------------------------------------
    // Owned id range; read its adjacency slice from global_graph.bin.
    // ------------------------------------------------------------------
    uint32_t my_start, my_count;
    slice_bounds(N, nranks, rank, my_start, my_count);

    const MPI_Offset goff_sec = (MPI_Offset)GLOBAL_HDR_BYTES;
    const MPI_Offset gnbr_sec = goff_sec + (MPI_Offset)((uint64_t)N + 1) * 8;

    std::vector<uint64_t> goff((size_t)my_count + 1);
    {
        FILE* fp = fopen((index_dir + "/global_graph.bin").c_str(), "rb");
        fseeko(fp, (off_t)goff_sec + (off_t)my_start * 8, SEEK_SET);
        if (fread(goff.data(), 8, my_count + 1, fp) != (size_t)my_count + 1) { fclose(fp); MPI_Abort(MPI_COMM_WORLD,1); }
        fclose(fp);
    }
    uint64_t edge_base   = goff[0];
    uint64_t local_edges = goff[my_count] - goff[0];

    std::vector<uint32_t> nbr_ids((size_t)local_edges);
    {
        FILE* fp = fopen((index_dir + "/global_graph.bin").c_str(), "rb");
        fseeko(fp, (off_t)gnbr_sec + (off_t)edge_base * 4, SEEK_SET);
        if (local_edges > 0 && fread(nbr_ids.data(), 4, local_edges, fp) != local_edges) { fclose(fp); MPI_Abort(MPI_COMM_WORLD,1); }
        fclose(fp);
    }

    // ------------------------------------------------------------------
    // Assemble local sections: full vectors, neighbour codes.
    // ------------------------------------------------------------------
    std::vector<float> full_vecs((size_t)my_count * dim);
    for (uint32_t i = 0; i < my_count; i++)
        std::copy(vec(my_start + i), vec(my_start + i) + dim,
                  full_vecs.data() + (size_t)i * dim);
    // Full vectors are stored UN-normalised even for IP: they drive the exact
    // scoring of results, and normalising them would turn MIPS into cosine
    // similarity (different ranking whenever base norms vary). Normalised
    // vectors are only used upstream (clustering, graph, PQ) for navigation.

    std::vector<uint8_t> nbr_codes((size_t)local_edges * mcode);
    #pragma omp parallel for schedule(static)
    for (uint64_t e = 0; e < local_edges; e++)
        std::copy(code(nbr_ids[e]), code(nbr_ids[e]) + mcode,
                  nbr_codes.data() + e * mcode);

    if (rank == 0) { printf("[rank 0] All ranks assembled node records.\n"); fflush(stdout); }

    // ------------------------------------------------------------------
    // Collective write of nodes.bin.
    // ------------------------------------------------------------------
    std::string out = index_dir + "/nodes.bin";
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, out.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    if (rank == 0) {
        uint32_t hdr32[6] = { KV_MAGIC, KV_VERSION, N, dim, mcode, 0u };
        MPI_File_write_at(fh, 0, hdr32, 24, MPI_BYTE, MPI_STATUS_IGNORE);
        MPI_File_write_at(fh, 24, &total_edges, 8, MPI_BYTE, MPI_STATUS_IGNORE);
    }

    // offsets[] (same CSR as global graph) — write owned slice + sentinel.
    MPI_File_write_at(fh, (MPI_Offset)kv_off_offsets() + (MPI_Offset)my_start * 8,
                      goff.data(), (int)(my_count * 8), MPI_BYTE, MPI_STATUS_IGNORE);
    if (rank == nranks - 1)
        MPI_File_write_at(fh, (MPI_Offset)kv_off_offsets() + (MPI_Offset)N * 8,
                          &total_edges, 8, MPI_BYTE, MPI_STATUS_IGNORE);

    // full_vecs[]
    write_at_large(fh, (MPI_Offset)kv_off_vecs(N) + (MPI_Offset)my_start * dim * 4,
                   full_vecs.data(), (uint64_t)my_count * dim * 4);
    // nbr_ids[]
    if (local_edges > 0)
        write_at_large(fh, (MPI_Offset)kv_off_ids(N, dim) + (MPI_Offset)edge_base * 4,
                       nbr_ids.data(), local_edges * 4);
    // nbr_codes[]
    if (local_edges > 0)
        write_at_large(fh, (MPI_Offset)kv_off_codes(N, dim, total_edges) + (MPI_Offset)edge_base * mcode,
                       nbr_codes.data(), local_edges * mcode);

    MPI_File_close(&fh);
    munmap((void*)base_map, sb.st_size); close(fd_base);
    munmap((void*)codes_map, sc.st_size); close(fd_codes);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[rank 0] Written %s\n", out.c_str());
        printf("\n[rank 0] === Step 6 complete ===\n");
    }
    MPI_Finalize();
    return 0;
}
