// DistributedANN — index construction (Step 4: PQ codebook + vector codes)
//
// Run with mpirun:
//   mpirun -n W ./build_pq
//       --base_path <fvecs> --index_dir <dir>
//       [--m 16] [--train_size 100000] [--iters 25]
//
// - Rank 0 trains a global PQ codebook on a strided sample, then broadcasts it.
// - Each rank encodes its contiguous slice of the dataset and writes the codes
//   to codes.bin via MPI-IO.
//
// Outputs:
//   <index_dir>/pq_codebook.bin   global PQ codebook
//   <index_dir>/codes.bin         N * m bytes (PQ code per vector, id-ordered)

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include "common/types.hpp"
#include "common/io.hpp"
#include "common/metric_io.hpp"
#include "common/metric.hpp"
#include "quantize/pq.hpp"

using namespace dann;

struct Args {
    std::string base_path;
    std::string index_dir;
    int m          = 16;
    int train_size = 100000;
    int iters      = 25;
    int chunk      = 200000;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc - 1; i++) {
        std::string k = argv[i], v = argv[i + 1];
        if      (k == "--base_path")  { a.base_path  = v;          i++; }
        else if (k == "--index_dir")  { a.index_dir  = v;          i++; }
        else if (k == "--m")          { a.m          = std::stoi(v); i++; }
        else if (k == "--train_size") { a.train_size = std::stoi(v); i++; }
        else if (k == "--iters")      { a.iters      = std::stoi(v); i++; }
        else if (k == "--chunk")      { a.chunk      = std::stoi(v); i++; }
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
                            "--index_dir <dir> [--m 16] [--train_size 100000] "
                            "[--iters 25]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    uint32_t N = 0;
    int dim = 0;
    if (rank == 0) { N = fvecs_count(args.base_path); dim = fvecs_dim(args.base_path); }
    MPI_Bcast(&N,   1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_INT,      0, MPI_COMM_WORLD);

    Metric metric = Metric::L2;
    if (rank == 0) metric = read_index_metric(args.index_dir);
    int metric_i = (int)metric;
    MPI_Bcast(&metric_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    metric = (Metric)metric_i;

    if (rank == 0) {
        printf("[rank 0] Step 4 (PQ): N=%u dim=%d m=%d ksub=256 -> code=%d bytes metric=%s\n",
               N, dim, args.m, args.m, metric_name(metric));
        printf("[rank 0] train_size=%d iters=%d\n", args.train_size, args.iters);
        fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Rank 0 trains the codebook on a strided sample; broadcast.
    // ------------------------------------------------------------------
    ProductQuantizer pq(dim, args.m, 256);
    pq.metric = metric;

    if (rank == 0) {
        int ts = std::min<uint32_t>(args.train_size, N);
        std::vector<float> sample((size_t)ts * dim);
        // Strided sample across the whole dataset.
        FILE* fp = fopen(args.base_path.c_str(), "rb");
        size_t bytes_per_vec = 4 + (size_t)dim * 4;
        for (int i = 0; i < ts; i++) {
            uint32_t id = (uint32_t)((uint64_t)i * N / ts);
            fseeko(fp, (off_t)id * bytes_per_vec + 4, SEEK_SET);
            if (fread(sample.data() + (size_t)i * dim, 4, dim, fp) != (size_t)dim)
                throw std::runtime_error("sample read failed");
        }
        fclose(fp);

        if (metric == Metric::IP) normalize_rows(sample.data(), ts, dim);

        printf("[rank 0] Training PQ on %d vectors...\n", ts);
        fflush(stdout);
        pq.train(sample.data(), ts, args.iters);
        pq.save(args.index_dir + "/pq_codebook.bin");
        printf("[rank 0] Written %s\n", (args.index_dir + "/pq_codebook.bin").c_str());
        fflush(stdout);
    }

    // Broadcast codebook to all ranks.
    MPI_Bcast(pq.centroids.data(), (int)pq.centroids.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Each rank encodes its slice and writes codes.bin via MPI-IO.
    // ------------------------------------------------------------------
    uint32_t my_start, my_count;
    slice_bounds(N, nranks, rank, my_start, my_count);

    std::vector<uint8_t> my_codes((size_t)my_count * args.m);
    for (uint32_t start = 0; start < my_count; start += (uint32_t)args.chunk) {
        uint32_t cnt = std::min((uint32_t)args.chunk, my_count - start);
        int rd = 0;
        std::vector<float> buf = read_fvecs_slice(args.base_path, my_start + start, cnt, rd);
        if (metric == Metric::IP) normalize_rows(buf.data(), (int)cnt, dim);
        #pragma omp parallel for schedule(static)
        for (uint32_t i = 0; i < cnt; i++)
            pq.encode(buf.data() + (size_t)i * dim,
                      my_codes.data() + (size_t)(start + i) * args.m);
    }

    std::string out = args.index_dir + "/codes.bin";
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, out.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (rank == 0) {
        uint32_t hdr[4] = { CODES_MAGIC, CODES_VERSION, N, (uint32_t)args.m };
        MPI_File_write_at(fh, 0, hdr, 16, MPI_BYTE, MPI_STATUS_IGNORE);
    }
    MPI_Offset pos = (MPI_Offset)CODES_HDR_BYTES + (MPI_Offset)my_start * args.m;
    MPI_File_write_at(fh, pos, my_codes.data(), (int)(my_count * args.m),
                      MPI_BYTE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[rank 0] Written %s (%u codes x %d bytes)\n", out.c_str(), N, args.m);
        printf("\n[rank 0] === Step 4 complete ===\n");
    }
    MPI_Finalize();
    return 0;
}
