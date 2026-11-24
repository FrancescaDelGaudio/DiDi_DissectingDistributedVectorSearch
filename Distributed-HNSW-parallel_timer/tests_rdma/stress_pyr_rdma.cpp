// Stress test of pyr_rdma.h with Pyramid's message pattern: rank size-2 = sender (tasks to every worker,
// blocking Send, then pings and shutdown), ranks 0..size-3 = workers (Iprobe + Recv from the sender with
// TASK tag, one result per task to the collector, pong to the sender), rank size-1 = collector (Recv
// from any source, any tag). Every payload carries a checksum; counts and checksums are verified.
// Run with a small GP_RDMA_RING_MB (e.g. 1) so that rings wrap and senders wait for credit.
#include <mpi.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "pyr_rdma.h"

constexpr int TASK_TAG = 1, RESULT_TAG = 2, PING_TAG = 3;
struct Task { int32_t index; int32_t dest; uint64_t sum; char pad[392]; };      // 408 B, like task_message
struct Result { int32_t index; int32_t from; uint64_t sum; char pad[432]; };    // like result_message

static uint64_t fill(char* p, size_t n, uint64_t seed) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; ++i) { p[i] = static_cast<char>((seed * 2654435761u + i) & 0xff); s += static_cast<unsigned char>(p[i]); }
    return s;
}
static uint64_t sum_of(const char* p, size_t n) { uint64_t s = 0; for (size_t i = 0; i < n; ++i) s += static_cast<unsigned char>(p[i]); return s; }

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int workers = size - 2, sender = size - 2, collector = size - 1;
    const int ntasks = argc > 1 ? std::atoi(argv[1]) : 200000;
    gp_rdma::init(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    long errors = 0;
    if (rank == sender) {
        std::mt19937 rng(7);
        std::vector<long> per(workers, 0);
        for (int i = 0; i < ntasks; ++i) {
            Task t{};
            t.index = i;
            t.dest = static_cast<int>(rng() % workers);
            t.sum = fill(t.pad, sizeof(t.pad), i);
            PYR_Send(&t, sizeof(t), MPI_BYTE, t.dest, TASK_TAG, MPI_COMM_WORLD);
            per[t.dest]++;
        }
        for (int w = 0; w < workers; ++w) {   // ping: returns only after the worker drained its tasks (FIFO)
            Task p{}; p.index = -2;
            PYR_Send(&p, sizeof(p), MPI_BYTE, w, TASK_TAG, MPI_COMM_WORLD);
            Result r;
            PYR_Recv(&r, sizeof(r), MPI_BYTE, w, PING_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (r.index != per[w]) { std::printf("sender: worker %d processed %d tasks, expected %ld\n", w, r.index, per[w]); ++errors; }
        }
        for (int w = 0; w < workers; ++w) { Task s{}; s.index = -1; PYR_Send(&s, sizeof(s), MPI_BYTE, w, TASK_TAG, MPI_COMM_WORLD); }
    } else if (rank == collector) {
        std::vector<char> seen(ntasks, 0);
        for (int got = 0; got < ntasks; ++got) {
            Result r;
            MPI_Status st;
            PYR_Recv(&r, sizeof(r), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            if (st.MPI_TAG != RESULT_TAG || st.MPI_SOURCE != r.from || r.index < 0 || r.index >= ntasks || seen[r.index]
                || sum_of(r.pad, sizeof(r.pad)) != r.sum) { if (errors < 5) std::printf("collector: bad result %d from %d\n", r.index, r.from); ++errors; }
            else seen[r.index] = 1;
        }
    } else {
        int done = 0, processed = 0;
        while (!done) {
            int flag = 0;
            PYR_Iprobe(sender, TASK_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            if (!flag) continue;
            Task t;
            PYR_Recv(&t, sizeof(t), MPI_BYTE, sender, TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (t.index == -1) { done = 1; break; }
            if (t.index == -2) { Result p{}; p.index = processed; PYR_Send(&p, sizeof(p), MPI_BYTE, sender, PING_TAG, MPI_COMM_WORLD); continue; }
            if (t.dest != rank || sum_of(t.pad, sizeof(t.pad)) != t.sum) { if (errors < 5) std::printf("worker %d: bad task %d\n", rank, t.index); ++errors; }
            Result r{};
            r.index = t.index; r.from = rank;
            r.sum = fill(r.pad, sizeof(r.pad), t.index + 99);
            PYR_Send(&r, sizeof(r), MPI_BYTE, collector, RESULT_TAG, MPI_COMM_WORLD);
            ++processed;
        }
    }
    const double dt = MPI_Wtime() - t0;
    long total = 0;
    MPI_Reduce(&errors, &total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    gp_rdma::finalize();
    if (rank == 0) std::printf("stress_pyr_rdma ranks=%d tasks=%d time_s=%.3f errors=%ld -> %s\n", size, ntasks, dt, total, total ? "FAIL" : "OK");
    MPI_Finalize();
    return total ? 1 : 0;
}
