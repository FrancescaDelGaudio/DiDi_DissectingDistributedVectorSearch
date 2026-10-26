// Empty-exchange replay of Harmony's query-time communication (group mode), no computation.
// Same layout as the real runs: rank 0 = master, ranks 1..W = workers, teams of 2 workers
// (team t = ranks 2t-1, 2t). Message sizes come from the real run's logs (per-team compared
// vectors), so the result is the time MPI itself needs to move the same data on this cluster.
//
// Phases (each timed after a barrier; time = slowest rank; repeated, first pass is warm-up):
//   dist  : master -> workers: bcast nq/k, bcast queries (nq*d float), bcast nprobe lists
//           (nq*nprobe int64), per-worker compare sizes (2 x ~nq int64), bcast heap tops (nq float)
//   team  : for each group g and block b, inside every team the first worker of the block
//           (alternating) sends that block's partial distances (float) to its teammate
//   heaps : the last worker of each block sends its heap (blockSize*k float + int64) to the
//           master; the master forwards each group's heaps to the next team (3 hops per group)
//
// Usage: harmony_comm_replay <nprobe> <team1_floats> <team2_floats> <team3_floats> <team4_floats>
//                            [reps=6] [nq=10000] [d=96] [k=10] [groups=4] [blocks=5]
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 6) {
        if (rank == 0) std::fprintf(stderr, "usage: %s nprobe t1 t2 t3 t4 [reps nq d k groups blocks]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const long nprobe = std::atol(argv[1]);
    std::vector<long long> team_floats = {std::atoll(argv[2]), std::atoll(argv[3]), std::atoll(argv[4]), std::atoll(argv[5])};
    const int reps = argc > 6 ? std::atoi(argv[6]) : 6;
    const long nq = argc > 7 ? std::atol(argv[7]) : 10000;
    const long d = argc > 8 ? std::atol(argv[8]) : 96;
    const long k = argc > 9 ? std::atol(argv[9]) : 10;
    const int groups = argc > 10 ? std::atoi(argv[10]) : 4;
    const int blocks = argc > 11 ? std::atoi(argv[11]) : 5;
    const int teams = 4, team_size = 2, workers = teams * team_size;
    if (size != workers + 1) {
        if (rank == 0) std::fprintf(stderr, "need %d ranks (master + %d workers), got %d\n", workers + 1, workers, size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const long group_size = nq / groups;
    const long block_size = group_size / blocks;
    const int team = rank == 0 ? -1 : (rank - 1) / team_size;          // 0-based team of a worker
    const int in_team = rank == 0 ? -1 : (rank - 1) % team_size;       // 0 or 1
    const int msgs_per_team = groups * blocks;
    long long my_msg_floats = 0;
    if (rank > 0) my_msg_floats = team_floats[team] / msgs_per_team;
    long long max_msg_floats = 0;
    for (long long t : team_floats) max_msg_floats = std::max(max_msg_floats, t / msgs_per_team);

    // Buffers (contents irrelevant)
    std::vector<float> queries(nq * d, 0.0f), heap_tops(nq, 0.0f);
    std::vector<int64_t> lists(nq * nprobe, 0), csize(nq, 0), cpresum(nq + 1, 0);
    std::vector<float> dist_buf(rank > 0 ? my_msg_floats : 1, 0.0f);
    std::vector<float> hd(block_size * k, 0.0f), gd(group_size * k, 0.0f), gtops(group_size, 0.0f);
    std::vector<int64_t> hi(block_size * k, 0), gi(group_size * k, 0);

    auto run_dist = [&]() {
        long long hdr[2] = {nq, k};
        MPI_Bcast(hdr, 2, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(queries.data(), (int) queries.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(lists.data(), (int) lists.size(), MPI_INT64_T, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (int w = 1; w <= workers; w++) {
                MPI_Send(csize.data(), (int) nq, MPI_INT64_T, w, 0, MPI_COMM_WORLD);
                MPI_Send(cpresum.data(), (int) nq + 1, MPI_INT64_T, w, 0, MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(csize.data(), (int) nq, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(cpresum.data(), (int) nq + 1, MPI_INT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Bcast(heap_tops.data(), (int) nq, MPI_FLOAT, 0, MPI_COMM_WORLD);
    };

    auto run_team = [&]() {
        if (rank == 0) return;
        const int mate = (in_team == 0) ? rank + 1 : rank - 1;
        for (int g = 0; g < groups; g++) {
            for (int b = 0; b < blocks; b++) {
                const int first = b % 2;   // first worker of the block alternates
                const int tag = g * blocks + b;
                if (in_team == first)
                    MPI_Send(dist_buf.data(), (int) my_msg_floats, MPI_FLOAT, mate, tag, MPI_COMM_WORLD);
                else
                    MPI_Recv(dist_buf.data(), (int) my_msg_floats, MPI_FLOAT, mate, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    };

    auto run_heaps = [&]() {
        for (int g = 0; g < groups; g++) {
            for (int b = 0; b < blocks; b++) {
                const int last = 1 - (b % 2);
                const int tag = 1000 + g * blocks + b;
                if (rank == 0) {
                    for (int t = 0; t < teams; t++) {
                        const int src = 1 + t * team_size + last;
                        MPI_Recv(hd.data(), (int) hd.size(), MPI_FLOAT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(hi.data(), (int) hi.size(), MPI_INT64_T, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                } else if (in_team == last) {
                    MPI_Send(hd.data(), (int) hd.size(), MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
                    MPI_Send(hi.data(), (int) hi.size(), MPI_INT64_T, 0, tag, MPI_COMM_WORLD);
                }
            }
            // forward this group's heaps to the next team (3 hops per group over 4 teams)
            for (int hop = 0; hop < teams - 1; hop++) {
                const int dst_team = (g + hop + 1) % teams;
                const int tag = 5000 + g * teams + hop;
                for (int r = 0; r < team_size; r++) {
                    const int dst = 1 + dst_team * team_size + r;
                    if (rank == 0) {
                        MPI_Send(gtops.data(), (int) gtops.size(), MPI_FLOAT, dst, tag, MPI_COMM_WORLD);
                        MPI_Send(gd.data(), (int) gd.size(), MPI_FLOAT, dst, tag, MPI_COMM_WORLD);
                        MPI_Send(gi.data(), (int) gi.size(), MPI_INT64_T, dst, tag, MPI_COMM_WORLD);
                    } else if (rank == dst) {
                        MPI_Recv(gtops.data(), (int) gtops.size(), MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(gd.data(), (int) gd.size(), MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(gi.data(), (int) gi.size(), MPI_INT64_T, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
        }
    };

    struct Phase { const char* name; void (*dummy)(); };
    const char* names[3] = {"dist", "team", "heaps"};
    std::vector<std::vector<double>> times(3);
    for (int r = 0; r < reps; r++) {
        for (int p = 0; p < 3; p++) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();
            if (p == 0) run_dist();
            else if (p == 1) run_team();
            else run_heaps();
            double local = MPI_Wtime() - t0, slowest = 0;
            MPI_Reduce(&local, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0 && r > 0) times[p].push_back(slowest);   // pass 0 = warm-up
        }
    }

    if (rank == 0) {
        const double dist_bytes = 16.0 + nq * d * 4.0 + nq * nprobe * 8.0 + (2.0 * nq + 1) * 8.0 + nq * 4.0;
        double team_bytes_max = 0, team_bytes_sum = 0;
        for (long long t : team_floats) {
            const double b = (double) (t / msgs_per_team) * msgs_per_team * 4.0;
            team_bytes_max = std::max(team_bytes_max, b);
            team_bytes_sum += b;
        }
        const double heap_in = (double) groups * blocks * teams * block_size * k * 12.0;
        const double heap_out = (double) groups * (teams - 1) * team_size * group_size * (4.0 + k * 12.0);
        const double bytes[3] = {dist_bytes, team_bytes_max, heap_in + heap_out};
        std::printf("#[replay] nprobe=%ld nq=%ld d=%ld k=%ld groups=%d blocks=%d reps=%d max_msg_MB=%.1f\n",
                    nprobe, nq, d, k, groups, blocks, reps - 1, max_msg_floats * 4.0 / 1e6);
        for (int p = 0; p < 3; p++) {
            double mn = 1e30, sum = 0;
            for (double t : times[p]) { mn = std::min(mn, t); sum += t; }
            std::printf("#[replay] phase=%s min_s=%.6f mean_s=%.6f bytes_per_busiest=%.0f team_bytes_total=%.0f\n",
                        names[p], mn, sum / times[p].size(), bytes[p], p == 1 ? team_bytes_sum : 0.0);
        }
    }
    MPI_Finalize();
    return 0;
}
