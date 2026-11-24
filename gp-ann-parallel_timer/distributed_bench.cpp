// OpenMPI pulls deprecated C++ MPI bindings unless skipped; those need libmpi_cxx,
// which is missing from many cluster OpenMPI installs (e.g. /softs builds).
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>
#include "distributed_query_benchmark.h"

#include "points_io.h"
#include "metis_io.h"
#include "recall.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>

namespace {

    size_t ComputeChunkSize(size_t a, size_t b) {
        return (a + b - 1) / b;
    }

    // GP_WARMUP=1: exercise, before the timer, every communication path the query phase uses, with the
    // same calls (point-to-point messages to every peer through Isend / Improbe / Mrecv, small and large;
    // the termination Iallreduce on 2 size_t; Gather), so that one-time set-up of the communication layer
    // (MPI/UCX connections, protocol lanes, first collective) is not inside the timed phase. Applied to both
    // the MPI and the RDMA build (the RDMA layer sets up its connections in gp_rdma::init anyway).
    // Own tag, all messages drained and a barrier before returning: nothing is left for the query phase.
    void CommWarmup(int rank, int size) {
        const int kTag = 9999;
        for (int bytes : {64, 512 * 1024}) {
            std::vector<char> sbuf(bytes, 1), rbuf(bytes);
            std::vector<MPI_Request> reqs(size > 1 ? size - 1 : 0);
            int k = 0;
            for (int p = 0; p < size; ++p)
                if (p != rank) MPI_Isend(sbuf.data(), bytes, MPI_BYTE, p, kTag, MPI_COMM_WORLD, &reqs[k++]);
            int got = 0;
            while (got < size - 1) {
                int flag = 0;
                MPI_Message msg;
                MPI_Status st;
                MPI_Improbe(MPI_ANY_SOURCE, kTag, MPI_COMM_WORLD, &flag, &msg, &st);
                if (flag) {
                    int count = 0;
                    MPI_Get_count(&st, MPI_BYTE, &count);
                    MPI_Mrecv(rbuf.data(), count, MPI_BYTE, &msg, MPI_STATUS_IGNORE);
                    ++got;
                }
            }
            if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }
        size_t counts[2] = {1, 1};
        MPI_Request r;
        MPI_Iallreduce(MPI_IN_PLACE, counts, 2, sizeof(size_t) == 8 ? MPI_UNSIGNED_LONG : MPI_UNSIGNED, MPI_SUM,
                       MPI_COMM_WORLD, &r);
        MPI_Waitall(1, &r, MPI_STATUSES_IGNORE);
        unsigned long long one = 1;
        std::vector<unsigned long long> all(rank == 0 ? size : 1);
        MPI_Gather(&one, 1, MPI_UNSIGNED_LONG_LONG, all.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

} // namespace

int main(int argc, const char* argv[]) {
    // Optional 7th arg: ef_search (HNSW search budget). Default remains HNSWParameters::ef_search (250).
    if (argc != 7 && argc != 8) {
        std::cerr << "Usage ./DistributedBench input-points queries ground-truth-file num_neighbors "
                     "partition-file router-file [ef_search]"
                  << std::endl;
        std::abort();
    }

    std::string point_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
    std::string k_string = argv[4];
    int num_neighbors = std::stoi(k_string);
    std::string partition_file = argv[5];
    std::string router_file = argv[6];
    const bool ef_search_overridden = (argc == 8);
    const size_t ef_search_cli = ef_search_overridden ? static_cast<size_t>(std::stoul(argv[7])) : 0;

    // parallel_timer: parlay workers route and search, only the main thread calls MPI.
    int mpi_provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &mpi_provided);
    if (mpi_provided < MPI_THREAD_FUNNELED) {
        std::cerr << "FATAL: MPI does not provide MPI_THREAD_FUNNELED" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#ifdef GP_RDMA
    gp_rdma::init(MPI_COMM_WORLD);   // queue pairs and registered buffers, before any timing
#endif

    DistributedQueryBenchmark bench;
    bench.num_neighbors = num_neighbors;
    if (ef_search_overridden) {
        if (ef_search_cli == 0) {
            if (rank == 0) {
                std::cerr << "FATAL: ef_search must be >= 1" << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
        bench.hnsw_parameters.ef_search = ef_search_cli;
    }
    bench.LoadPartition(partition_file);
    if (comm_size != bench.num_shards) {
        if (rank == 0) {
            std::cerr << "MPI world size (" << comm_size << ") must equal num_shards (" << bench.num_shards
                      << ") from the partition file." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    bench.LoadShardPointSet(point_file);
    bench.BuildInShardIndex();
    bench.LoadRouter(router_file);

    {
        unsigned long long local_points = bench.shard_points.n;
        unsigned long long total_replicated_points = 0;
        MPI_Reduce(&local_points, &total_replicated_points, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "partition layout: " << (bench.overlapping_partition_layout ? "overlapping (cover)" : "flat METIS")
                      << " num_shards=" << bench.num_shards << " num_probes=" << bench.num_probes
                      << " ef_search=" << bench.hnsw_parameters.ef_search
                      << " replicated point copies across ranks=" << total_replicated_points << std::endl;
        }
    }

    PointSet queries = ReadPoints(query_file);
    std::vector<int> query_ids(queries.n);
    std::iota(query_ids.begin(), query_ids.end(), 0);
    size_t chunk_size = ComputeChunkSize(query_ids.size(), comm_size);
    std::vector<int> my_query_ids(query_ids.begin() + rank * chunk_size, query_ids.begin() + std::min(query_ids.size(), (rank + 1) * chunk_size));

    // GP_ALT_TRANSPORT=1 (GP_RDMA build only): MPI and RDMA batches in the same process, on the same loaded
    // index. Batch 1 = MPI, batch 2 = RDMA (first use of each transport), then from batch 3 the pattern
    // MPI RDMA RDMA MPI repeats. The switch happens between batches, after a barrier, with nothing outstanding.
    const bool alt_transport = std::getenv("GP_ALT_TRANSPORT") && std::atoi(std::getenv("GP_ALT_TRANSPORT")) == 1;
    auto transport_of = [&](int b) -> const char* {
#ifdef GP_RDMA
        if (!alt_transport) return "rdma";
        if (b <= 2) return b == 1 ? "mpi" : "rdma";
        const int k = (b - 3) % 4;
        return (k == 0 || k == 3) ? "mpi" : "rdma";
#else
        (void) b;
        return "mpi";
#endif
    };
    auto set_transport = [&](int b) {
#ifdef GP_RDMA
        gp_rdma::use_mpi = std::string(transport_of(b)) == "mpi";
#else
        (void) b;
#endif
    };
    set_transport(1);
    if (std::getenv("GP_WARMUP") && std::atoi(std::getenv("GP_WARMUP")) == 1) {
        CommWarmup(rank, comm_size);
        // Then one untimed pass over the same queries in communication-only mode (GP_COMM_ONLY=1: same
        // messages, sizes, destinations and buffers, dummy results, no shard search), so that per-buffer
        // set-up of the communication layer (e.g. memory registration of the real message buffers) is also
        // done before the timer, while the shard search stays cold as in the throughput curves. Routing
        // runs in this pass (it is part of the replay).
        const char* prev = std::getenv("GP_COMM_ONLY");
        const std::string prev_comm = prev ? prev : "";
        setenv("GP_COMM_ONLY", "1", 1);
        setenv("GP_IN_WARMUP", "1", 1);
        MPI_Barrier(MPI_COMM_WORLD);
        bench.ProcessQueries3(my_query_ids, queries);
        MPI_Barrier(MPI_COMM_WORLD);
        unsetenv("GP_IN_WARMUP");
        if (prev) setenv("GP_COMM_ONLY", prev_comm.c_str(), 1); else unsetenv("GP_COMM_ONLY");
        if (rank == 0) std::cout << "#[warmup] communication warm-up (messages + comm-only pass) done before the timer" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Timer query_timer;
    query_timer.Start();
    const auto local_query_results = bench.ProcessQueries3(my_query_ids, queries);
    const double local_query_s = query_timer.Stop();

    MPI_Barrier(MPI_COMM_WORLD);

    double query_phase_s = 0.0;
    MPI_Reduce(&local_query_s, &query_phase_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double query_qps = static_cast<double>(queries.n) / query_phase_s;
        std::cout << "Query phase took " << query_phase_s << " s (max rank wall; "
                  << queries.n << " queries, distributed)" << std::endl;
        std::cout << "Query QPS " << query_qps << std::endl;
        std::cout << "#[batch] i=1 query_s=" << query_phase_s << " transport=" << transport_of(1) << std::endl;
    }

    // GP_BATCHES=N: process the same query batch N times in this process. Batch 1 is the one timed above
    // (first communication of the process), the others run on established connections and buffers.
    const char* batches_env = std::getenv("GP_BATCHES");
    const int batches = batches_env ? std::atoi(batches_env) : 1;
    for (int b = 2; b <= batches; ++b) {
        set_transport(b);
        MPI_Barrier(MPI_COMM_WORLD);
        Timer batch_timer;
        batch_timer.Start();
        bench.ProcessQueries3(my_query_ids, queries);
        const double local_batch_s = batch_timer.Stop();
        MPI_Barrier(MPI_COMM_WORLD);
        double batch_s = 0.0;
        MPI_Reduce(&local_batch_s, &batch_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) std::cout << "#[batch] i=" << b << " query_s=" << batch_s << " transport=" << transport_of(b) << std::endl;
    }

    std::vector<NNVec> ground_truth = ReadGroundTruth(ground_truth_file);
    if (ground_truth.size() < queries.n) {
        if (rank == 0) {
            std::cerr << "Ground-truth has fewer queries than the query file: gt=" << ground_truth.size()
                      << " queries=" << queries.n << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (!ground_truth.empty() && static_cast<int>(ground_truth[0].size()) < num_neighbors) {
        if (rank == 0) {
            std::cerr << "Ground-truth has fewer neighbors than requested k: gt_k=" << ground_truth[0].size()
                      << " k=" << num_neighbors << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    if (ground_truth.size() > queries.n) {
        ground_truth.resize(queries.n);
    }

    PointSet points_rank0;
    if (rank == 0) {
        points_rank0 = ReadPoints(point_file);
    }

    unsigned long long local_matches = 0;
    for (const auto& [query_id, neighbors] : local_query_results) {
        std::unordered_set<uint32_t> gt_ids;
        gt_ids.reserve(num_neighbors);
        for (int j = 0; j < num_neighbors; ++j) {
            gt_ids.insert(ground_truth[query_id][j].second);
        }

        const int eval_k = std::min(num_neighbors, static_cast<int>(neighbors.size()));
        for (int j = 0; j < eval_k; ++j) {
            if (gt_ids.count(neighbors[j].second) > 0) {
                local_matches++;
            }
        }
    }

    unsigned long long global_matches = 0;
    MPI_Reduce(&local_matches, &global_matches, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<float> distance_to_kth_neighbor(queries.n, 0.0f);
    if (rank == 0) {
        std::vector<NNVec> gt_for_threshold = ground_truth;
        distance_to_kth_neighbor = ConvertGroundTruthToDistanceToKthNeighbor(gt_for_threshold, num_neighbors, points_rank0, queries);
    }
    const int nq = static_cast<int>(queries.n);
    MPI_Bcast(distance_to_kth_neighbor.data(), nq, MPI_FLOAT, 0, MPI_COMM_WORLD);

    unsigned long long local_threshold_hits = 0;
    for (const auto& [query_id, neighbors] : local_query_results) {
        const float threshold = distance_to_kth_neighbor[static_cast<size_t>(query_id)];
        for (const auto& x : neighbors) {
            if (x.first <= threshold) {
                local_threshold_hits++;
            }
        }
    }
    unsigned long long global_threshold_hits = 0;
    MPI_Reduce(&local_threshold_hits, &global_threshold_hits, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double denom = static_cast<double>(queries.n) * static_cast<double>(num_neighbors);
        const double mean_recall_id = static_cast<double>(global_matches) / denom;
        const double mean_recall_threshold = static_cast<double>(global_threshold_hits) / denom;
        std::cout << "Mean recall@" << num_neighbors << " (ID overlap) " << mean_recall_id << std::endl;
        std::cout << "Mean recall@" << num_neighbors << " (distance threshold, k-th GT neighbor) " << mean_recall_threshold << std::endl;
    }

#ifdef GP_RDMA
    gp_rdma::finalize();
#endif
    MPI_Finalize();
    return 0;
}
