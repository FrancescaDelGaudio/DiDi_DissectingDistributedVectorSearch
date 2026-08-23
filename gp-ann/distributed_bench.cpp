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

    MPI_Init(nullptr, nullptr);

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

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

    MPI_Barrier(MPI_COMM_WORLD);

    const auto local_query_results = bench.ProcessQueries3(my_query_ids, queries);

    MPI_Barrier(MPI_COMM_WORLD);

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

    MPI_Finalize();
    return 0;
}
