#pragma once
#include <cstdlib>
#include <string>

#include "points_io.h"
#include "metis_io.h"

#include "hnsw_router.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"

#include "../external/message-queue/include/message-queue/buffered_queue.hpp"

#include <parlay/primitives.h>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

class DistributedQueryBenchmark {
public:
    int rank;
    int comm_size;

    int num_shards;
    int dim;
    int num_neighbors;

    PointSet shard_points;
    std::vector<int> partition;
    std::vector<int> routing_index_partition;
    Cover cover;
    bool overlapping_partition_layout = false;
    std::vector<uint32_t> local_to_global_ids;

    #ifdef MIPS_DISTANCE
    using SpaceT = hnswlib::InnerProductSpace;
    #else
    using SpaceT = hnswlib::L2Space;
    #endif

    std::unique_ptr<SpaceT> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
    std::unique_ptr<HNSWRouter> router;
    HNSWParameters hnsw_parameters;
    int num_voting_neighbors = 250;
    // shardsProbe: max overlapping shards queried per query (override at build: -DDISTBENCH_SHARDS_PROBE=N)
#ifndef DISTBENCH_SHARDS_PROBE
#define DISTBENCH_SHARDS_PROBE 6
#endif
    int num_probes = DISTBENCH_SHARDS_PROBE;

    // A routed query request sent from the owner rank to a target shard rank.
    struct Request {
        int query_id = -1;
        int origin_rank = -1;
        int target_shard = -1;
        std::vector<float> coordinates;
    };

    // One local ANN hit produced by a target shard.
    struct NeighborHit {
        uint32_t global_id = 0;
        float distance = 0.0f;
    };

    // A shard-local response sent back to the query owner rank.
    struct Response {
        int query_id = -1;
        int origin_rank = -1;
        int responder_rank = -1;
        int target_shard = -1;
        std::vector<NeighborHit> topk;
    };

private:
    static constexpr uint32_t kRequestMagic = 0x71524551;   // "qREQ"
    static constexpr uint32_t kResponseMagic = 0x70534552;  // "pSER"

    static float PackU32(uint32_t value) {
        float out;
        std::memcpy(&out, &value, sizeof(float));
        return out;
    }

    static uint32_t UnpackU32(float value) {
        uint32_t out;
        std::memcpy(&out, &value, sizeof(float));
        return out;
    }

    static void EncodeRequest(std::vector<float>& buf, const Request& req, int dim) {
        if (static_cast<int>(req.coordinates.size()) != dim) {
            throw std::runtime_error("Invalid request payload: query dimension mismatch");
        }

        buf.push_back(PackU32(kRequestMagic));
        buf.push_back(PackU32(static_cast<uint32_t>(req.query_id)));
        buf.push_back(PackU32(static_cast<uint32_t>(req.origin_rank)));
        buf.push_back(PackU32(static_cast<uint32_t>(req.target_shard)));
        buf.push_back(PackU32(static_cast<uint32_t>(dim)));
        buf.insert(buf.end(), req.coordinates.begin(), req.coordinates.end());
    }

    static std::vector<Request> DecodeRequests(const auto& buf) {
        std::vector<Request> out;
        size_t i = 0;
        while (i < buf.size()) {
            if (i + 5 > buf.size()) {
                throw std::runtime_error("Corrupted request payload: incomplete header");
            }

            const uint32_t magic = UnpackU32(buf[i++]);
            if (magic != kRequestMagic) {
                throw std::runtime_error("Corrupted request payload: bad magic");
            }

            Request req;
            req.query_id = static_cast<int>(UnpackU32(buf[i++]));
            req.origin_rank = static_cast<int>(UnpackU32(buf[i++]));
            req.target_shard = static_cast<int>(UnpackU32(buf[i++]));
            const uint32_t dim = UnpackU32(buf[i++]);

            if (i + dim > buf.size()) {
                throw std::runtime_error("Corrupted request payload: incomplete coordinates");
            }

            req.coordinates.assign(buf.begin() + i, buf.begin() + i + dim);
            i += dim;
            out.push_back(std::move(req));
        }
        return out;
    }

    static void EncodeResponse(std::vector<float>& buf, const Response& resp) {
        buf.push_back(PackU32(kResponseMagic));
        buf.push_back(PackU32(static_cast<uint32_t>(resp.query_id)));
        buf.push_back(PackU32(static_cast<uint32_t>(resp.origin_rank)));
        buf.push_back(PackU32(static_cast<uint32_t>(resp.responder_rank)));
        buf.push_back(PackU32(static_cast<uint32_t>(resp.target_shard)));
        buf.push_back(PackU32(static_cast<uint32_t>(resp.topk.size())));
        for (const auto& hit : resp.topk) {
            buf.push_back(PackU32(hit.global_id));
            buf.push_back(hit.distance);
        }
    }

    static std::vector<Response> DecodeResponses(const auto& buf) {
        std::vector<Response> out;
        size_t i = 0;
        while (i < buf.size()) {
            if (i + 6 > buf.size()) {
                throw std::runtime_error("Corrupted response payload: incomplete header");
            }

            const uint32_t magic = UnpackU32(buf[i++]);
            if (magic != kResponseMagic) {
                throw std::runtime_error("Corrupted response payload: bad magic");
            }

            Response resp;
            resp.query_id = static_cast<int>(UnpackU32(buf[i++]));
            resp.origin_rank = static_cast<int>(UnpackU32(buf[i++]));
            resp.responder_rank = static_cast<int>(UnpackU32(buf[i++]));
            resp.target_shard = static_cast<int>(UnpackU32(buf[i++]));
            const uint32_t k = UnpackU32(buf[i++]);

            if (i + (2 * k) > buf.size()) {
                throw std::runtime_error("Corrupted response payload: incomplete top-k list");
            }

            resp.topk.reserve(k);
            for (uint32_t j = 0; j < k; ++j) {
                NeighborHit hit;
                hit.global_id = UnpackU32(buf[i++]);
                hit.distance = buf[i++];
                resp.topk.push_back(hit);
            }
            out.push_back(std::move(resp));
        }
        return out;
    }

public:
    DistributedQueryBenchmark() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    }

    bool PointOnRank(uint32_t point_id) const {
        if (point_id >= cover.size()) {
            return false;
        }
        for (int shard : cover[point_id]) {
            if (shard == rank) {
                return true;
            }
        }
        return false;
    }

    void LoadPartition(const std::string& partition_file) {
        overlapping_partition_layout = PartitionFileIsClusterFormat(partition_file);
        if (overlapping_partition_layout) {
            const Clusters clusters = ReadClusters(partition_file);
            num_shards = static_cast<int>(clusters.size());
            cover = ConvertClustersToCover(clusters);
            partition.assign(cover.size(), -1);
            for (size_t point_id = 0; point_id < cover.size(); ++point_id) {
                if (!cover[point_id].empty()) {
                    partition[point_id] = cover[point_id][0];
                }
            }
        } else {
            partition = ReadMetisPartition(partition_file);
            num_shards = NumPartsInPartition(partition);
            cover = ConvertPartitionToCover(partition);
        }
    }

    void LoadShardPointSet(const std::string& point_set_file) {
        uint32_t n, d;
        {
            std::ifstream header_in(point_set_file, std::ios::binary);
            header_in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
            header_in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        }

        if (cover.size() < n) {
            cover.resize(n);
        }

        size_t num_points_in_shard = 0;
        for (uint32_t point_id = 0; point_id < n; ++point_id) {
            if (PointOnRank(point_id)) {
                num_points_in_shard++;
            }
        }

        const size_t header_bytes = 2 * sizeof(uint32_t);
        const size_t point_stride = static_cast<size_t>(d) * sizeof(float);

        shard_points.n = num_points_in_shard;
        shard_points.d = d;
        shard_points.coordinates.resize(shard_points.n * shard_points.d);
        local_to_global_ids.clear();
        local_to_global_ids.reserve(num_points_in_shard);
        dim = static_cast<int>(d);

        std::ifstream in(point_set_file, std::ios::binary);
        in.seekg(header_bytes);

        size_t coords_end = 0;
        for (uint32_t point_id = 0; point_id < n; ++point_id) {
            if (!PointOnRank(point_id)) {
                continue;
            }
            const size_t begin = header_bytes + static_cast<size_t>(point_id) * point_stride;
            in.seekg(static_cast<std::streamoff>(begin));
            in.read(reinterpret_cast<char*>(&shard_points.coordinates[coords_end]), point_stride);
            coords_end += d;
            local_to_global_ids.push_back(point_id);
        }
    }

    // parallel_timer: GP_COMM_ONLY=1 is a communication-only replay used to measure the MPI cost
    // of query processing: no shard index is built and every request is answered with a dummy
    // top-k of the same size, so the same messages (same destinations, sizes and buffering) are
    // exchanged without any search computation. Recall is meaningless in this mode.
    static bool CommOnly() {
        const char* v = std::getenv("GP_COMM_ONLY");
        return v && std::string(v) == "1";
    }

    void BuildInShardIndex() {
        if (CommOnly()) {
            shard_points.Drop();
            return;
        }
        space = std::make_unique<SpaceT>(dim);
        // parallel_timer: GP_SHARD_INDEX_DIR=<dir> loads this rank's shard graph from <dir> if it
        // is there (same graph on every run, no rebuild), otherwise builds it and saves it there.
        // The directory must be specific to the partition file; the file name has rank, n, M, efc.
        std::string index_file;
        if (const char* dir = std::getenv("GP_SHARD_INDEX_DIR"); dir && *dir) {
            index_file = std::string(dir) + "/shard_r" + std::to_string(rank) + "_n" + std::to_string(shard_points.n) +
                         "_M" + std::to_string(hnsw_parameters.M) + "_efc" + std::to_string(hnsw_parameters.ef_construction) + ".hnsw";
            if (std::filesystem::exists(index_file)) {
                hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), index_file);
                if (hnsw->cur_element_count == shard_points.n) {
                    // The file is read by one thread, so every page of the level-0 graph lands on that
                    // thread's NUMA node, while the parallel inserts of a build spread it over all
                    // nodes (search ~60% slower per query otherwise). Copy it in parallel so that a
                    // loaded graph has the same memory layout as a freshly built one.
                    const size_t bytes = hnsw->max_elements_ * hnsw->size_data_per_element_;
                    char* spread = static_cast<char*>(std::malloc(bytes));
                    const size_t chunk = size_t(1) << 20;
                    parlay::parallel_for(0, (bytes + chunk - 1) / chunk, [&](size_t c) {
                        const size_t b = c * chunk;
                        std::memcpy(spread + b, hnsw->data_level0_memory_ + b, std::min(chunk, bytes - b));
                    }, 1);
                    std::free(hnsw->data_level0_memory_);
                    hnsw->data_level0_memory_ = spread;
                    std::cout << "rank " << rank << " loaded shard index " << index_file << std::endl;
                    hnsw->setEf(hnsw_parameters.ef_search);
                    shard_points.Drop();
                    return;
                }
                std::cerr << "rank " << rank << " WARN: " << index_file << " has " << hnsw->cur_element_count
                          << " points, expected " << shard_points.n << "; rebuilding" << std::endl;
            }
        }
        hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), shard_points.n, hnsw_parameters.M, hnsw_parameters.ef_construction, /* random seed = */ 555);
        // parallel_timer: GP_DETERMINISTIC_BUILD=1 builds the graph sequentially (same graph
        // on every run), used only to check that the parallel search returns the same results.
        if (std::getenv("GP_DETERMINISTIC_BUILD") && std::string(std::getenv("GP_DETERMINISTIC_BUILD")) == "1") {
            for (size_t i = 0; i < shard_points.n; ++i) hnsw->addPoint(shard_points.GetPoint(i), i);
        } else {
            parlay::parallel_for(0, shard_points.n, [&](size_t i) { hnsw->addPoint(shard_points.GetPoint(i), i); });
        }
        if (!index_file.empty()) {
            const std::string tmp = index_file + ".tmp";
            hnsw->saveIndex(tmp);
            std::filesystem::rename(tmp, index_file);
            std::cout << "rank " << rank << " saved shard index " << index_file << std::endl;
        }
        hnsw->setEf(hnsw_parameters.ef_search);
        shard_points.Drop();    // TODO we could do the HNSW insert during IO --> halve the memory requirement
    }

    void LoadRouter(const std::string& hnsw_router_file) {
        const std::vector<int>* router_partition = &partition;
        const std::string routing_sidecar = hnsw_router_file + ".routing_index_partition";
        if (std::filesystem::exists(routing_sidecar)) {
            routing_index_partition = ReadMetisPartition(routing_sidecar);
            router_partition = &routing_index_partition;
        } else {
            routing_index_partition.clear();
        }
        router = std::make_unique<HNSWRouter>(hnsw_router_file, dim, *router_partition);
    }

    std::vector<int> Route(float* Q) {
        std::vector<int> probes = router->Query(Q, num_voting_neighbors).RoutingQuery();
        probes.resize(std::min<int>(probes.size(), num_probes));
        return probes;
    }

    std::vector<std::pair<int, NNVec>> ProcessQueries3(const std::vector<int>& query_ids, PointSet& queries) {
        // parallel_timer: per-phase timers ("#[prof ]" line per rank), steady_clock.
        //   route_work_s  = sum over QA threads of time in Route() (CPU work)
        //   route_wall_s  = wall time of the parallel routing regions
        //   post_req_s    = posting requests (main thread)
        //   search_work_s = sum over QA threads of time in resolve_request (HNSW search)
        //   search_wall_s = wall time of the parallel search regions
        //   req_phase_s   = requests_queue.terminate(): receiving requests, searching them,
        //                   posting responses (search_wall_s is part of it)
        //   resp_phase_s  = responses_queue.terminate(): receiving responses
        //   merge_s       = final owner-side merge (dedup + top-k)
        using prof_clock = std::chrono::steady_clock;
        auto prof_sec = [](prof_clock::time_point a, prof_clock::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };
        // Query answering (routing + local search) runs on QA_THREADS OpenMP threads (default
        // 16); index construction keeps using parlay on all cores, as in the original code.
        const char* qa_env = std::getenv("QA_THREADS");
        const int qa_threads = std::max(1, qa_env ? std::atoi(qa_env) : 16);
        const size_t prof_workers = static_cast<size_t>(qa_threads);
        std::vector<double> prof_route_w(prof_workers, 0.0), prof_search_w(prof_workers, 0.0);
        double prof_route_wall = 0, prof_post_req = 0, prof_search_wall = 0;
        size_t prof_search_calls = 0, prof_search_batches = 0;
        // parallel_timer: GP_LOG_BYTES=1 prints a "#[bytes]" line per rank with the encoded bytes
        // sent to each rank (requests, responses). With the default (unlimited) buffer thresholds
        // each non-zero entry is exactly one MPI message; input of comm_replay/gen_trace_gp.py.
        std::vector<unsigned long long> bytes_req_to(comm_size, 0), bytes_resp_to(comm_size, 0);
        const auto prof_t0 = prof_clock::now();
        auto merge_requests = [this](auto& buf, message_queue::PEID buffer_destination, message_queue::PEID my_rank,
                    message_queue::Envelope auto msg) {
            for (const Request& req : msg.message) {
                EncodeRequest(buf, req, dim);
            }
        };

        auto split_requests = [this](message_queue::MPIBuffer<float> auto const& buf, message_queue::PEID buffer_origin,
                    message_queue::PEID my_rank) {
            using EnvelopeT = message_queue::MessageEnvelope<std::vector<Request>>;
            std::vector<EnvelopeT> out;
            out.push_back(EnvelopeT{ .message = DecodeRequests(buf), .sender = buffer_origin, .receiver = my_rank, .tag = 0 });
            return out;
        };

        auto requests_queue =
                message_queue::make_buffered_queue<Request, float>(MPI_COMM_WORLD, merge_requests, split_requests);


        auto merge_responses = [](auto& buf, message_queue::PEID buffer_destination, message_queue::PEID my_rank,
                    message_queue::Envelope auto msg) {
            for (const Response& resp : msg.message) {
                EncodeResponse(buf, resp);
            }
        };

        auto split_responses = [](message_queue::MPIBuffer<float> auto const& buf, message_queue::PEID buffer_origin,
                    message_queue::PEID my_rank) {
            using EnvelopeT = message_queue::MessageEnvelope<std::vector<Response>>;
            std::vector<EnvelopeT> out;
            out.push_back(EnvelopeT{ .message = DecodeResponses(buf), .sender = buffer_origin, .receiver = my_rank, .tag = 0 });
            return out;
        };

        auto responses_queue =
                message_queue::make_buffered_queue<Response, float>(MPI_COMM_WORLD, merge_responses, split_responses);

        std::vector<Request> local_requests;
        std::unordered_map<int, int> expected_responses;
        std::unordered_map<int, int> received_responses;
        std::unordered_map<int, std::vector<NeighborHit>> partial_results;
        expected_responses.reserve(query_ids.size());
        received_responses.reserve(query_ids.size());
        partial_results.reserve(query_ids.size());

        // we want to
        // a) process queries in parallel
        // b) dont lock message buffers to avoid overheads
        // c) get the first requests out as soon as possible so that a machine that runs out of routing
        //    work can get started on retrieving neighbors right away
        // The following performs multiple steps with 2^^i queries routed in step i, followed by sending their messages

        size_t step_size = 128;
        auto qq = parlay::make_slice(query_ids);
        for (size_t i = 0; i < query_ids.size(); i += step_size, step_size *= 2) {
            const auto queries_this_step = qq.cut(i, std::min(query_ids.size(), i + step_size));
            const auto prof_r0 = prof_clock::now();
            std::vector<std::vector<int>> probes_nested(queries_this_step.size());
#pragma omp parallel for num_threads(qa_threads) schedule(dynamic, 16)
            for (size_t j = 0; j < queries_this_step.size(); ++j) {
                const auto t0 = prof_clock::now();
                probes_nested[j] = Route(queries.GetPoint(queries_this_step[j]));
                prof_route_w[omp_get_thread_num()] += prof_sec(t0, prof_clock::now());
            }
            const auto prof_p0 = prof_clock::now();
            prof_route_wall += prof_sec(prof_r0, prof_p0);
            //auto probes = parlay::flatten(probes_nested); // no point in doing the zip and flatten in parallel. we have to post messages sequentially anyway
            // well actually, merging the embedding streams in parallel could be nice
            for (size_t j = 0; j < queries_this_step.size(); ++j) {
                int query_id = queries_this_step[j];
                float* Q = queries.GetPoint(query_id);
                expected_responses[query_id] = static_cast<int>(probes_nested[j].size());
                received_responses[query_id] = 0;
                partial_results[query_id] = {};
                for (int shard_id : probes_nested[j]) {
                    Request req;
                    req.query_id = query_id;
                    req.origin_rank = rank;
                    req.target_shard = shard_id;
                    req.coordinates.assign(Q, Q + dim);
                    if (shard_id == rank) {
                        local_requests.push_back(std::move(req));
                    } else {
                        bytes_req_to[shard_id] += (5 + static_cast<unsigned long long>(dim)) * sizeof(float);
                        requests_queue.post_message(std::move(req), shard_id);
                    }
                }
            }
            prof_post_req += prof_sec(prof_p0, prof_clock::now());
        }

        const bool comm_only = CommOnly();
        auto resolve_request = [&](const Request& r) {
            Response response;
            response.query_id = r.query_id;
            response.origin_rank = r.origin_rank;
            response.responder_rank = rank;
            response.target_shard = r.target_shard;
            if (comm_only) {
                response.topk.assign(num_neighbors, NeighborHit{ .global_id = 0, .distance = 0.0f });
                return response;
            }
            auto result = hnsw->searchKnn(const_cast<float*>(r.coordinates.data()), num_neighbors);
            while (!result.empty()) {
                auto next = result.top();
                const uint32_t local_id = static_cast<uint32_t>(next.second);
                if (local_id >= local_to_global_ids.size()) {
                    throw std::runtime_error("Invalid local neighbor id while mapping to global id");
                }
                response.topk.push_back(NeighborHit{ .global_id = local_to_global_ids[local_id], .distance = next.first });
                result.pop();
            }
            return response;
        };

        auto accept_response = [&](const Response& response) {
            if (response.origin_rank != rank) {
                return;
            }
            auto expected_it = expected_responses.find(response.query_id);
            if (expected_it == expected_responses.end()) {
                throw std::runtime_error("Received response for unknown query id");
            }
            auto& buf = partial_results[response.query_id];
            buf.insert(buf.end(), response.topk.begin(), response.topk.end());
            received_responses[response.query_id] += 1;
        };

        // parallel_timer: search every request of an envelope in parallel (QA_THREADS OpenMP
        // threads, HNSW search only), then post the responses from the main thread. Same per-request
        // results as the sequential loop; only the order of posting is per envelope.
        auto resolve_batch = [&](const auto& reqs) {
            const size_t n = reqs.size();
            const auto w0 = prof_clock::now();
            std::vector<Response> responses(n);
#pragma omp parallel for num_threads(qa_threads) schedule(dynamic, 1)
            for (size_t i = 0; i < n; ++i) {
                const auto t0 = prof_clock::now();
                responses[i] = resolve_request(reqs[i]);
                prof_search_w[omp_get_thread_num()] += prof_sec(t0, prof_clock::now());
            }
            prof_search_wall += prof_sec(w0, prof_clock::now());
            prof_search_calls += n;
            prof_search_batches++;
            return responses;
        };

        size_t remote_requests_received = 0;
        auto return_neighbors = [&](message_queue::Envelope<Request> auto request_envelope) {
            const auto& reqs = request_envelope.message;
            remote_requests_received += reqs.size();
            auto responses = resolve_batch(reqs);
            for (size_t i = 0; i < responses.size(); ++i) {
                const int origin = reqs[i].origin_rank;
                bytes_resp_to[origin] += (6 + 2 * static_cast<unsigned long long>(responses[i].topk.size())) * sizeof(float);
                responses_queue.post_message(std::move(responses[i]), origin);
            }
        };

        auto accept_returned_neighbors = [&](message_queue::Envelope<Response> auto response_envelope) {
            for (const Response& response : response_envelope.message) {
                accept_response(response);
            }
        };

        const auto prof_q0 = prof_clock::now();
        requests_queue.terminate(return_neighbors);
        const auto prof_q1 = prof_clock::now();

        const size_t local_requests_resolved = local_requests.size();
        {
            auto local_responses = resolve_batch(local_requests);
            for (const Response& response : local_responses) {
                accept_response(response);
            }
        }

        const auto prof_s0 = prof_clock::now();
        responses_queue.terminate(accept_returned_neighbors);
        const auto prof_s1 = prof_clock::now();

        // Load-balance stats: HNSW searches executed on this rank (not query ownership).
        const unsigned long long local_ull = static_cast<unsigned long long>(local_requests_resolved);
        const unsigned long long remote_ull = static_cast<unsigned long long>(remote_requests_received);
        const unsigned long long searches_ull = local_ull + remote_ull;
        const unsigned long long owned_ull = static_cast<unsigned long long>(query_ids.size());

        std::vector<unsigned long long> all_local(static_cast<size_t>(comm_size));
        std::vector<unsigned long long> all_remote(static_cast<size_t>(comm_size));
        std::vector<unsigned long long> all_searches(static_cast<size_t>(comm_size));
        std::vector<unsigned long long> all_owned(static_cast<size_t>(comm_size));
        MPI_Gather(&local_ull, 1, MPI_UNSIGNED_LONG_LONG, all_local.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&remote_ull, 1, MPI_UNSIGNED_LONG_LONG, all_remote.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&searches_ull, 1, MPI_UNSIGNED_LONG_LONG, all_searches.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&owned_ull, 1, MPI_UNSIGNED_LONG_LONG, all_owned.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            unsigned long long min_s = std::numeric_limits<unsigned long long>::max();
            unsigned long long max_s = 0;
            unsigned long long sum_s = 0;
            std::cout << "Search load per rank "
                      << "(searches_on_rank = remote_requests_received + local_requests_resolved):"
                      << std::endl;
            for (int r = 0; r < comm_size; ++r) {
                const unsigned long long s = all_searches[static_cast<size_t>(r)];
                min_s = std::min(min_s, s);
                max_s = std::max(max_s, s);
                sum_s += s;
                std::cout << "  rank " << r
                          << ": owned_queries=" << all_owned[static_cast<size_t>(r)]
                          << " remote_requests_received=" << all_remote[static_cast<size_t>(r)]
                          << " local_requests_resolved=" << all_local[static_cast<size_t>(r)]
                          << " searches_on_rank=" << s
                          << std::endl;
            }
            const double avg_s = static_cast<double>(sum_s) / static_cast<double>(comm_size);
            const double imbalance = (min_s == 0) ? 0.0 : static_cast<double>(max_s) / static_cast<double>(min_s);
            std::cout << "Search load summary: total_searches=" << sum_s
                      << " min=" << min_s
                      << " max=" << max_s
                      << " avg=" << avg_s
                      << " max/min=" << imbalance
                      << std::endl;
        }

        size_t completed = 0;
        for (int query_id : query_ids) {
            const int expected = expected_responses[query_id];
            const int received = received_responses[query_id];
            if (expected == received) {
                completed++;
            }
        }
        if (completed != query_ids.size()) {
            throw std::runtime_error("Distributed query processing incomplete: missing responses");
        }

        const auto prof_m0 = prof_clock::now();
        // Final owner-side merge: dedup by global id, keep best distance, then top-k.
        std::unordered_map<int, std::vector<NeighborHit>> merged_results;
        merged_results.reserve(query_ids.size());
        for (int query_id : query_ids) {
            auto partial_it = partial_results.find(query_id);
            if (partial_it == partial_results.end()) {
                merged_results[query_id] = {};
                continue;
            }

            std::unordered_map<uint32_t, float> best_distance_by_id;
            best_distance_by_id.reserve(partial_it->second.size());
            for (const NeighborHit& hit : partial_it->second) {
                auto it = best_distance_by_id.find(hit.global_id);
                if (it == best_distance_by_id.end()) {
                    best_distance_by_id.emplace(hit.global_id, hit.distance);
                } else if (hit.distance < it->second) {
                    it->second = hit.distance;
                }
            }

            std::vector<NeighborHit> merged;
            merged.reserve(best_distance_by_id.size());
            for (const auto& [global_id, distance] : best_distance_by_id) {
                merged.push_back(NeighborHit{ .global_id = global_id, .distance = distance });
            }

            std::sort(merged.begin(), merged.end(), [](const NeighborHit& a, const NeighborHit& b) {
                if (a.distance != b.distance) return a.distance < b.distance;
                return a.global_id < b.global_id;
            });

            if (static_cast<int>(merged.size()) > num_neighbors) {
                merged.resize(num_neighbors);
            }
            merged_results[query_id] = std::move(merged);
        }

        // Keep the variable used until result serialization/evaluation is added.
        size_t total_merged_hits = 0;
        for (const auto& [query_id, hits] : merged_results) {
            total_merged_hits += hits.size();
        }
        if (!query_ids.empty() && total_merged_hits == 0) {
            throw std::runtime_error("Distributed query processing produced no merged results");
        }
        {
            const auto prof_m1 = prof_clock::now();
            double route_work = 0, search_work = 0;
            for (double v : prof_route_w) route_work += v;
            for (double v : prof_search_w) search_work += v;
            char host[256];
            if (gethostname(host, sizeof(host)) != 0) std::snprintf(host, sizeof(host), "?");
            host[sizeof(host) - 1] = 0;
            char line[1024];
            std::snprintf(line, sizeof(line),
                "#[prof ] rank%d host=%s threads=%zu owned_queries=%zu searches=%zu search_batches=%zu avg_batch=%.1f "
                "total_s=%.6f route_work_s=%.6f route_wall_s=%.6f post_req_s=%.6f search_work_s=%.6f search_wall_s=%.6f "
                "req_phase_s=%.6f resp_phase_s=%.6f merge_s=%.6f\n",
                rank, host, prof_workers, query_ids.size(), prof_search_calls, prof_search_batches,
                prof_search_calls / (double) std::max<size_t>(1, prof_search_batches),
                prof_sec(prof_t0, prof_m1), route_work, prof_route_wall, prof_post_req, search_work, prof_search_wall,
                prof_sec(prof_q0, prof_q1), prof_sec(prof_s0, prof_s1), prof_sec(prof_m0, prof_m1));
            std::fputs(line, stdout);
            if (const char* lb = std::getenv("GP_LOG_BYTES"); lb && std::string(lb) == "1") {
                std::string b = "#[bytes] rank" + std::to_string(rank) + " req_to=";
                for (int r = 0; r < comm_size; ++r) b += (r ? "," : "") + std::to_string(bytes_req_to[r]);
                b += " resp_to=";
                for (int r = 0; r < comm_size; ++r) b += (r ? "," : "") + std::to_string(bytes_resp_to[r]);
                b += "\n";
                std::fputs(b.c_str(), stdout);
            }
            std::fflush(stdout);
        }

        std::vector<std::pair<int, NNVec>> local_query_results;
        local_query_results.reserve(query_ids.size());

        for (int query_id : query_ids) {
            const auto it = merged_results.find(query_id);
            if (it == merged_results.end()) {
                local_query_results.emplace_back(query_id, NNVec{});
                continue;
            }

            NNVec nn;
            nn.reserve(it->second.size());
            for (const auto& hit : it->second) {
                nn.emplace_back(hit.distance, hit.global_id);
            }
            local_query_results.emplace_back(query_id, std::move(nn));
        }
        return local_query_results;
    }

};
