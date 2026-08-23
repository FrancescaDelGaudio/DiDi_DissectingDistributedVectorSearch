#pragma once

#include "points_io.h"
#include "metis_io.h"

#include "hnsw_router.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"

#include "../external/message-queue/include/message-queue/buffered_queue.hpp"

#include <parlay/primitives.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

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

    void BuildInShardIndex() {
        space = std::make_unique<SpaceT>(dim);
        hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), shard_points.n, hnsw_parameters.M, hnsw_parameters.ef_construction, /* random seed = */ 555);
        parlay::parallel_for(0, shard_points.n, [&](size_t i) { hnsw->addPoint(shard_points.GetPoint(i), i); });
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
            auto probes_nested = parlay::map(queries_this_step, [&](int query_id) { return Route(queries.GetPoint(query_id)); });
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
                        requests_queue.post_message(std::move(req), shard_id);
                    }
                }
            }
        }

        auto resolve_request = [&](const Request& r) {
            auto result = hnsw->searchKnn(const_cast<float*>(r.coordinates.data()), num_neighbors);
            Response response;
            response.query_id = r.query_id;
            response.origin_rank = r.origin_rank;
            response.responder_rank = rank;
            response.target_shard = r.target_shard;
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

        auto return_neighbors = [&](message_queue::Envelope<Request> auto request_envelope) {
            for (const Request& r : request_envelope.message) {
                Response response = resolve_request(r);
                responses_queue.post_message(std::move(response), r.origin_rank);
            }
        };

        auto accept_returned_neighbors = [&](message_queue::Envelope<Response> auto response_envelope) {
            for (const Response& response : response_envelope.message) {
                accept_response(response);
            }
        };

        requests_queue.terminate(return_neighbors);

        for (const Request& req : local_requests) {
            Response response = resolve_request(req);
            accept_response(response);
        }

        responses_queue.terminate(accept_returned_neighbors);

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
