#pragma once

#include "defs.h"
#include "dist.h"
#include "topn.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <parlay/parallel.h>

std::vector<float> ComputeDistanceToKthNeighbor(PointSet& points, PointSet& queries, int k) {
    std::vector<float> d(queries.n);
    parlay::parallel_for(0, queries.n, [&](size_t i) {
        TopN top_k(k);
        float* Q = queries.GetPoint(i);
        for (uint32_t j = 0; j < points.n; ++j) {
            float* P = points.GetPoint(j);
            float dist = distance(P, Q, points.d);
            top_k.Add(std::make_pair(dist, j));
        }
        d[i] = top_k.Top().first;
    }, 1);
    return d;
}

std::vector<NNVec> ComputeGroundTruth(PointSet& points, PointSet& queries, int k) {
    std::vector<NNVec> res(queries.n);
    parlay::parallel_for(0, queries.n, [&](size_t i) {
        TopN top_k(k);
        float* Q = queries.GetPoint(i);
        for (uint32_t j = 0; j < points.n; ++j) {
            float* P = points.GetPoint(j);
            float dist = distance(P, Q, points.d);
            top_k.Add(std::make_pair(dist, j));
        }
        res[i] = top_k.Take();
        std::sort(res[i].begin(), res[i].end());    // should be = std::reverse
    }, 1);
    return res;
}

void OracleRecall(const std::vector<NNVec>& ground_truth, const std::vector<int>& partition, int num_neighbors) {
    int num_shards = NumPartsInPartition(partition);
    std::vector<size_t> hits(num_shards, 0);

    for (const auto& neigh : ground_truth) {
        std::vector<std::pair<int, int>> freq(num_shards);
        for (int i = 0; i < num_shards; ++i) {
            freq[i].first = 0;
            freq[i].second = i;
        }
        for (int i = 0; i < num_neighbors; ++i) {
            freq[partition[neigh[i].second]].first++;
        }

        std::sort(freq.begin(), freq.end(), std::greater<>());
        for (int i = 0; i < num_shards; ++i) {
            hits[i] += freq[i].first;
        }
    }

    size_t total = 0;
    for (int i = 0; i < num_shards && hits[i] > 0; ++i) {
        total += hits[i];
        double recall = static_cast<double>(total) / ground_truth.size() / num_neighbors;
        std::cout << "nprobes = " << i + 1 << " oracle recall = " << recall << std::endl;
    }

}


/**
 * This function also checks whether the computed distances and order in the ground truth are correct. If not, it will emit a warning and reorder the candidates.
 */
std::vector<float> ConvertGroundTruthToDistanceToKthNeighbor(std::vector<NNVec>& ground_truth, int k, PointSet& points, PointSet& queries) {
    if (ground_truth.size() != queries.n) {
        std::cout << "Ground truth size and number of queries don't match." << std::endl;
        std::exit(0);
    }
    std::vector<float> distance_to_kth_neighbor(ground_truth.size());

    parlay::parallel_for(0, queries.n, [&](size_t q) {
        auto& neighs = ground_truth[q];
        auto comp = [](const std::pair<float, uint32_t>& l, const std::pair<float, uint32_t>& r) {
            return l.first < r.first;
        };

        float* Q = queries.GetPoint(q);
        for (int j = 0; j < k; ++j) {
            uint32_t point_id = neighs[j].second;
            float true_dist = distance(points.GetPoint(point_id), Q, points.d);
            neighs[j].first = true_dist;
        }

        if (!std::is_sorted(neighs.begin(), neighs.begin() + k, comp)) {
            std::sort(neighs.begin(), neighs.begin() + k, comp);
        }
        distance_to_kth_neighbor[q] = neighs[k-1].first;
    });

    return distance_to_kth_neighbor;
}

double Recall(const std::vector<NNVec>& neighbors_per_query, const std::vector<float>& distance_to_kth_neighbor, int k) {
    size_t hits = 0;
    for (size_t i = 0; i < neighbors_per_query.size(); ++i) {
        for (const auto& x : neighbors_per_query[i]) {
            if (x.first <= distance_to_kth_neighbor[i]) {
                hits++;
            }
        }
    }
    double recall = static_cast<double>(hits) / static_cast<double>(neighbors_per_query.size() * k);
    return recall;
}
