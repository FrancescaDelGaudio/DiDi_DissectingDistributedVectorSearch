#pragma once

#include "defs.h"

struct OverlapMemoryBudget {
    bool enabled = false;
    size_t bytes = 0;
};

Clusters OverlappingGraphPartitioning(
    PointSet& points,
    int requested_num_clusters,
    double epsilon,
    double overlap,
    bool strong,
    bool fixed_num_clusters = false,
    OverlapMemoryBudget memory_budget = {});

void MakeOverlappingWithCentroids(PointSet& points, Clusters& clusters, size_t max_cluster_size, size_t num_extra_assignments);

Clusters OverlappingKMeansPartitioningSPANN(PointSet& points, const Partition& partition, int requested_num_clusters, double epsilon, double overlap);
