//
// Algorithm 5 (Pyramid): MIPS index construction (steps 3-15).
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>
#include <string>
#include <sys/stat.h>

#include <omp.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include "dhnswlib/time.hpp"
#include "distributed/partition.hpp"
#include "hnswlib/hnswalg.h"
#include "hnswlib/space_ip.h"
#include "matrix.hpp"
#include "trainer/trainer.hpp"

namespace dhnsw {

inline void ensure_parent_dir(const std::string &file_path) {
    const std::size_t pos = file_path.find_last_of('/');
    if (pos == std::string::npos) {
        return;
    }
    const std::string dir = file_path.substr(0, pos);
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        std::cerr << "[mips_trainer]# failed to create directory " << dir
                  << ": " << std::strerror(errno) << std::endl;
        std::exit(1);
    }
}

inline void normalize_inplace(float *v, int dim) {
    float norm_sq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        norm_sq += v[j] * v[j];
    }
    if (norm_sq <= 0.0f) {
        return;
    }
    float inv = 1.0f / std::sqrt(norm_sq);
    for (int j = 0; j < dim; ++j) {
        v[j] *= inv;
    }
}

inline void sample_indices(int dataset_size, int sample_size, std::vector<int> &indices) {
    indices.resize(dataset_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(1234);
    if (sample_size < dataset_size) {
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(sample_size);
    }
}

inline void spherical_kmeans_centroids(
    const float *samples,
    int sample_size,
    int dim,
    int num_centroid,
    std::vector<std::vector<float> > &centroids) {
    faiss::Clustering clustering(dim, num_centroid);
    clustering.spherical = true;
    clustering.niter = 25;
    clustering.verbose = true;
    clustering.seed = 1234;
    clustering.min_points_per_centroid = 1;

    faiss::IndexFlatIP index(dim);
    clustering.train(sample_size, samples, index);

    centroids.resize(num_centroid);
    for (int i = 0; i < num_centroid; ++i) {
        centroids[i].resize(dim);
        for (int j = 0; j < dim; ++j) {
            centroids[i][j] = clustering.centroids[i * dim + j];
        }
        normalize_inplace(centroids[i].data(), dim);
    }
}

inline void populate_meta_hnsw(
    const std::vector<std::vector<float> > &centroids,
    hnswlib::HierarchicalNSW<float> &meta) {
    for (int i = 0; i < 1; ++i) {
        meta.addPoint((void *)centroids[i].data(), (size_t)i);
    }
#pragma omp parallel for
    for (int i = 1; i < (int)centroids.size(); ++i) {
        meta.addPoint((void *)centroids[i].data(), (size_t)i);
    }
}

inline void assign_primary_clusters(
    ss::Matrix<float> &data,
    hnswlib::HierarchicalNSW<float> &meta,
    int hnsw_ef_cons,
    std::vector<std::vector<int> > &clusters,
    std::vector<int> &partition_map,
    const std::vector<int> &centroid_to_shard) {
    const int n = data.getSize();
    const int m = (int)centroid_to_shard.size();
    clusters.assign(m, std::vector<int>());
    partition_map.assign(n, -1);
    meta.setEf(hnsw_ef_cons);

#pragma omp parallel for schedule(dynamic, 256)
    for (int x = 0; x < n; ++x) {
        auto result = meta.searchKnn((void *)data[x], 1);
        int nearest = (int)result.top().second;
        result.pop();
#pragma omp critical
        {
            clusters[nearest].push_back(x);
            partition_map[x] = centroid_to_shard[nearest];
        }
    }
}

inline void replicate_top_mips_neighbors(
    ss::Matrix<float> &data,
    const std::vector<std::vector<float> > &centroids,
    int replica_r,
    std::vector<std::vector<int> > &clusters) {
    const int n = data.getSize();
    const int m = (int)centroids.size();
    const int dim = data.getDim();
    if (replica_r <= 0) {
        return;
    }

    faiss::IndexFlatIP index(dim);
    index.add(n, data.getData());

    std::vector<float> distances((size_t)m * replica_r);
    std::vector<faiss::idx_t> labels((size_t)m * replica_r);
    std::vector<float> centroid_flat((size_t)m * dim);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < dim; ++j) {
            centroid_flat[i * dim + j] = centroids[i][j];
        }
    }

    index.search(m, centroid_flat.data(), replica_r, distances.data(), labels.data());

    std::vector<std::unordered_set<int> > cluster_sets(m);
    for (int c = 0; c < m; ++c) {
        cluster_sets[c].insert(clusters[c].begin(), clusters[c].end());
    }

    for (int c = 0; c < m; ++c) {
        for (int j = 0; j < replica_r; ++j) {
            int id = (int)labels[c * replica_r + j];
            if (id < 0 || id >= n) {
                continue;
            }
            if (cluster_sets[c].insert(id).second) {
                clusters[c].push_back(id);
            }
        }
    }
}

inline void single_machine_mips_trainer(
    int sample_size_n0,
    int num_centroid_m,
    int num_subhnsw_w,
    int replica_r,
    const std::string &data_path,
    const std::string &centroid_path,
    const std::string &map_path,
    const std::string &partition_map_path,
    const std::string &hnsw_path,
    const std::string &cluster_path,
    mt::Partition &partition,
    int hnsw_m = 32,
    int hnsw_ef_cons = 100) {
    omp_set_num_threads(31);

    ss::Matrix<float> data(data_path);
    const int dim = data.getDim();
    const int n = data.getSize();
    if (sample_size_n0 <= 0 || sample_size_n0 > n) {
        sample_size_n0 = std::min(n, std::max(num_centroid_m * 10, 100000));
    }
    if (replica_r < 0) {
        replica_r = 0;
    }

    ensure_parent_dir(centroid_path);

    dhnsw::MiliTimer timer;
    timer.update_time();

    std::vector<int> sample_ids;
    sample_indices(n, sample_size_n0, sample_ids);
    std::vector<float> samples((size_t)sample_size_n0 * dim);
    for (int i = 0; i < sample_size_n0; ++i) {
        float *dst = &samples[(size_t)i * dim];
        const float *src = data[sample_ids[i]];
        for (int j = 0; j < dim; ++j) {
            dst[j] = src[j];
        }
        normalize_inplace(dst, dim);
    }
    std::cout << "[mips_trainer]# sampled " << sample_size_n0 << " items in "
              << timer.span_and_update() << " ms" << std::endl;

    std::vector<std::vector<float> > centroids;
    spherical_kmeans_centroids(samples.data(), sample_size_n0, dim, num_centroid_m, centroids);
    assert((int)centroids.size() == num_centroid_m);
    std::cout << "[mips_trainer]# spherical k-means done in " << timer.span_and_update()
              << " ms" << std::endl;

    save_centroids(centroid_path, centroids);

    hnswlib::InnerProductSpace ip_space(dim);
    hnswlib::HierarchicalNSW<float> meta(&ip_space, centroids.size(), hnsw_m, hnsw_ef_cons);
    populate_meta_hnsw(centroids, meta);
    std::cout << "[mips_trainer]# meta-HNSW built in " << timer.span_and_update() << " ms"
              << std::endl;

    std::vector<std::vector<int> > graph;
    int num_edges = meta.getLevel0Graph(graph);
    std::vector<int> map = partition.getPartition(graph, centroids, num_edges, num_subhnsw_w);
    std::cout << "[mips_trainer]# graph partitioned in " << timer.span_and_update() << " ms"
              << std::endl;

    std::vector<std::vector<int> > clusters;
    std::vector<int> partition_map;
    assign_primary_clusters(data, meta, hnsw_ef_cons, clusters, partition_map, map);
    std::cout << "[mips_trainer]# primary MIPS assignment done in " << timer.span_and_update()
              << " ms" << std::endl;

    replicate_top_mips_neighbors(data, centroids, replica_r, clusters);
    std::cout << "[mips_trainer]# MIPS replication (r=" << replica_r << ") done in "
              << timer.span_and_update() << " ms" << std::endl;

    meta.saveIndex(hnsw_path);
    save_cluster(clusters, cluster_path);
    save_map(map_path, map, num_subhnsw_w);
    save_map(partition_map_path, partition_map, num_subhnsw_w);

    std::cout << "[mips_trainer]# MIPS index artifacts written to output directory" << std::endl;
}

} // namespace dhnsw
