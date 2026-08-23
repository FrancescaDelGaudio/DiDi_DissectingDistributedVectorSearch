#pragma once

#include <cstdio>
#include <string>

#include "../IndexIVF.h"
#include "../index_io.h"
#include "../profile.h"

namespace auncel_dist {

struct BundleMeta {
    size_t train_nq = 0;
    size_t k = 0;
    size_t d = 0;
    size_t nb = 0;
    std::string index_key;
    // Per-shard (a,b) calibration written by build_shard (0 = not calibrated).
    int hyperparam_id = 0;
    float multipler = 1.0f;
    float std_m = 1.0f;
    int calibrated = 0;
};

std::string faissindex_path(const std::string& prefix);
std::string profile_path(const std::string& prefix);
std::string meta_path(const std::string& prefix);

void recompute_interdis_cem(faiss::IndexIVF* ix);

void write_error_profile(const faiss::error_pro* profile, const char* path);
faiss::error_pro* read_error_profile(const char* path);

void write_meta(const BundleMeta& meta, const char* path);
BundleMeta read_meta(const char* path);

void save_bundle(
        faiss::Index* index,
        const BundleMeta& meta,
        const std::string& prefix);

faiss::Index* load_bundle(
        const std::string& prefix,
        BundleMeta* meta_out);

void attach_error_sys(
        faiss::Error_sys* err_sys,
        faiss::Index* index,
        const BundleMeta& meta);

// Tune-mode search indexes train_D by query id; must cover all nq queries at search time.
void bind_profile_ground_truth(
        faiss::error_pro* profile,
        size_t train_nq,
        size_t max_topk,
        const float* gt_dis,
        const faiss::Index::idx_t* gt_ids);

bool bundle_exists(const std::string& prefix);

}  // namespace auncel_dist
