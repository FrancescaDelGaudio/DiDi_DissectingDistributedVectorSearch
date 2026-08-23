#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace faiss {
class Error_sys;
class IndexIVF;
using idx_t = int64_t;
}  // namespace faiss

namespace auncel_dist {

struct HyperparamRow {
    float multipler = 1.0f;
    float std_m = 1.0f;
};

struct ShardCalibrateConfig {
    bool enabled = true;
    float target_recall = 0.9f;
    // Validate (a,b) on queries [holdout_start, nq); train traces on [0, holdout_start).
    double holdout_frac = 0.5;
    std::string hyperparam_file;
};

struct ShardCalibrateResult {
    int best_id = 0;  // 1-based row in hyperparameter file; 0 = none chosen
    float multipler = 1.0f;
    float std_m = 1.0f;
    float min_recall = 0.0f;
    size_t holdout_start = 0;
    size_t holdout_nq = 0;
};

bool load_hyperparameter_rows(
        const std::string& path,
        std::vector<HyperparamRow>* rows,
        std::string* err);

std::string default_hyperparameter_file();

// Sweep hyperparameter rows; pick (a,b) with best min recall@k on holdout queries.
ShardCalibrateResult calibrate_shard_hyperparams(
        faiss::IndexIVF* ivf,
        faiss::Error_sys* err_sys,
        const float* xq,
        size_t nq,
        size_t k,
        size_t d,
        const faiss::idx_t* gt_ids,
        const ShardCalibrateConfig& cfg,
        const std::vector<HyperparamRow>& rows);

}  // namespace auncel_dist
