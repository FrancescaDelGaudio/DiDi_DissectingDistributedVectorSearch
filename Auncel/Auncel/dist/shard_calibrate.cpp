#include "shard_calibrate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "../FaissAssert.h"
#include "../Index.h"
#include "../IndexIVF.h"
#include "../profile.h"

#define DC(classname) classname* ix = dynamic_cast<classname*>(ivf)

namespace auncel_dist {

namespace {

void apply_row(faiss::error_pro* t, const HyperparamRow& row) {
    t->multipler = row.multipler;
    t->std_m = row.std_m;
    t->profile = false;
}

float eval_holdout_min_recall(
        faiss::Error_sys* err_sys,
        faiss::IndexIVF* ivf,
        const float* xq,
        size_t nq,
        size_t k,
        size_t d,
        size_t holdout_start,
        size_t holdout_nq,
        float target_recall,
        const faiss::idx_t* gt_ids) {
    DC(faiss::IndexIVF);
    FAISS_THROW_IF_NOT(ix && ix->t);

    err_sys->set_topk(k);
    std::vector<float> acc(nq, target_recall);
    err_sys->set_queries(nq, xq, acc.data(), nq);

    std::vector<float> D(holdout_nq * k);
    std::vector<int64_t> I(holdout_nq * k);

    for (size_t qi = 0; qi < holdout_nq; qi++) {
        const size_t global_q = holdout_start + qi;
        err_sys->search(D.data() + qi * k, I.data() + qi * k, global_q, 1);
    }

    float min_recall = 1.0f;
    for (size_t qi = 0; qi < holdout_nq; qi++) {
        const float r = err_sys->recall(
                reinterpret_cast<faiss::Index::idx_t*>(I.data() + qi * k),
                const_cast<faiss::Index::idx_t*>(
                        reinterpret_cast<const faiss::Index::idx_t*>(
                                gt_ids + (holdout_start + qi) * k)),
                k);
        min_recall = std::min(min_recall, r);
    }
    return min_recall;
}

}  // namespace

std::string default_hyperparameter_file() {
    const char* env = std::getenv("AUNCEL_HYPERPARAM_FILE");
    if (env && env[0])
        return env;
    return "../hyperparameter.txt";
}

bool load_hyperparameter_rows(
        const std::string& path,
        std::vector<HyperparamRow>* rows,
        std::string* err) {
    if (!rows) {
        if (err)
            *err = "rows is null";
        return false;
    }
    rows->clear();
    std::ifstream in(path);
    if (!in) {
        if (err)
            *err = "could not open " + path;
        return false;
    }
    HyperparamRow row;
    while (in >> row.multipler >> row.std_m)
        rows->push_back(row);
    if (rows->empty()) {
        if (err)
            *err = "no rows in " + path;
        return false;
    }
    return true;
}

ShardCalibrateResult calibrate_shard_hyperparams(
        faiss::IndexIVF* ivf,
        faiss::Error_sys* err_sys,
        const float* xq,
        size_t nq,
        size_t k,
        size_t d,
        const faiss::idx_t* gt_ids,
        const ShardCalibrateConfig& cfg,
        const std::vector<HyperparamRow>& rows) {
    ShardCalibrateResult out;
    FAISS_THROW_IF_NOT(ivf && err_sys && xq && gt_ids);
    FAISS_THROW_IF_NOT(err_sys->is_trained);

    size_t holdout_nq = static_cast<size_t>(std::llround(cfg.holdout_frac * nq));
    if (holdout_nq < 1)
        holdout_nq = 1;
    if (holdout_nq >= nq)
        holdout_nq = nq / 2 > 0 ? nq / 2 : 1;
    const size_t holdout_start = nq - holdout_nq;
    out.holdout_start = holdout_start;
    out.holdout_nq = holdout_nq;

    printf("Shard hyperparam calibration: nq=%zu holdout=[%zu,%zu) target_recall=%.4f rows=%zu\n",
           nq, holdout_start, holdout_start + holdout_nq, cfg.target_recall,
           rows.size());

    int best_id = 0;
    float best_min = -1.0f;
    bool best_meets = false;

    for (size_t i = 0; i < rows.size(); i++) {
        apply_row(ivf->t, rows[i]);
        const float min_r = eval_holdout_min_recall(
                err_sys, ivf, xq, nq, k, d, holdout_start, holdout_nq,
                cfg.target_recall, gt_ids);
        const bool meets = min_r + 1e-6f >= cfg.target_recall;
        printf("  row %zu: multipler=%.3f std_m=%.3f -> min_recall=%.4f%s\n",
               i + 1, rows[i].multipler, rows[i].std_m, min_r,
               meets ? " (ok)" : "");

        const bool better = (best_id == 0) ||
                (meets && !best_meets) ||
                (meets == best_meets && min_r > best_min);
        if (better) {
            best_id = static_cast<int>(i + 1);
            best_min = min_r;
            best_meets = meets;
            out.multipler = rows[i].multipler;
            out.std_m = rows[i].std_m;
        }
    }

    if (best_id == 0) {
        out.multipler = 1.0f;
        out.std_m = 1.0f;
        HyperparamRow defaults;
        defaults.multipler = out.multipler;
        defaults.std_m = out.std_m;
        apply_row(ivf->t, defaults);
        printf("Shard calibration: no row selected; using defaults (1, 1)\n");
        return out;
    }

    out.best_id = best_id;
    out.min_recall = best_min;
    HyperparamRow chosen;
    chosen.multipler = out.multipler;
    chosen.std_m = out.std_m;
    apply_row(ivf->t, chosen);
    printf("Shard calibration chose row %d: multipler=%.3f std_m=%.3f min_recall=%.4f\n",
           out.best_id, out.multipler, out.std_m, out.min_recall);
    if (!best_meets)
        printf("  warning: best row still below target_recall %.4f\n",
               cfg.target_recall);
    return out;
}

}  // namespace auncel_dist
