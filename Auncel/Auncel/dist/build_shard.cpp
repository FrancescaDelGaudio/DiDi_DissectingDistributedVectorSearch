// Offline per-shard indexer: IVF train + add + error-profile train -> disk bundle.
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <sys/time.h>

#include "../AutoTune.h"
#include "../IndexIVF.h"
#include "dist_defaults.h"
#include "dist_io.h"
#include "shard_bundle.h"
#include "shard_calibrate.h"

#define DC(classname) classname* ix = dynamic_cast<classname*>(index)

namespace {

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --out-prefix <path/prefix> --db <shard.fbin> "
        "--query <query.fbin> --gt <gt.bin|gti.ivecs> [--gtd <gtd.fvecs>] "
        "[--index-key IVF1024,Flat] [--omp-threads N]\n"
        "  [--no-calibrate] [--calibration-recall 0.9] [--calibration-holdout-frac 0.5]\n"
        "  [--hyperparam-file path/to/hyperparameter.txt]\n"
        "\n"
        "Writes:\n"
        "  <prefix>.faissindex  Faiss IVF index with vectors\n"
        "  <prefix>.profile     Auncel error-profile traces\n"
        "  <prefix>.meta        train_nq, k, d, nb, index_key\n"
        "\n"
        "Environment overrides: AUNCEL_BASE, AUNCEL_QUERY, AUNCEL_GT, "
        "AUNCEL_INDEX_KEY, OMP_NUM_THREADS\n",
        prog);
}

struct Args {
    std::string out_prefix;
    std::string db;
    std::string query;
    std::string gtI;
    std::string gtD;
    std::string index_key;
    int omp_threads = 4;
    bool calibrate = true;
    float calibration_recall = 0.9f;
    double calibration_holdout_frac = 0.5;
    std::string hyperparam_file;
};

Args parse_args(int argc, char** argv) {
    Args a;
    a.db = auncel_dist::default_base();
    a.query = auncel_dist::default_query();
    a.gtI = auncel_dist::default_gt();
    a.index_key = std::getenv("AUNCEL_INDEX_KEY")
        ? std::getenv("AUNCEL_INDEX_KEY") : "IVF8,Flat";
    if (const char* t = std::getenv("OMP_NUM_THREADS"))
        a.omp_threads = std::atoi(t);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", flag);
                usage(argv[0]);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "--out-prefix") {
            a.out_prefix = need("--out-prefix");
        } else if (arg == "--db") {
            a.db = need("--db");
        } else if (arg == "--query") {
            a.query = need("--query");
        } else if (arg == "--gt") {
            a.gtI = need("--gt");
        } else if (arg == "--gtd") {
            a.gtD = need("--gtd");
        } else if (arg == "--index-key") {
            a.index_key = need("--index-key");
        } else if (arg == "--omp-threads") {
            a.omp_threads = std::atoi(need("--omp-threads").c_str());
        } else if (arg == "--no-calibrate") {
            a.calibrate = false;
        } else if (arg == "--calibration-recall") {
            a.calibration_recall = std::stof(need("--calibration-recall"));
        } else if (arg == "--calibration-holdout-frac") {
            a.calibration_holdout_frac = std::stod(need("--calibration-holdout-frac"));
        } else if (arg == "--hyperparam-file") {
            a.hyperparam_file = need("--hyperparam-file");
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            std::exit(1);
        }
    }
    if (a.out_prefix.empty()) {
        fprintf(stderr, "--out-prefix is required\n");
        usage(argv[0]);
        std::exit(1);
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    omp_set_num_threads(args.omp_threads);

    double t0 = elapsed();
    size_t d = 0, nt = 0;
    printf("[%.3f s] Loading train set %s\n", elapsed() - t0, args.db.c_str());
    float* xt = load_vectors(args.db, &d, &nt);

    if (args.index_key == "IVF1024,Flat" && nt < 1024) {
        fprintf(stderr,
            "IVF1024 needs >=1024 vectors (got %zu). "
            "Use IVF8,Flat for toy shards or omit --index-key.\n",
            nt);
        return 1;
    }

    printf("[%.3f s] Building index \"%s\" d=%zu nt=%zu\n",
           elapsed() - t0, args.index_key.c_str(), d, nt);
    faiss::Index* index = faiss::index_factory(int(d), args.index_key.c_str());
    if (dynamic_cast<faiss::IndexIVF*>(index) == nullptr) {
        fprintf(stderr, "index_key must produce IndexIVF\n");
        return 1;
    }

    printf("[%.3f s] Training IVF\n", elapsed() - t0);
    index->set_tune_mode();
    index->train(nt, xt);
    index->set_tune_off();
    delete[] xt;

    printf("[%.3f s] Adding %zu vectors\n", elapsed() - t0, nt);
    size_t nb = 0, d2 = 0;
    float* xb = load_vectors(args.db, &d2, &nb);
    assert(d == d2);
    index->add(nb, xb);
    delete[] xb;

    printf("[%.3f s] Loading queries %s\n", elapsed() - t0, args.query.c_str());
    size_t nq = 0, dq = 0;
    float* xq = load_vectors(args.query, &dq, &nq);
    assert(d == dq);
    assert(nq % 10 == 0 || !"Error_sys requires nq divisible by 10");

    std::vector<faiss::Index::idx_t> gt_ids;
    std::vector<float> gt_dis;
    size_t k = 0, gt_nq = 0;
    printf("[%.3f s] Loading ground truth %s\n", elapsed() - t0, args.gtI.c_str());
    if (args.gtI.size() >= 4 && args.gtI.substr(args.gtI.size() - 4) == ".bin") {
        load_ground_truth(args.gtI, &gt_nq, &k, gt_ids, gt_dis);
        assert(gt_nq == nq);
    } else {
        if (args.gtD.empty())
            args.gtD = paired_distance_path(args.gtI);
        load_ground_truth(args.gtI, &gt_nq, &k, gt_ids, gt_dis);
        assert(gt_nq == nq);
    }

    printf("[%.3f s] Error-profile training (nq=%zu k=%zu)\n",
           elapsed() - t0, nq, k);
    faiss::Error_sys err_sys(index, nq, k);
    err_sys.set_gt(gt_dis.data(), gt_ids.data());

    size_t trace_train_nq = nq;
    if (args.calibrate) {
        size_t holdout_nq = static_cast<size_t>(
                std::llround(args.calibration_holdout_frac * nq));
        if (holdout_nq < 1)
            holdout_nq = 1;
        if (holdout_nq >= nq)
            holdout_nq = nq / 2 > 0 ? nq / 2 : 1;
        trace_train_nq = nq - holdout_nq;
        printf("[%.3f s] ELP trace training on first %zu queries; "
               "holdout %zu for (a,b) calibration\n",
               elapsed() - t0, trace_train_nq, holdout_nq);
    }
    err_sys.sys_train(trace_train_nq, xq);

    auncel_dist::BundleMeta meta;
    meta.train_nq = nq;
    meta.k = k;
    meta.d = d;
    meta.nb = nb;
    meta.index_key = args.index_key;

    if (args.calibrate) {
        std::vector<auncel_dist::HyperparamRow> rows;
        std::string hp_err;
        const std::string hp_path = args.hyperparam_file.empty()
                ? auncel_dist::default_hyperparameter_file()
                : args.hyperparam_file;
        if (!auncel_dist::load_hyperparameter_rows(hp_path, &rows, &hp_err)) {
            fprintf(stderr, "hyperparameter load failed: %s\n", hp_err.c_str());
            return 1;
        }
        auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
        auncel_dist::ShardCalibrateConfig ccfg;
        ccfg.enabled = true;
        ccfg.target_recall = args.calibration_recall;
        ccfg.holdout_frac = args.calibration_holdout_frac;
        ccfg.hyperparam_file = hp_path;
        const auto cres = auncel_dist::calibrate_shard_hyperparams(
                ivf, &err_sys, xq, nq, k, d, gt_ids.data(), ccfg, rows);
        meta.hyperparam_id = cres.best_id;
        meta.multipler = cres.multipler;
        meta.std_m = cres.std_m;
        meta.calibrated = cres.best_id > 0 ? 1 : 0;
    } else if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index)) {
        meta.multipler = ivf->t ? ivf->t->multipler : 1.0f;
        meta.std_m = ivf->t ? ivf->t->std_m : 1.0f;
    }

    // Trace training may use a query subset; tune-mode search needs GT for all nq.
    if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index)) {
        FAISS_THROW_IF_NOT_MSG(ivf->t, "IVF must have error profile after sys_train");
        auncel_dist::bind_profile_ground_truth(
                ivf->t, nq, k, gt_dis.data(), gt_ids.data());
    }

    printf("[%.3f s] Saving bundle prefix=%s\n", elapsed() - t0, args.out_prefix.c_str());
    auncel_dist::save_bundle(index, meta, args.out_prefix);

    delete[] xq;
    delete index;
    printf("[%.3f s] build_shard done.\n", elapsed() - t0);
    return 0;
}
