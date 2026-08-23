// Copyright (c) Zili Zhang.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/time.h>

#include <omp.h>

#include "../AutoTune.h"
#include "../IndexIVF.h"
#include "../profile.h"
#include "dist_defaults.h"
#include "dist_io.h"
#include "shard_bundle.h"

#define MAXLINE 4096
#define DC(classname) classname* ix = dynamic_cast<classname*>(index)

namespace {

struct WorkerConfig {
    int port = 3456;
    int omp_threads = 4;
    int hyperparam_id = 1;
    float required_recall = 0.9f;
    std::string load_prefix;
    std::string db;
    std::string query;
    std::string gtI;
    std::string gtD;
    std::string index_key = "IVF8,Flat";
    bool prebuilt = false;
};

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  Online train+search (legacy):\n"
        "    %s [port] [base.fbin] [query.fbin] [gt.bin|gtI.ivecs] [gtD.fvecs] "
        "[index_key] [required_recall] [hyperparam_id] [omp_threads]\n"
        "\n"
        "  Prebuilt index (search-only, load from disk):\n"
        "    %s --load <prefix> --port N --query <query.fbin> "
        "[--gt <gt.bin|gti.ivecs>] [--gtd <gtd.fvecs>] "
        "[--required-recall 0.9] [--hyperparam-id 0] [--omp-threads N]\n"
        "\n"
        "  --hyperparam-id 0 (default with --load): use (a,b) saved in the shard bundle.\n"
        "  --hyperparam-id N>=1: override with row N from hyperparameter.txt.\n"
        "\n"
        "  <prefix> must have sibling files .faissindex .profile .meta "
        "(from build_shard).\n"
        "  Master command 'train' is a no-op when --load is used.\n",
        prog, prog);
}

WorkerConfig parse_config(int argc, char** argv) {
    WorkerConfig cfg;
    cfg.db = auncel_dist::default_base();
    cfg.query = auncel_dist::default_query();
    cfg.gtI = auncel_dist::default_gt();
    cfg.port = auncel_dist::default_port();
    cfg.required_recall = auncel_dist::default_require_recall();
    cfg.hyperparam_id = auncel_dist::default_hyperparam_id();

    bool positional = true;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--load") {
            positional = false;
            cfg.prebuilt = true;
            cfg.load_prefix = argv[++i];
            continue;
        }
        if (!positional) {
            auto need = [&](const char* flag) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "missing value for %s\n", flag);
                    usage(argv[0]);
                    std::exit(1);
                }
                return std::string(argv[++i]);
            };
            if (arg == "--port")
                cfg.port = std::stoi(need("--port"));
            else if (arg == "--query")
                cfg.query = need("--query");
            else if (arg == "--gt")
                cfg.gtI = need("--gt");
            else if (arg == "--gtd")
                cfg.gtD = need("--gtd");
            else if (arg == "--required-recall")
                cfg.required_recall = std::stof(need("--required-recall"));
            else if (arg == "--hyperparam-id")
                cfg.hyperparam_id = std::stoi(need("--hyperparam-id"));
            else if (arg == "--omp-threads")
                cfg.omp_threads = std::stoi(need("--omp-threads"));
            else {
                fprintf(stderr, "unknown flag: %s\n", arg.c_str());
                usage(argv[0]);
                std::exit(1);
            }
            continue;
        }
    }

    if (positional) {
        if (argc >= 2 && argv[1][0] != '-')
            cfg.port = std::stoi(argv[1]);
        if (argc >= 3) cfg.db = argv[2];
        if (argc >= 4) cfg.query = argv[3];
        if (argc >= 5) cfg.gtI = argv[4];
        if (argc >= 6) cfg.gtD = argv[5];
        if (argc >= 7) cfg.index_key = argv[6];
        if (argc >= 8) cfg.required_recall = std::stof(argv[7]);
        if (argc >= 9) cfg.hyperparam_id = std::stoi(argv[8]);
        if (argc >= 10) cfg.omp_threads = std::stoi(argv[9]);
    } else if (const char* p = std::getenv("AUNCEL_PORT"))
        cfg.port = std::atoi(p);

    if (const char* t = std::getenv("OMP_NUM_THREADS"))
        cfg.omp_threads = std::atoi(t);
    if (cfg.prebuilt && !std::getenv("AUNCEL_HP_ID"))
        cfg.hyperparam_id = 0;
    return cfg;
}

void load_ground_truth_worker(
        const WorkerConfig& cfg,
        size_t nq,
        size_t* k_out,
        faiss::Index::idx_t** gt_out,
        float** gt_v_out) {
    std::vector<faiss::Index::idx_t> ids;
    std::vector<float> dis;
    size_t gt_nq = 0, k = 0;
    load_ground_truth(cfg.gtI, &gt_nq, &k, ids, dis);
    assert(gt_nq == nq);
    *k_out = k;
    *gt_out = new faiss::Index::idx_t[ids.size()];
    *gt_v_out = new float[dis.size()];
    memcpy(*gt_out, ids.data(), ids.size() * sizeof(faiss::Index::idx_t));
    memcpy(*gt_v_out, dis.data(), dis.size() * sizeof(float));
}

void apply_search_hyperparams(
        faiss::IndexIVF* ivf,
        int hyperparam_id,
        const auncel_dist::BundleMeta* meta) {
    if (!ivf || !ivf->t)
        return;
    if (hyperparam_id > 0) {
        ivf->t->setparam(hyperparam_id);
        printf("Search hyperparams: global override id=%d -> multipler=%.3f std_m=%.3f\n",
               hyperparam_id, ivf->t->multipler, ivf->t->std_m);
        return;
    }
    if (meta && meta->calibrated) {
        ivf->t->multipler = meta->multipler;
        ivf->t->std_m = meta->std_m;
        ivf->t->profile = false;
        printf("Search hyperparams: shard bundle row=%d -> multipler=%.3f std_m=%.3f\n",
               meta->hyperparam_id, ivf->t->multipler, ivf->t->std_m);
        return;
    }
    ivf->t->profile = false;
    printf("Search hyperparams: profile defaults multipler=%.3f std_m=%.3f\n",
           ivf->t->multipler, ivf->t->std_m);
}

}  // namespace

int main(int argc, char** argv) {
    WorkerConfig cfg = parse_config(argc, argv);
    omp_set_num_threads(cfg.omp_threads);

    printf("Worker config:\n");
    printf("  mode=%s  port=%d  required_recall=%.3f  hyperparam_id=%d\n",
           cfg.prebuilt ? "prebuilt(search)" : "online(train+search)",
           cfg.port, cfg.required_recall, cfg.hyperparam_id);
    if (cfg.prebuilt)
        printf("  load_prefix=%s\n", cfg.load_prefix.c_str());
    else
        printf("  db=%s  index_key=%s\n", cfg.db.c_str(), cfg.index_key.c_str());
    printf("  query=%s  gt=%s\n", cfg.query.c_str(), cfg.gtI.c_str());

    double t0 = elapsed();
    faiss::Index* index = nullptr;
    auncel_dist::BundleMeta bundle_meta;
    size_t d = 0;
    size_t k = 0;
    faiss::Index::idx_t* gt = nullptr;
    size_t nq = 0;
    float* xq = nullptr;
    float* gt_v = nullptr;
    size_t topk = 0;
    faiss::Error_sys err_sys;
    bool is_trained = false;

    if (cfg.prebuilt) {
        if (!auncel_dist::bundle_exists(cfg.load_prefix)) {
            fprintf(stderr, "bundle not found for prefix %s\n",
                    cfg.load_prefix.c_str());
            return 1;
        }
        index = auncel_dist::load_bundle(cfg.load_prefix, &bundle_meta);
        d = bundle_meta.d;
        k = bundle_meta.k;
        topk = k;
        auncel_dist::attach_error_sys(&err_sys, index, bundle_meta);
        is_trained = true;

        printf("[%.3f s] Loading queries %s\n", elapsed() - t0, cfg.query.c_str());
        size_t dq = 0;
        xq = load_vectors(cfg.query, &dq, &nq);
        assert(d == dq);
        if (nq > bundle_meta.train_nq) {
            fprintf(stderr,
                "query count %zu exceeds profile train_nq %zu\n",
                nq, bundle_meta.train_nq);
            return 1;
        }

        printf("[%.3f s] Loading ground truth for recall\n", elapsed() - t0);
        load_ground_truth_worker(cfg, nq, &k, &gt, &gt_v);
        topk = k;
        if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index)) {
            if (ivf->t)
                auncel_dist::bind_profile_ground_truth(ivf->t, nq, k, gt_v, gt);
        }
        printf("[%.3f s] Prebuilt bundle ready (nb=%zu d=%zu train_nq=%zu k=%zu"
               "%s)\n",
               elapsed() - t0, bundle_meta.nb, d, bundle_meta.train_nq, k,
               bundle_meta.calibrated
                       ? ", per-shard (a,b) calibrated"
                       : "");
    } else {
        size_t nt = 0;
        printf("[%.3f s] Loading train set\n", elapsed() - t0);
        float* xt = load_vectors(cfg.db, &d, &nt);

        printf("[%.3f s] Preparing index \"%s\" d=%zu (nt=%zu)\n",
               elapsed() - t0, cfg.index_key.c_str(), d, nt);
        if (cfg.index_key == "IVF1024,Flat" && nt < 1024) {
            fprintf(stderr,
                "IVF1024 needs >=1024 training vectors (got %zu). "
                "Use IVF8,Flat for the toy dataset.\n",
                nt);
            return 1;
        }
        index = faiss::index_factory(int(d), cfg.index_key.c_str());
        if (dynamic_cast<faiss::IndexIVF*>(index) == nullptr) {
            fprintf(stderr, "index_key must produce an IndexIVF (got %s)\n",
                    cfg.index_key.c_str());
            return 1;
        }
        delete[] xt;
    }

    int listenfd = -1;
    struct sockaddr_in servaddr;

    if ((listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        printf("create socket error: %s(errno: %d)\n", strerror(errno), errno);
        return 1;
    }

    int opt = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(cfg.port);

    if (bind(listenfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1) {
        printf("bind socket error: %s(errno: %d)\n", strerror(errno), errno);
        return 1;
    }

    if (listen(listenfd, 10) == -1) {
        printf("listen socket error: %s(errno: %d)\n", strerror(errno), errno);
        return 1;
    }

    printf("======waiting for master's request (port %d)======\n", cfg.port);
    while (true) {
        int connfd = accept(listenfd, (struct sockaddr*)NULL, NULL);
        if (connfd == -1) {
            printf("accept socket error: %s(errno: %d)\n", strerror(errno), errno);
            continue;
        }

        char buff[MAXLINE];
        int n = recv(connfd, buff, MAXLINE, 0);
        if (n <= 0) {
            close(connfd);
            continue;
        }
        buff[n] = '\0';
        std::string s = buff;
        printf("recv msg from master: %s\n", buff);

        if (s == "train" && cfg.prebuilt) {
            printf("Prebuilt bundle loaded; train is a no-op\n");
            std::string su = "success";
            send(connfd, su.c_str(), su.size(), 0);
            close(connfd);
        } else if (s == "train" && !is_trained) {
            size_t nt = 0;
            float* xt = load_vectors(cfg.db, &d, &nt);
            printf("[%.3f s] Training on %zu vectors\n", elapsed() - t0, nt);
            index->set_tune_mode();
            index->train(nt, xt);
            index->set_tune_off();
            delete[] xt;

            printf("[%.3f s] Loading database\n", elapsed() - t0);
            size_t nb = 0, d2 = 0;
            float* xb = load_vectors(cfg.db, &d2, &nb);
            assert(d == d2);

            printf("[%.3f s] Indexing database, size %zu*%zu\n",
                   elapsed() - t0, nb, d);
            index->add(nb, xb);
            delete[] xb;

            printf("[%.3f s] Loading queries\n", elapsed() - t0);
            size_t d2q = 0;
            xq = load_vectors(cfg.query, &d2q, &nq);
            assert(d == d2q);
            assert(nq % 10 == 0 || !"Error_sys requires nq divisible by 10");

            load_ground_truth_worker(cfg, nq, &k, &gt, &gt_v);
            topk = k;

            printf("[%.3f s] Preparing error profile system, k=%zu nq=%zu\n",
                   elapsed() - t0, k, nq);
            faiss::Error_sys tmp_err_sys(index, nq, k);
            err_sys = tmp_err_sys;
            err_sys.set_gt(gt_v, gt);

            printf("[%.3f s] Start error profile system training\n", elapsed() - t0);
            err_sys.sys_train(nq, xq);
            printf("[%.3f s] Finish error profile system training\n", elapsed() - t0);

            is_trained = true;
            std::string su = "success";
            send(connfd, su.c_str(), su.size(), 0);
            close(connfd);
        } else if (s == "train") {
            printf("Index has already trained\n");
            std::string su = "success";
            send(connfd, su.c_str(), su.size(), 0);
            close(connfd);
        } else if (s.compare(0, 6, "search") == 0) {
            if (!is_trained) {
                std::string su = "index has not trained";
                send(connfd, su.c_str(), su.size(), 0);
                close(connfd);
                continue;
            }

            // Master may send "search <local_recall>" after §4.5 calibration.
            float search_recall = cfg.required_recall;
            if (s.size() > 6) {
                const char* p = s.c_str() + 6;
                while (*p == ' ')
                    ++p;
                if (*p) {
                    char* end = nullptr;
                    const double v = strtod(p, &end);
                    if (end != p)
                        search_recall = static_cast<float>(v);
                }
            }

            if (DC(faiss::IndexIVF))
                apply_search_hyperparams(ix, cfg.hyperparam_id,
                        cfg.prebuilt ? &bundle_meta : nullptr);

            const size_t demo_size = nq;
            const size_t ts = 0;

            std::vector<float> D(demo_size * k);
            std::vector<int64_t> I(demo_size * k);
            std::vector<float> acc(ts + demo_size, search_recall);

            err_sys.set_topk(topk);
            err_sys.set_queries(demo_size, xq, acc.data(), ts + demo_size);

            printf("[%.3f s] Start error profile system search "
                   "(nq=%zu, k=%zu, required_recall=%.6f"
                   "%s)\n",
                   elapsed() - t0, demo_size, topk, search_recall,
                   search_recall != cfg.required_recall ? ", calibrated by leader" : "");
            t0 = elapsed();
            err_sys.search(D.data(), I.data(), ts);
            printf("Finish error profile system search: %.3f s\n", elapsed() - t0);

            if (gt) {
                float min_recall = 1.0f;
                for (size_t qi = 0; qi < demo_size; qi++) {
                    float r = err_sys.recall(
                        reinterpret_cast<faiss::Index::idx_t*>(I.data() + qi * k),
                        gt + qi * k,
                        k);
                    min_recall = std::min(min_recall, r);
                    printf("  query %zu recall@%zu = %.3f\n", qi, topk, r);
                }
                printf("Min recall@%zu = %.3f (required %.6f)\n",
                       topk, min_recall, search_recall);
            }

            std::vector<int> int_I(demo_size * topk);
            for (size_t j = 0; j < demo_size * topk; j++)
                int_I[j] = static_cast<int>(I[j]);

            FILE* f = fopen("./idx.ivecs", "wb");
            for (size_t i = 0; i < demo_size; i++) {
                int kk = static_cast<int>(topk);
                fwrite(&kk, sizeof(int), 1, f);
                fwrite(int_I.data() + topk * i, sizeof(int), topk, f);
            }
            fclose(f);

            FILE* tf = fopen("./dis.fvecs", "wb");
            for (size_t i = 0; i < demo_size; i++) {
                int kk = static_cast<int>(topk);
                fwrite(&kk, sizeof(int), 1, tf);
                fwrite(D.data() + topk * i, sizeof(float), topk, f);
            }
            fclose(tf);

            char buffer[MAXLINE];
            std::string file_end = "file done";
            FILE* fp = fopen("./idx.ivecs", "rb");
            while (true) {
                size_t len = fread(buffer, 1, sizeof(buffer), fp);
                if (len == 0) break;
                if (len == file_end.size() &&
                    memcmp(buffer, file_end.c_str(), len) == 0)
                    break;
                if (send(connfd, buffer, len, 0) != (ssize_t)len) {
                    printf("slave send idx file failed\n");
                    break;
                }
            }
            fclose(fp);
            send(connfd, file_end.c_str(), file_end.size(), 0);

            FILE* fp2 = fopen("./dis.fvecs", "rb");
            while (true) {
                size_t len = fread(buffer, 1, sizeof(buffer), fp2);
                if (len == 0) break;
                if (len == file_end.size() &&
                    memcmp(buffer, file_end.c_str(), len) == 0)
                    break;
                if (send(connfd, buffer, len, 0) != (ssize_t)len) {
                    printf("slave send dis file failed\n");
                    break;
                }
            }
            fclose(fp2);
            send(connfd, file_end.c_str(), file_end.size(), 0);

            std::string su = "success";
            send(connfd, su.c_str(), su.size(), 0);
            close(connfd);

            if (std::getenv("AUNCEL_EXIT_AFTER_SEARCH")) {
                printf("AUNCEL_EXIT_AFTER_SEARCH set — exiting worker\n");
                close(listenfd);
                delete[] xq;
                delete[] gt;
                delete[] gt_v;
                delete index;
                return 0;
            }
        } else {
            std::string su = "unknown command";
            send(connfd, su.c_str(), su.size(), 0);
            close(connfd);
        }
    }

    close(listenfd);
    delete[] xq;
    delete[] gt;
    delete[] gt_v;
    delete index;
    return 0;
}
