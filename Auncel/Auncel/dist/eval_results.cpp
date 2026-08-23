// Compute mean recall@k on merged (or single-worker) search results vs global GT.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <sys/time.h>

#include "dist_io.h"

namespace {

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --gt <gt.bin|gtI.ivecs> --result <merged_idx.ivecs|result_dir>\n"
        "  %s --gt <gt> --result <dir> --num-workers N [--offsets o0 o1 ...]\n"
        "\n"
        "If --result is a directory, reads merged_idx.ivecs inside it when present,\n"
        "otherwise merges 0idx.ivecs .. (N-1)idx.ivecs with optional id offsets.\n"
        "\n"
        "Prints per-query recall@k and mean recall.\n",
        prog, prog);
}

float recall_at_k(const int* pred, const int* gt, size_t k) {
    std::vector<int> p(pred, pred + k);
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());

    size_t hits = 0;
    for (size_t i = 0; i < k; i++) {
        if (std::binary_search(p.begin(), p.end(), gt[i]))
            hits++;
    }
    return float(hits) / float(k);
}

std::vector<int> load_result_ids(const std::string& path) {
    size_t k = 0, nq = 0;
    int* raw = ivecs_read(path.c_str(), &k, &nq);
    std::vector<int> ids(raw, raw + nq * k);
    delete[] raw;
    return ids;
}

void merge_two_shards(
        const std::vector<int>& a,
        const std::vector<float>& ad,
        const std::vector<int>& b,
        const std::vector<float>& bd,
        size_t k,
        size_t nq,
        int offset_b,
        std::vector<int>* out_ids,
        std::vector<float>* out_dis) {
    out_ids->resize(nq * k);
    out_dis->resize(nq * k);
    for (size_t qi = 0; qi < nq; qi++) {
        size_t i1 = 0, i2 = 0, o = 0;
        while (o < k && (i1 < k || i2 < k)) {
            bool take1 = false;
            if (i1 >= k)
                take1 = false;
            else if (i2 >= k)
                take1 = true;
            else
                take1 = ad[qi * k + i1] < bd[qi * k + i2];
            if (take1) {
                (*out_dis)[qi * k + o] = ad[qi * k + i1];
                (*out_ids)[qi * k + o] = a[qi * k + i1];
                i1++;
            } else {
                (*out_dis)[qi * k + o] = bd[qi * k + i2];
                (*out_ids)[qi * k + o] = b[qi * k + i2] + offset_b;
                i2++;
            }
            o++;
        }
    }
}

std::vector<int> load_merged_from_dir(
        const std::string& dir,
        int num_workers,
        const std::vector<int>& offsets) {
    std::string merged = dir;
    if (!merged.empty() && merged.back() != '/')
        merged += '/';
    std::string merged_path = merged + "merged_idx.ivecs";
    FILE* probe = fopen(merged_path.c_str(), "rb");
    if (probe) {
        fclose(probe);
        return load_result_ids(merged_path);
    }

    auto shard_path = [&](int w, const char* suffix) {
        return merged + std::to_string(w) + suffix;
    };

    size_t k = 0, nq = 0;
    float* dis0 = fvecs_read(shard_path(0, "dis.fvecs").c_str(), &k, &nq);
    delete[] dis0;
    int* id0 = ivecs_read(shard_path(0, "idx.ivecs").c_str(), &k, &nq);
    std::vector<int> cur_ids(id0, id0 + nq * k);
    delete[] id0;
    float* cur_dis = fvecs_read(shard_path(0, "dis.fvecs").c_str(), &k, &nq);
    std::vector<float> cur_d(cur_dis, cur_dis + nq * k);
    delete[] cur_dis;

    for (int w = 1; w < num_workers; w++) {
        int* idi = ivecs_read(shard_path(w, "idx.ivecs").c_str(), &k, &nq);
        std::vector<int> nxt_ids(idi, idi + nq * k);
        delete[] idi;
        float* disi = fvecs_read(shard_path(w, "dis.fvecs").c_str(), &k, &nq);
        std::vector<float> nxt_d(disi, disi + nq * k);
        delete[] disi;
        std::vector<int> merged_ids;
        std::vector<float> merged_d;
        merge_two_shards(cur_ids, cur_d, nxt_ids, nxt_d, k, nq, offsets[w],
                         &merged_ids, &merged_d);
        cur_ids.swap(merged_ids);
        cur_d.swap(merged_d);
    }
    return cur_ids;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gt_path;
    std::string result_path;
    int num_workers = 0;
    std::vector<int> offsets;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                std::exit(1);
            }
            return std::string(argv[++i]);
        };
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--gt") {
            gt_path = need("--gt");
        } else if (arg == "--result") {
            result_path = need("--result");
        } else if (arg == "--num-workers") {
            num_workers = std::stoi(need("--num-workers"));
        } else if (arg == "--offsets") {
            while (i + 1 < argc && argv[i + 1][0] != '-')
                offsets.push_back(std::stoi(argv[++i]));
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (gt_path.empty() || result_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    double t0 = elapsed();

    std::vector<faiss::Index::idx_t> gt_ids;
    std::vector<float> gt_dis;
    size_t nq = 0, k = 0;
    load_ground_truth(gt_path, &nq, &k, gt_ids, gt_dis);

    std::vector<int> pred_ids;
    if (result_path.size() >= 6 &&
        result_path.substr(result_path.size() - 6) == ".ivecs") {
        pred_ids = load_result_ids(result_path);
    } else {
        if (num_workers <= 0)
            num_workers = 1;
        if (offsets.size() < size_t(num_workers))
            offsets.resize(num_workers, 0);
        pred_ids = load_merged_from_dir(result_path, num_workers, offsets);
    }

    if (pred_ids.size() != nq * k) {
        fprintf(stderr,
            "result size mismatch: got %zu ids, expected %zu (nq=%zu k=%zu)\n",
            pred_ids.size(), nq * k, nq, k);
        return 1;
    }

    double sum = 0.0;
    float min_r = 1.0f;
    for (size_t qi = 0; qi < nq; qi++) {
        std::vector<int> gt_row(k);
        for (size_t j = 0; j < k; j++)
            gt_row[j] = static_cast<int>(gt_ids[qi * k + j]);
        float r = recall_at_k(pred_ids.data() + qi * k, gt_row.data(), k);
        sum += r;
        min_r = std::min(min_r, r);
        printf("  query %zu recall@%zu = %.4f\n", qi, k, r);
    }
    const float mean_r = float(sum / double(nq));
    printf("Mean recall@%zu = %.4f  min = %.4f  (nq=%zu)\n", k, mean_r, min_r, nq);
    printf("Eval time: %.3f s\n", elapsed() - t0);
    return 0;
}
