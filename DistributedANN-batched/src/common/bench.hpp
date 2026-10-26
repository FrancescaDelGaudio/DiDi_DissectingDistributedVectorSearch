#pragma once

// Ground-truth I/O and recall@k for query evaluation.
//
// Supported GT formats:
//   .lshbox  — text (Distributed-HNSW style)
//   .bin     — binary (gp-ann / graph-partitioning / Auncel style):
//               int32 nq, int32 k, int32 ids[nq*k], float32 dists[nq*k]

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace dann {

struct GroundTruth {
    int num_queries = 0;
    int top_k       = 0;
    std::vector<std::vector<uint32_t>> gt;

    void load_lshbox(const std::string& path) {
        FILE* fp = fopen(path.c_str(), "r");
        if (!fp) throw std::runtime_error("Cannot open ground truth: " + path);
        if (fscanf(fp, "%d %d", &num_queries, &top_k) != 2) {
            fclose(fp);
            throw std::runtime_error("Bad ground truth header: " + path);
        }
        gt.assign((size_t)num_queries, {});
        char line[65536];
        fgets(line, sizeof(line), fp);
        for (int q = 0; q < num_queries; q++) {
            if (!fgets(line, sizeof(line), fp)) break;
            std::istringstream ss(line);
            int qid = 0;
            ss >> qid;
            int id = 0;
            float dist = 0.f;
            while (ss >> id >> dist)
                gt[(size_t)q].push_back((uint32_t)id);
        }
        fclose(fp);
    }

    // gp-ann / Auncel binary GT (see Auncel/dist/dist_io.h gt_bin_read).
    void load_bin(const std::string& path) {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) throw std::runtime_error("Cannot open ground truth: " + path);
        int32_t nq = 0, k = 0;
        if (fread(&nq, 4, 1, fp) != 1 || fread(&k, 4, 1, fp) != 1) {
            fclose(fp);
            throw std::runtime_error("Bad GT bin header: " + path);
        }
        num_queries = (int)nq;
        top_k       = (int)k;
        std::vector<int32_t> raw((size_t)nq * (size_t)k);
        std::vector<float>   dist((size_t)nq * (size_t)k);
        if (fread(raw.data(), 4, raw.size(), fp) != raw.size() ||
            fread(dist.data(), 4, dist.size(), fp) != dist.size()) {
            fclose(fp);
            throw std::runtime_error("Truncated GT bin: " + path);
        }
        fclose(fp);
        gt.assign((size_t)nq, {});
        for (int q = 0; q < num_queries; q++)
            for (int j = 0; j < top_k; j++)
                gt[(size_t)q].push_back((uint32_t)raw[(size_t)q * (size_t)k + (size_t)j]);
    }

    void load(const std::string& path) {
        // Prefer binary loader for *.bin, and also when the file starts with
        // (nq,k) int32 header (e.g. GT.100M without a .bin suffix).
        bool use_bin = (path.size() >= 4 && path.substr(path.size() - 4) == ".bin");
        if (!use_bin) {
            FILE* fp = fopen(path.c_str(), "rb");
            if (fp) {
                int32_t nq = 0, k = 0;
                if (fread(&nq, 4, 1, fp) == 1 && fread(&k, 4, 1, fp) == 1 &&
                    nq > 0 && k > 0 && nq < 10000000 && k < 100000) {
                    use_bin = true;
                }
                fclose(fp);
            }
        }
        if (use_bin)
            load_bin(path);
        else
            load_lshbox(path);
    }
};

inline float recall_at_k(const std::vector<uint32_t>& result,
                         const std::vector<uint32_t>& truth, int k)
{
    if (k <= 0 || truth.empty()) return 0.f;
    int kk = std::min(k, (int)truth.size());
    std::unordered_set<uint32_t> truth_set(truth.begin(), truth.begin() + kk);
    int hits = 0;
    int rk   = std::min(k, (int)result.size());
    for (int i = 0; i < rk; i++)
        if (truth_set.count(result[i])) hits++;
    return (float)hits / (float)kk;
}

} // namespace dann
