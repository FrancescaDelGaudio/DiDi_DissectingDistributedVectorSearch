#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "../Index.h"

inline float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int n = 0, d = 0;
    fread(&n, sizeof(int), 1, f);
    fread(&d, sizeof(int), 1, f);
    assert(d > 0 && d < 1000000);
    *d_out = d;
    *n_out = n;
    float* x = new float[size_t(n) * size_t(d)];
    size_t nr = fread(x, sizeof(float), size_t(n) * size_t(d), f);
    assert(nr == size_t(n) * size_t(d));
    fclose(f);
    return x;
}

inline float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d = 0;
    fread(&d, sizeof(int), 1, f);
    assert(d > 0 && d < 1000000);
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    size_t n = sz / (size_t((d + 1) * 4));
    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1));
    for (size_t i = 0; i < n; i++) {
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(float));
    }
    fclose(f);
    return x;
}

inline int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

inline float* load_vectors(const std::string& path, size_t* d_out, size_t* n_out) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".fbin")
        return fbin_read(path.c_str(), d_out, n_out);
    return fvecs_read(path.c_str(), d_out, n_out);
}

inline void gt_bin_read(
        const char* fname,
        size_t* nq_out,
        size_t* k_out,
        std::vector<faiss::Index::idx_t>& ids,
        std::vector<float>& dis) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int nq = 0, k = 0;
    fread(&nq, sizeof(int), 1, f);
    fread(&k, sizeof(int), 1, f);
    *nq_out = nq;
    *k_out = k;
    std::vector<int> raw_ids(nq * k);
    fread(raw_ids.data(), sizeof(int), nq * k, f);
    dis.resize(nq * k);
    fread(dis.data(), sizeof(float), nq * k, f);
    fclose(f);
    ids.resize(nq * k);
    for (size_t i = 0; i < ids.size(); i++) {
        ids[i] = raw_ids[i];
    }
}

inline std::string paired_distance_path(const std::string& idx_path) {
    std::string dis = idx_path;
    size_t pos = dis.rfind("gti.ivecs");
    if (pos != std::string::npos) {
        dis.replace(pos, 9, "gtd.fvecs");
        return dis;
    }
    pos = dis.rfind("idx.ivecs");
    if (pos != std::string::npos) {
        dis.replace(pos, 9, "dis.fvecs");
        return dis;
    }
    if (dis.size() > 6 && dis.substr(dis.size() - 6) == ".ivecs") {
        dis.replace(dis.size() - 6, 6, ".fvecs");
    }
    return dis;
}

inline void load_ground_truth(
        const std::string& gt_path,
        size_t* nq_out,
        size_t* k_out,
        std::vector<faiss::Index::idx_t>& ids,
        std::vector<float>& dis) {
    if (gt_path.size() >= 4 &&
        gt_path.substr(gt_path.size() - 4) == ".bin") {
        gt_bin_read(gt_path.c_str(), nq_out, k_out, ids, dis);
        return;
    }
    size_t nq1 = 0, nq2 = 0, k1 = 0, k2 = 0;
    int* raw_ids = ivecs_read(gt_path.c_str(), &k1, &nq1);
    std::string dis_path = paired_distance_path(gt_path);
    float* raw_dis = fvecs_read(dis_path.c_str(), &k2, &nq2);
    assert(nq1 == nq2 && k1 == k2);
    *nq_out = nq1;
    *k_out = k1;
    ids.resize(nq1 * k1);
    dis.resize(nq1 * k1);
    for (size_t i = 0; i < ids.size(); i++) {
        ids[i] = raw_ids[i];
        dis[i] = raw_dis[i];
    }
    delete[] raw_ids;
    delete[] raw_dis;
}
