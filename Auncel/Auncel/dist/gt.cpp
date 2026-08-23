// Copyright (c) Zili Zhang.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/time.h>

#include <omp.h>

#include "../IndexFlat.h"

namespace {

const char* kDefaultQuery =
    "/gpfs/scratch/delgaudio/vectors/toy/generated_deep_files/random_queries_10_k5.fbin";

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <shard.fbin> [query.fbin] [topk]\n"
        "  Defaults: query=%s  topk=5\n"
        "  Writes <prefix>gti.ivecs and <prefix>gtd.fvecs next to shard "
        "(e.g. 0db.fbin -> 0gti.ivecs).\n",
        prog, kDefaultQuery);
}

float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d, n;
    fread(&n, sizeof(int), 1, f);
    fread(&d, sizeof(int), 1, f);
    size_t total_size = size_t(d) * size_t(n);
    float* x = new float[total_size];
    size_t nr = fread(x, sizeof(float), total_size, f);
    assert(nr == total_size || !"could not read whole file");
    fclose(f);
    *d_out = d;
    *n_out = n;
    printf("Read fbin %s: n=%d d=%d\n", fname, n, d);
    return x;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, sizeof(int), 1, f);
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    size_t n = sz / ((d + 1) * 4);
    float* x = new float[n * (d + 1)];
    fread(x, sizeof(float), n * (d + 1), f);
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));
    fclose(f);
    *d_out = d;
    *n_out = n;
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    std::string dbfile = argv[1];
    std::string query_path = (argc >= 3) ? argv[2] : kDefaultQuery;
    size_t topk = (argc >= 4) ? size_t(std::stoi(argv[3])) : 5;

    size_t suffix = dbfile.rfind("db.fbin");
    if (suffix == std::string::npos) {
        fprintf(stderr, "Expected shard name ending with db.fbin, got %s\n",
                dbfile.c_str());
        return 1;
    }
    std::string prefix = dbfile.substr(0, suffix);
    std::string gtidx = prefix + "gti.ivecs";
    std::string gtdis = prefix + "gtd.fvecs";

    double t0 = elapsed();
    size_t d, nb, d2, nq;
    float* xb = fbin_read(dbfile.c_str(), &d, &nb);
    float* xq = nullptr;
    if (query_path.size() >= 5 &&
        query_path.substr(query_path.size() - 5) == ".fbin") {
        xq = fbin_read(query_path.c_str(), &d2, &nq);
    } else {
        xq = fvecs_read(query_path.c_str(), &d2, &nq);
    }
    assert(d == d2);

    faiss::IndexFlatL2 index(d);
    index.add(nb, xb);
    delete[] xb;

    std::vector<faiss::Index::idx_t> I(nq * topk);
    std::vector<float> D(nq * topk);

    omp_set_num_threads(4);
    printf("[%.3f s] Brute-force GT on shard (%zu vectors, %zu queries, k=%zu)\n",
           elapsed() - t0, nb, nq, topk);
    index.search(nq, xq, topk, D.data(), I.data());
    printf("[%.3f s] Search finished\n", elapsed() - t0);

    std::vector<int> int_I(nq * topk);
    for (size_t j = 0; j < nq * topk; j++)
        int_I[j] = static_cast<int>(I[j]);

    int k_int = static_cast<int>(topk);
    FILE* f = fopen(gtidx.c_str(), "wb");
    for (size_t i = 0; i < nq; i++) {
        fwrite(&k_int, sizeof(int), 1, f);
        fwrite(int_I.data() + topk * i, sizeof(int), topk, f);
    }
    fclose(f);

    FILE* tf = fopen(gtdis.c_str(), "wb");
    for (size_t i = 0; i < nq; i++) {
        fwrite(&k_int, sizeof(int), 1, tf);
        fwrite(D.data() + topk * i, sizeof(float), topk, f);
    }
    fclose(tf);

    printf("Wrote %s and %s\n", gtidx.c_str(), gtdis.c_str());
    delete[] xq;
    return 0;
}
