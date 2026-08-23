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
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/time.h>

using vp = std::pair<int*, float*>;

namespace {

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <result_dir> <num_workers> [id_offset_per_worker ...]\n"
        "  Reads <i>idx.ivecs / <i>dis.fvecs for i in [0, num_workers).\n"
        "  Optional id offsets remap local shard ids to global ids.\n"
        "  Writes merged_idx.ivecs and merged_dis.fvecs into result_dir.\n"
        "\n"
        "Example (2 shards, second shard global id offset 50):\n"
        "  %s ./results 2 0 50\n",
        prog, prog);
}

float* fvecs_read(const char* fname, int* d_out, int* n_out) {
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
    int sz = st.st_size;
    int n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    fread(x, sizeof(float), n * (d + 1), f);
    for (int i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));
    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, int* d_out, int* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

void fvecs_write(const char* fname, float* x, int d, int n) {
    FILE* f = fopen(fname, "wb");
    for (int i = 0; i < n; i++) {
        fwrite(&d, sizeof(int), 1, f);
        fwrite(x + i * d, sizeof(float), d, f);
    }
    fclose(f);
}

void ivecs_write(const char* fname, int* x, int d, int n) {
    FILE* f = fopen(fname, "wb");
    for (int i = 0; i < n; i++) {
        fwrite(&d, sizeof(int), 1, f);
        fwrite(x + i * d, sizeof(int), d, f);
    }
    fclose(f);
}

vp load_shard(const std::string& prefix) {
    std::string disfile = prefix + "dis.fvecs";
    std::string idxfile = prefix + "idx.ivecs";
    int d, n, d1, n1;
    float* dis = fvecs_read(disfile.c_str(), &d, &n);
    int* idx = ivecs_read(idxfile.c_str(), &d1, &n1);
    assert(d == d1 && n == n1);
    return vp(idx, dis);
}

void apply_offset(vp& shard, int offset) {
    if (offset == 0) return;
    int k = 0;
    // infer k from first row header not stored in flat array; use heuristic via file
    // caller passes k separately through merge functions
    (void)shard;
    (void)offset;
}

void merge_two(const vp& a1,
               const vp& a2,
               vp* out,
               int k,
               int nq,
               int id_offset2) {
    out->first = new int[nq * k];
    out->second = new float[nq * k];
    for (int qi = 0; qi < nq; qi++) {
        int i1 = 0, i2 = 0, out_i = 0;
        while (out_i < k && (i1 < k || i2 < k)) {
            bool take1 = false;
            if (i1 >= k)
                take1 = false;
            else if (i2 >= k)
                take1 = true;
            else
                take1 = a1.second[qi * k + i1] < a2.second[qi * k + i2];

            if (take1) {
                out->second[qi * k + out_i] = a1.second[qi * k + i1];
                out->first[qi * k + out_i] = a1.first[qi * k + i1];
                i1++;
            } else {
                out->second[qi * k + out_i] = a2.second[qi * k + i2];
                out->first[qi * k + out_i] = a2.first[qi * k + i2] + id_offset2;
                i2++;
            }
            out_i++;
        }
    }
}

void free_vp(vp& v) {
    delete[] v.first;
    delete[] v.second;
}

std::string path_join(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    std::string result_dir = argv[1];
    int num_workers = std::stoi(argv[2]);
    if (num_workers < 1) {
        fprintf(stderr, "num_workers must be >= 1\n");
        return 1;
    }

    std::vector<int> offsets(num_workers, 0);
    for (int i = 0; i < num_workers; i++) {
        if (argc >= 4 + i)
            offsets[i] = std::stoi(argv[3 + i]);
    }

    double t0 = elapsed();
    vp cur = load_shard(path_join(result_dir, "0"));
    int k = 0, nq = 0;
  {
    std::string probe = path_join(result_dir, "0dis.fvecs");
    int d1, n1;
    float* tmp = fvecs_read(probe.c_str(), &k, &nq);
    delete[] tmp;
  }

    for (int w = 1; w < num_workers; w++) {
        vp nxt = load_shard(path_join(result_dir, std::to_string(w)));
        vp merged;
        merge_two(cur, nxt, &merged, k, nq, offsets[w]);
        free_vp(cur);
        free_vp(nxt);
        cur = merged;
    }

    std::string out_idx = path_join(result_dir, "merged_idx.ivecs");
    std::string out_dis = path_join(result_dir, "merged_dis.fvecs");
    ivecs_write(out_idx.c_str(), cur.first, k, nq);
    fvecs_write(out_dis.c_str(), cur.second, k, nq);
  free_vp(cur);

    printf("Reduce time: %.3f s\n", elapsed() - t0);
    printf("Wrote %s and %s (nq=%d, k=%d)\n",
           out_idx.c_str(), out_dis.c_str(), nq, k);
    return 0;
}
