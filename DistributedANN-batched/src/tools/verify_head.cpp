// Quick verification: load a persisted head index and run a few searches.
#include <cstdio>
#include <vector>
#include <string>
#include "common/io.hpp"
#include "hnswlib/hnswlib.h"

using namespace dann;

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <head.hnsw> <base.fvecs> <dim>\n", argv[0]); return 1; }
    std::string hpath = argv[1], base = argv[2];
    int dim = std::stoi(argv[3]);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, hpath);   // load
    index.setEf(200);
    printf("Loaded head index: cur_elements=%zu  dim=%d\n",
           (size_t)index.cur_element_count.load(), dim);

    int qd = 0;
    std::vector<float> q = read_fvecs_slice(base, 0, 5, qd);   // 5 base vectors as queries
    for (int i = 0; i < 5; i++) {
        auto res = index.searchKnn(q.data() + (size_t)i * dim, 10);
        // results come out worst-first; collect and reverse
        std::vector<std::pair<float, hnswlib::labeltype>> v;
        while (!res.empty()) { v.push_back(res.top()); res.pop(); }
        printf("query %d -> top: ", i);
        bool ok = true;
        for (auto it = v.rbegin(); it != v.rend(); ++it) {
            printf("(id=%lu d=%.3f) ", (unsigned long)it->second, it->first);
            if (it->first < 0) ok = false;
        }
        printf(" [%s]\n", ok ? "OK" : "BAD");
    }
    return 0;
}
