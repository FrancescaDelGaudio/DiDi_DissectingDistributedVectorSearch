#pragma once

// Read index metadata written by build_index (Step 1).

#include <cstdio>
#include <string>
#include <stdexcept>

#include "metric.hpp"

namespace dann {

inline Metric read_index_metric(const std::string& index_dir) {
    std::string path = index_dir + "/meta.json";
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp) throw std::runtime_error("Cannot open index meta: " + path);

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (std::strstr(line, "\"metric\"") != nullptr) {
            if (std::strstr(line, "\"ip\"") != nullptr) {
                fclose(fp);
                return Metric::IP;
            }
            if (std::strstr(line, "\"l2\"") != nullptr) {
                fclose(fp);
                return Metric::L2;
            }
        }
    }
    fclose(fp);
    return Metric::L2;   // legacy indexes default to L2
}

} // namespace dann
