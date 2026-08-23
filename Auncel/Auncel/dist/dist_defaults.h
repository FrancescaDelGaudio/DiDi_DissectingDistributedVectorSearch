// Default paths and parameters for smoke tests / toy dataset.
#pragma once

#include <cstdlib>
#include <string>

#include <sys/stat.h>

namespace auncel_dist {

static const char* kToyDir =
    "/gpfs/scratch/delgaudio/vectors/toy/generated_deep_files";

inline std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? std::string(v) : fallback;
}

inline std::string default_base() {
    return env_or("AUNCEL_BASE",
                  std::string(kToyDir) + "/random_base_100.fbin");
}

inline std::string default_query() {
    return env_or("AUNCEL_QUERY",
                  std::string(kToyDir) + "/random_queries_10_k5.fbin");
}

inline std::string default_gt() {
    return env_or("AUNCEL_GT",
                  std::string(kToyDir) + "/random_GT_10_k5.bin");
}

inline int default_port() {
    const char* v = std::getenv("AUNCEL_PORT");
    return v ? std::atoi(v) : 3456;
}

inline float default_require_recall() {
    const char* v = std::getenv("AUNCEL_RECALL");
    return v ? std::atof(v) : 0.9f;  // error_bound = 1 - recall
}

inline int default_id_offset() {
    const char* v = std::getenv("AUNCEL_ID_OFFSET");
    return v ? std::atoi(v) : 0;
}

inline int default_hyperparam_id() {
    const char* v = std::getenv("AUNCEL_HP_ID");
    return v ? std::atoi(v) : 1;
}

// IVF nlist for small shards (<= 10k vectors).
inline const char* index_key_for_size(size_t nb) {
    if (nb <= 128) return "IVF8,Flat";
    if (nb <= 1024) return "IVF32,Flat";
    if (nb <= 100000) return "IVF256,Flat";
    return "IVF1024,Flat";
}

}  // namespace auncel_dist
