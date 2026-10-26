#pragma once

// Candidate / result lists for Algorithm 2 (paper §2.4).
// We keep (id, squared-L2 distance) pairs sorted ascending; lower is better.

#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace dann {

struct IdDist {
    uint32_t id   = 0;
    float    dist = 0.f;
    bool operator<(const IdDist& o) const { return dist < o.dist; }
};

inline void dedupe_best(std::vector<IdDist>& v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end(), [](const IdDist& a, const IdDist& b) {
        return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
    });
    std::vector<IdDist> out;
    out.reserve(v.size());
    for (auto& e : v) {
        if (out.empty() || out.back().id != e.id) out.push_back(e);
    }
    v.swap(out);
}

inline void merge_cap(std::vector<IdDist>& acc, const std::vector<IdDist>& add, int cap) {
    if (add.empty()) return;
    acc.insert(acc.end(), add.begin(), add.end());
    dedupe_best(acc);
    if ((int)acc.size() > cap) acc.resize((size_t)cap);
}

// Worst (largest) distance among the current candidate set (peekworst in Alg. 2).
inline float candidate_threshold(const std::vector<IdDist>& candidates, int L) {
    if (candidates.empty()) return std::numeric_limits<float>::infinity();
    int idx = std::min(L, (int)candidates.size()) - 1;
    return candidates[(size_t)idx].dist;
}

// Take up to `bw` best unvisited ids from sorted candidates.
inline int pick_beam(const std::vector<IdDist>& candidates, int bw,
                     std::unordered_map<uint32_t, char>& visited,
                     uint32_t* keys_out)
{
    int n = 0;
    for (auto& e : candidates) {
        if (n >= bw) break;
        if (visited.count(e.id)) continue;
        visited[e.id] = 1;
        keys_out[n++] = e.id;
    }
    return n;
}

} // namespace dann
