#pragma once

#include <cstdint>
#include <limits>

namespace dann {

// Global vector id (up to ~4 billion vectors)
using idx_t   = uint32_t;
// Cluster id
using clust_t = uint32_t;

static constexpr idx_t   IDX_NONE   = std::numeric_limits<idx_t>::max();
static constexpr clust_t CLUST_NONE = std::numeric_limits<clust_t>::max();
static constexpr float   F_INF      = std::numeric_limits<float>::infinity();

} // namespace dann
