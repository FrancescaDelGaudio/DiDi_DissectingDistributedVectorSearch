#pragma once

#ifdef DHNSW_USE_INNER_PRODUCT
#include "hnswlib/space_ip.h"
namespace dhnsw {
typedef hnswlib::InnerProductSpace HnswSpace;
}
#else
#include "hnswlib/space_l2.h"
namespace dhnsw {
typedef hnswlib::L2Space HnswSpace;
}
#endif
