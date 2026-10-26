//
// Pyramid Algorithm 5: single-node MIPS index construction.
//

#include <string>

#include "distributed/partition.hpp"
#include "executor.hpp"
#include "parameters.hpp"
#include "trainer/mips_trainer.hpp"

int main(int argc, char **argv) {
    mt::Partition partition(1);
    ss::parameter para;
    LoadOptions(argc, argv, para);

    dhnsw::single_machine_mips_trainer(
        para.mips_sample_n0,
        para.num_centroid,
        para.num_subhnsw,
        para.mips_replica_r,
        para.base_data,
        para.out_dir + "/centroids",
        para.out_dir + "/map",
        para.out_dir + "/partition_map",
        para.out_dir + "/hnsw_meta",
        para.out_dir + "/clusters",
        partition,
        32,
        100);

    return 0;
}
