
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <omp.h>

#include <boost/program_options.hpp>

#include "coromem/include/share_mem.h"
#include "coromem/include/utils.h"
#include "index/scala_build.h"
#include "index/top_index.h"
#include "index/vamana/vamana.h"

namespace {

bool env_truthy(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool final_scala_artifacts_ready(const IndexParameter &p) {
  for (uint32_t m = 0; m < MACHINE_NUM; ++m) {
    if (!std::filesystem::exists(p.final_part_scala_index_path[m])) {
      return false;
    }
    if (!std::filesystem::exists(p.final_part_scala_data_path[m])) {
      return false;
    }
    if (!std::filesystem::exists(p.final_part_b1_index_path[m])) {
      return false;
    }
  }
  return std::filesystem::exists(p.temp_id2part_map_file) &&
         std::filesystem::exists(p.temp_pos2label_map_file) &&
         std::filesystem::exists(p.temp_label2pos_map_file) &&
         std::filesystem::exists(p.medoids_path);
}

void load_global_medoid(IndexParameter &p) {
  std::ifstream mf(p.medoids_path, std::ios::binary);
  if (!mf) {
    throw std::runtime_error("missing medoids file: " + p.medoids_path);
  }
  uint32_t nshards = 0;
  uint32_t one = 0;
  mf.read(reinterpret_cast<char *>(&nshards), sizeof(uint32_t));
  mf.read(reinterpret_cast<char *>(&one), sizeof(uint32_t));
  mf.read(reinterpret_cast<char *>(&p.medoid), sizeof(uint32_t));
  std::cout << "Loaded global medoid " << p.medoid << " from "
            << p.medoids_path << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  SharedMem coromem;
  commandLine cmd(argc, argv, "Usage : \n");
  RdmaParameter rdma_param(cmd);
  auto &tp = getThreadPool();
  tp.poolPause();

  IndexParameter index_param(argc, argv);

  ScalaBuild scala_build(index_param, rdma_param);

  const bool resume_merge_top =
      env_truthy("COTRA_RESUME_MERGE_TOP") &&
      final_scala_artifacts_ready(index_param) &&
      !std::filesystem::exists(index_param.top_index_path);

  if (resume_merge_top) {
    std::cout << "==== COTRA_RESUME_MERGE_TOP: skipping build; merge b1 + top "
                 "index only ===="
              << std::endl;
  }

  // Start rdma server.
  scala_build.init_rdma();

  if (!resume_merge_top) {
    // Start to do partition and dispatch.
    scala_build.partition();

    // Build local partition index.
    scala_build.swap_partition_info();

    scala_build.build();
  }

  tp.poolContiue();

  if (index_param.graph_type == GraphType::SCALA_V3) {
    if (!resume_merge_top) {
      scala_build.write_partition_info();

      // Load partiiton.
      scala_build.partition_checkpoint();

      // Merge graph from other machine.
      scala_build.merge();

      // Now reformat b1 graph to scalagraph v3.
      scala_build.transfer_to_scala_v3();
    } else {
      scala_build.partition_checkpoint();
      if (index_param.machine_id == 0) {
        load_global_medoid(index_param);
      }
    }

    // Leader-only: merge partition b1 graphs and build HNSW top index.
    if (index_param.machine_id == 0 &&
        !std::filesystem::exists(index_param.top_index_path)) {
      scala_build.merge_all_b1_index();
      scala_build.topindex_checkpoint();
      HNSWTopIndex top_index;
      top_index.build_hnsw_top_index(scala_build.index_param, cmd);
    }

    // All ranks must stay alive (RDMA threads running) until leader finishes.
    scala_build.rdma_comm.build_sync();
  } else if (index_param.graph_type == GraphType::SharedNothing) {
    if (!resume_merge_top) {
      scala_build.write_partition_info();
      // transfer to b2graph
      scala_build.shared_nothing_checkpoint();
      scala_build.trans_to_b2graph();
    } else {
      scala_build.shared_nothing_checkpoint();
      if (index_param.machine_id == 0) {
        load_global_medoid(index_param);
      }
    }

    // for K-means shard-nothing
    if (index_param.machine_id == 0 &&
        !std::filesystem::exists(index_param.top_index_path)) {
      scala_build.topindex_checkpoint();
      HNSWTopIndex top_index;
      top_index.build_hnsw_top_index(scala_build.index_param, cmd);
    }
    scala_build.rdma_comm.build_sync();
  }
  
  // Delete temp files.
  // scala_build.delete_temp_files();
  return 0;
}