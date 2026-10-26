#!/usr/bin/bash
# Build build_rep_np<N>/DistributedBench (L2) for the repetition and replay campaign:
# same code as build_np<N>, plus GP_SHARD_INDEX_DIR (save/load shard graphs) and GP_LOG_BYTES.
# The older build_np<N> / build_comm_np<N> are left untouched.
# Usage: scripts/build_rep.sh "1 2 3 4"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH=/softs/cmake/3.29.1/bin:/softs/openmpi/gcc-10.3.0/4.1.1/bin:/softs/gcc/11.3.0/bin:${PATH}
export LD_LIBRARY_PATH=/softs/gcc/11.3.0/lib64:/softs/openmpi/gcc-10.3.0/4.1.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

for np in ${1:-1 2 3 4}; do
  dir="${ROOT}/build_rep_np${np}"
  echo "=== nprobe=${np} -> ${dir} ==="
  cmake -S "${ROOT}" -B "${dir}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
    -DDISTBENCH_SHARDS_PROBE="${np}" -DMIPS_DISTANCE=OFF \
    -DTBB_DIR=/gpfs/softs/intel/oneapi/tbb/2021.2.0/lib/cmake/tbb > "${dir}.configure.log" 2>&1
  cmake --build "${dir}" --target DistributedBench -j 16 > "${dir}.build.log" 2>&1
  ls -la "${dir}/DistributedBench"
done
