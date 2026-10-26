#!/usr/bin/bash
# Build extra DistributedBench variants (nprobe is a compile-time constant).
#   L2   : build_np<N>
#   MIPS : build_mips_np<N>
# Usage: scripts/build_distbench_variants.sh "l2:8 mips:1 mips:2 ..."
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH=/softs/cmake/3.29.1/bin:/softs/openmpi/gcc-10.3.0/4.1.1/bin:/softs/gcc/11.3.0/bin:${PATH}
export LD_LIBRARY_PATH=/softs/gcc/11.3.0/lib64:/softs/openmpi/gcc-10.3.0/4.1.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

for v in ${1:?variants required, e.g. "l2:8 mips:1"}; do
  metric="${v%%:*}"
  np="${v##*:}"
  if [[ "$metric" == "mips" ]]; then
    dir="${ROOT}/build_mips_np${np}"; mips=ON
  else
    dir="${ROOT}/build_np${np}"; mips=OFF
  fi
  echo "=== ${metric} nprobe=${np} -> ${dir} ==="
  cmake -S "${ROOT}" -B "${dir}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
    -DDISTBENCH_SHARDS_PROBE="${np}" -DMIPS_DISTANCE="${mips}" \
    -DTBB_DIR=/gpfs/softs/intel/oneapi/tbb/2021.2.0/lib/cmake/tbb > "${dir}.configure.log" 2>&1
  cmake --build "${dir}" --target DistributedBench -j 16 > "${dir}.build.log" 2>&1
  ls -la "${dir}/DistributedBench"
done
