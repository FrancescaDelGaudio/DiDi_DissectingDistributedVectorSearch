#!/usr/bin/bash
# Build release_l2_distbench_np<N>/DistributedBench (L2 / DEEP) with fixed shardsProbe.
#
# Usage:
#   ./scripts/build_release_l2_distbench.sh 3
#   ./scripts/build_release_l2_distbench.sh 4
#   ./scripts/build_release_l2_distbench.sh 6
#
# Output: ${GP_ANN_ROOT}/release_l2_distbench_np<N>/DistributedBench

set -euo pipefail

NP="${1:?shardsProbe value required (e.g. 3, 4, or 6)}"
GP_ANN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${GP_ANN_ROOT}/release_l2_distbench_np${NP}"
MPI_ROOT="/softs/openmpi/gcc-10.3.0/4.1.1"

module purge 2>/dev/null || true
module unload gcc 2>/dev/null || true
module load gcc/11.3.0
module load cmake/3.29.1

TBB_ROOT="${TBB_ROOT:-/gpfs/softs/intel/oneapi/tbb/2021.2.0}"
TBB_CMAKE_DIR="${TBB_ROOT}/lib/cmake/tbb"
if module load tbb 2>/dev/null; then
  echo "TBB OK (module)"
elif test -f "${TBB_CMAKE_DIR}/TBBConfig.cmake"; then
  echo "TBB OK (filesystem: ${TBB_ROOT})"
else
  echo "FATAL: TBB not found. Try: module avail tbb  or set TBB_ROOT." >&2
  exit 1
fi

echo "=== Configuring ${BUILD_DIR} (DISTBENCH_SHARDS_PROBE=${NP}) ==="
cmake -S "${GP_ANN_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${MPI_ROOT}/bin/mpicc" \
  -DCMAKE_CXX_COMPILER="${MPI_ROOT}/bin/mpicxx" \
  -DMPI_C_COMPILER="${MPI_ROOT}/bin/mpicc" \
  -DMPI_CXX_COMPILER="${MPI_ROOT}/bin/mpicxx" \
  -DMPI_HOME="${MPI_ROOT}" \
  -DDISTBENCH_SHARDS_PROBE="${NP}" \
  -DTBB_DIR="${TBB_CMAKE_DIR}"

echo "=== Building DistributedBench ==="
cmake --build "${BUILD_DIR}" --target DistributedBench -j16

test -x "${BUILD_DIR}/DistributedBench"
echo "=== OK: ${BUILD_DIR}/DistributedBench (shardsProbe=${NP}) ==="
"${BUILD_DIR}/DistributedBench" 2>&1 | head -1 || true
