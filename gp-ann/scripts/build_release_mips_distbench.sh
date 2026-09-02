#!/usr/bin/bash
# Build release_mips_distbench_np<N>/DistributedBench (MIPS / TTI) with fixed shardsProbe.
#
# Usage:
#   ./scripts/build_release_mips_distbench.sh 3
#   ./scripts/build_release_mips_distbench.sh 4
#   ./scripts/build_release_mips_distbench.sh 6
#
# Verify after build:
#   strings release_mips_distbench_np3/DistributedBench | grep -E 'InnerProduct|ef_search'

set -euo pipefail

NP="${1:?shardsProbe value required (e.g. 3, 4, or 6)}"
GP_ANN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${GP_ANN_ROOT}/release_mips_distbench_np${NP}"
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

echo "=== Configuring ${BUILD_DIR} (MIPS_DISTANCE=ON DISTBENCH_SHARDS_PROBE=${NP}) ==="
cmake -S "${GP_ANN_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${MPI_ROOT}/bin/mpicc" \
  -DCMAKE_CXX_COMPILER="${MPI_ROOT}/bin/mpicxx" \
  -DMPI_C_COMPILER="${MPI_ROOT}/bin/mpicc" \
  -DMPI_CXX_COMPILER="${MPI_ROOT}/bin/mpicxx" \
  -DMPI_HOME="${MPI_ROOT}" \
  -DMIPS_DISTANCE=ON \
  -DDISTBENCH_SHARDS_PROBE="${NP}" \
  -DTBB_DIR="${TBB_CMAKE_DIR}"

echo "=== Building DistributedBench ==="
cmake --build "${BUILD_DIR}" --target DistributedBench -j16

test -x "${BUILD_DIR}/DistributedBench"
IP_COUNT=$(strings "${BUILD_DIR}/DistributedBench" | grep -cE 'InnerProduct' || true)
L2_COUNT=$(strings "${BUILD_DIR}/DistributedBench" | grep -cE 'L2Space' || true)
if [[ "${IP_COUNT}" -lt 1 ]] || [[ "${L2_COUNT}" -gt 0 ]]; then
  echo "FATAL: expected MIPS (InnerProduct) binary; InnerProduct=${IP_COUNT} L2Space=${L2_COUNT}" >&2
  exit 1
fi
echo "=== OK: ${BUILD_DIR}/DistributedBench (MIPS, shardsProbe=${NP}, InnerProduct symbols=${IP_COUNT}) ==="
"${BUILD_DIR}/DistributedBench" 2>&1 | head -1 || true
