#!/bin/bash
# Build CoTra with MACHINE_NUM=6 into build-6p/ (separate from build/ for 4 partitions).
#
# Produces:
#   build-6p/tests/scala_index_6p
#   build-6p/tests/scala_anns_6p
#
# Does NOT touch build/tests/scala_index (4p jobs in queue keep using that binary).
#
# Usage:
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   bash script_slurm/build_cotra_6p.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${COTRA_BUILD_DIR:-${ROOT}/build-6p}"

# shellcheck disable=SC1091
source "${ROOT}/script_slurm/env_slurm.sh"
module load cmake/3.29.1

export COTRA_DEPS_PREFIX="${COTRA_DEPS_PREFIX:-/gpfs/scratch/delgaudio/cotra/deps}"
export LD_LIBRARY_PATH="${COTRA_DEPS_PREFIX}/lib:${COTRA_DEPS_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"

FAISS_ROOT="${FAISS_ROOT:-/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/third_party/faiss}"
FAISS_BUILD="${FAISS_ROOT}/build-gcc10"
if [[ ! -f "${FAISS_BUILD}/faiss/libfaiss.so" && ! -f "${FAISS_BUILD}/libfaiss.so" ]]; then
  echo "FATAL: Faiss not built. Run Distributed-HNSW/script/build_faiss_gcc10.sh first."
  exit 1
fi

echo "=== CMake configure (MACHINE_NUM=6): ${BUILD_DIR} ==="
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(which g++)" \
  -DCMAKE_C_COMPILER="$(which gcc)" \
  -DCMAKE_PREFIX_PATH="${COTRA_DEPS_PREFIX}" \
  -DCOTRA_DEPS_PREFIX="${COTRA_DEPS_PREFIX}" \
  -DCOTRA_MACHINE_NUM=6 \
  -DFAISS_ROOT="${FAISS_ROOT}" \
  -DFAISS_BUILD_DIR="${FAISS_BUILD}" \
  -DGPERFTOOLS_ROOT="${COTRA_DEPS_PREFIX}"

echo "=== Building scala_index_6p scala_anns_6p ==="
cmake --build "${BUILD_DIR}" --target scala_index_6p scala_anns_6p -j"$(nproc)"

echo "=== 6-partition binaries ==="
ls -la "${BUILD_DIR}/tests/scala_index_6p" "${BUILD_DIR}/tests/scala_anns_6p"
"${BUILD_DIR}/tests/scala_index_6p" --help 2>&1 | head -1 || true
echo "Submit indexing: sbatch script_slurm/scala_index_deep10M_6p.slurm"
echo "                 sbatch script_slurm/scala_index_tti10M_6p.slurm"
