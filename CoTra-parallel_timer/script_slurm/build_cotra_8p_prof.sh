#!/bin/bash
# parallel_timer: build CoTra scala_anns_8p with its built-in profiler (-DPROFILER) into build-8p-prof/.
# Same sources and options as build_cotra_8p.sh; the search code is unchanged.
#
# Produces:
#   build-8p/tests/scala_index_8p
#   build-8p/tests/scala_anns_8p
#
# Usage:
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   bash script_slurm/build_cotra_8p.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${COTRA_BUILD_DIR:-${ROOT}/build-8p-prof}"

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

LINK_FLAGS="-L/softs/gcc/11.3.0/lib64 -Wl,-rpath,/softs/gcc/11.3.0/lib64:/softs/gcc/10.3.0/lib64"

echo "=== CMake configure (MACHINE_NUM=8): ${BUILD_DIR} ==="
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(which g++)" \
  -DCMAKE_C_COMPILER="$(which gcc)" \
  -DCMAKE_EXE_LINKER_FLAGS="${LINK_FLAGS}" \
  -DCMAKE_SHARED_LINKER_FLAGS="${LINK_FLAGS}" \
  -DCMAKE_PREFIX_PATH="${COTRA_DEPS_PREFIX}" \
  -DCOTRA_DEPS_PREFIX="${COTRA_DEPS_PREFIX}" \
  -DCOTRA_MACHINE_NUM=8 \
  -DCMAKE_CXX_FLAGS="-DPROFILER" \
  -DFAISS_ROOT="${FAISS_ROOT}" \
  -DFAISS_BUILD_DIR="${FAISS_BUILD}" \
  -DGPERFTOOLS_ROOT="${COTRA_DEPS_PREFIX}"

echo "=== Building scala_index_8p scala_anns_8p ==="
cmake --build "${BUILD_DIR}" --target scala_anns_8p -j"$(nproc)"

echo "=== 8-partition binaries ==="
ls -la "${BUILD_DIR}/tests/scala_anns_8p"
"${BUILD_DIR}/tests/scala_index_8p" --help 2>&1 | head -1 || true
echo "Submit indexing: sbatch script_slurm/scala_index_deep100M_8p.slurm"
