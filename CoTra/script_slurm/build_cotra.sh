#!/bin/bash
# Configure and build CoTra on Stella (login node).
#
# Prerequisites:
#   bash script_slurm/build_cotra_deps.sh
#   /gpfs/users/delgaudio/exp_eval/Distributed-HNSW/script/build_faiss_gcc10.sh
#
# Usage:
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   bash script_slurm/build_cotra.sh
#   bash script_slurm/build_cotra.sh --target scala_index

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${COTRA_BUILD_DIR:-${ROOT}/build}"
TARGET="${1:-}"

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

if [[ ! -f "${COTRA_DEPS_PREFIX}/include/gperftools/malloc_extension.h" ]]; then
  echo "FATAL: gperftools not in ${COTRA_DEPS_PREFIX}. Run script_slurm/build_cotra_deps.sh"
  exit 1
fi

echo "=== CMake configure: ${BUILD_DIR} ==="
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(which g++)" \
  -DCMAKE_C_COMPILER="$(which gcc)" \
  -DCMAKE_PREFIX_PATH="${COTRA_DEPS_PREFIX}" \
  -DCOTRA_DEPS_PREFIX="${COTRA_DEPS_PREFIX}" \
  -DFAISS_ROOT="${FAISS_ROOT}" \
  -DFAISS_BUILD_DIR="${FAISS_BUILD}" \
  -DGPERFTOOLS_ROOT="${COTRA_DEPS_PREFIX}"

if [[ "$TARGET" == "--target" ]]; then
  shift
  TARGET="${1:?missing target name after --target}"
  echo "=== Building target: ${TARGET} ==="
  cmake --build "${BUILD_DIR}" --target "${TARGET}" -j"$(nproc)"
else
  echo "=== Building all ==="
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

echo "=== Binaries ==="
ls -la "${BUILD_DIR}/tests/"scala_* 2>/dev/null || true
