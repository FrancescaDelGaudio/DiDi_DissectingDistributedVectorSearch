#!/bin/bash
# Build vendored Faiss (CPU only) with gcc 10.3 + MKL for Distributed-HNSW.
#
# Output:
#   third_party/faiss/build-gcc10/libfaiss.so
#   third_party/faiss/build-gcc10/libfaiss.a   (if static variant built)
#
# Usage:
#   ./script/build_faiss_gcc10.sh
#   FAISS_OPT_LEVEL=avx2 ./script/build_faiss_gcc10.sh

set -euo pipefail

ROOT="/gpfs/users/delgaudio/exp_eval/Distributed-HNSW"
SRC="${ROOT}/third_party/faiss"
BUILD="${SRC}/build-gcc10"
FAISS_OPT_LEVEL="${FAISS_OPT_LEVEL:-generic}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

module purge
module load cmake/3.29.1
module load gcc/10.3.0
module load intel/20U4/mkl

export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "${BUILD}"
cd "${BUILD}"

cmake "${SRC}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_RAFT=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_C_API=OFF \
  -DFAISS_OPT_LEVEL="${FAISS_OPT_LEVEL}" \
  -DBUILD_SHARED_LIBS=ON

make -j"$(nproc)" faiss

echo "==== Faiss built with $(gcc --version | head -1) ===="
ls -la "${BUILD}/faiss/libfaiss.so" 2>/dev/null || ls -la "${BUILD}/libfaiss.so"
echo "Include: ${SRC}"
echo "Link:    ${BUILD}/faiss/libfaiss.so (or libfaiss.a)"
