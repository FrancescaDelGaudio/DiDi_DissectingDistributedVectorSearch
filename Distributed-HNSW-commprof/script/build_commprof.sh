#!/usr/bin/bash
# Build the commprof (communication breakdown) binaries in their own tree.
# Never touches /gpfs/users/delgaudio/exp_eval/Distributed-HNSW/build.
#
# Produces build/bug_{4,6,8}_k{10,100}, build/bug_mips_{4,6,8}_k{10,100}, build/bug_toy
# with the same flags as the production binaries (-g -O2 -DNDEBUG, gcc 15.3.0).
#
# Usage:
#   ./script/build_commprof.sh            # all layout binaries + bug_toy
#   ./script/build_commprof.sh bug_toy    # only the given targets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD="${BUILD:-${ROOT}/build}"
JOBS="${JOBS:-8}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
module purge
module load gcc/15.3.0
module load openmpi/gcc-15.3.0/4.1.8
module load cmake/3.29.1
export LD_LIBRARY_PATH="/softs/gcc/15.3.0/lib64:/softs/gcc/15.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

TARGETS=("$@")
if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  TARGETS=(bug_toy)
  for w in 4 6 8; do
    for k in 10 100; do
      TARGETS+=("bug_${w}_k${k}" "bug_mips_${w}_k${k}")
    done
  done
fi

# -include cstdint: gcc 15 no longer pulls it in transitively (utils/calculator.hpp needs uint64_t).
cmake -S "$ROOT" -B "$BUILD" \
  -DBUILD_MPI_ONLY=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER="$(command -v mpicc)" \
  -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
  -DCMAKE_CXX_FLAGS="-I/softs/eigen/3.4.0/include/eigen3 -include cstdint"

make -C "$BUILD" -j "$JOBS" "${TARGETS[@]}"

echo "OK: built in $BUILD:"
ls -la "${TARGETS[@]/#/$BUILD/}"
