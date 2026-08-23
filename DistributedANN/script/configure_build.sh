#!/usr/bin/bash
# Configure and build DistributedANN with the cluster MPI stack (gcc 10 + OpenMPI).
# Run this instead of bare "cmake .." to avoid picking up Anaconda MPI from the login env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

module purge
module load gcc/10.3.0
module load openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "mpicxx=$(command -v mpicxx)"
echo "mpirun=$(command -v mpirun)"

mkdir -p "${BUILD}"
cd "${BUILD}"
if [[ "${1:-}" == "--clean" ]]; then
  rm -rf ./*
  shift
fi

cmake .. -DCMAKE_BUILD_TYPE=Release "$@"
make -j"$(nproc 2>/dev/null || echo 4)"

echo
echo "=== MPI linkage check (build_index) ==="
ldd "${BUILD}/bin/build_index" | grep -E "libmpi|not found"
readelf -d "${BUILD}/bin/build_index" | grep RUNPATH || true
