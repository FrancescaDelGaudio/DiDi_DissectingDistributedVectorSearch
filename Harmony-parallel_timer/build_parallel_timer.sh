#!/usr/bin/bash
# Build Harmony-parallel_timer (Harmony + "#[prof ]" per-phase timers) into build-harmony/,
# with the same toolchain and CMake options as /gpfs/users/delgaudio/exp_eval/Harmony/build-harmony.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
eval "$(~/bin/micromamba shell hook -s bash)"
micromamba activate /gpfs/users/delgaudio/local/gcc13
source /etc/profile.d/modules.sh
module purge
module load cmake/3.29.1 intel/20U4/mkl eigen/3.4.0
module unload gcc 2>/dev/null || true
module load openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cmake -B build-harmony -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
  -DFAISS_OPT_LEVEL=generic -DHARMONY_ENABLE_AVX512=OFF \
  -DGTest_DIR=/gpfs/users/delgaudio/local/gtest/lib64/cmake/GTest .
cmake --build build-harmony -j "${JOBS:-16}" --target query
ls -la build-harmony/bin/query
