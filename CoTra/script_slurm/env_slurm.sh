#!/bin/bash
# Shared Slurm / module environment for CoTra jobs.
# Usage: source script_slurm/env_slurm.sh

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

command -v module >/dev/null 2>&1 || {
  echo "FATAL: 'module' command not available."
  return 1 2>/dev/null || exit 1
}

module purge
module load gcc/10.3.0
module load infiniband-diags 2>/dev/null || true
module load rdma-core 2>/dev/null || true

# Build CoTra on login node: module load cmake/3.29.1
# module load tbb

export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# User-built memcached / libmemcached (see script_slurm/build_cotra_deps.sh)
# Stella/ncpum nodes expose mlx5_0 (not mlx4_0); pass -d to scala_index/scala_anns.
export COTRA_IB_DEV="${COTRA_IB_DEV:-mlx5_0}"

export COTRA_DEPS_PREFIX="${COTRA_DEPS_PREFIX:-/gpfs/scratch/delgaudio/cotra/deps}"
if [[ -d "$COTRA_DEPS_PREFIX/bin" ]]; then
  export PATH="$COTRA_DEPS_PREFIX/bin:$PATH"
  export LD_LIBRARY_PATH="$COTRA_DEPS_PREFIX/lib:$COTRA_DEPS_PREFIX/lib64:${LD_LIBRARY_PATH}"
  export PKG_CONFIG_PATH="$COTRA_DEPS_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  export CMAKE_PREFIX_PATH="$COTRA_DEPS_PREFIX:${CMAKE_PREFIX_PATH:-}"
fi
