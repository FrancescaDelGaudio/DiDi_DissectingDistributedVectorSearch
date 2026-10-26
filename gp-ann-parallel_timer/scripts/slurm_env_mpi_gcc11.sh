#!/usr/bin/env bash
# Source from sbatch scripts:  source .../scripts/slurm_env_mpi_gcc11.sh
# Sets OpenMPI (modules) + libstdc++ from gcc/11.3 for release_l2/DistributedBench.
slurm_env_mpi_gcc11() {
  USE_SYSTEM_MPI="${USE_SYSTEM_MPI:-0}"
  if [[ "$USE_SYSTEM_MPI" == "1" ]]; then
    export PATH="/usr/lib64/openmpi/bin:/usr/bin:/bin:${PATH}"
    if ! command -v mpirun >/dev/null 2>&1; then
      echo "FATAL: mpirun assente sotto /usr su questo nodo." >&2
      return 1
    fi
    export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    return 0
  fi

  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi

  OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/gcc-10.3.0/4.1.1}"
  module unload gcc 2>/dev/null || true
  module unload intel 2>/dev/null || true
  module unload mpi/openmpi-x86_64 2>/dev/null || true
  module unload mpi 2>/dev/null || true
  echo "[slurm_env_mpi] module load ${OPENMPI_MODULE} ..." >&2
  if ! module load "$OPENMPI_MODULE"; then
    echo "FATAL: module load $OPENMPI_MODULE fallito." >&2
    module avail openmpi 2>&1 | head -40 || true
    return 1
  fi

  MPIROOT="$(dirname "$(dirname "$(command -v mpirun)")")"
  export LD_LIBRARY_PATH="${MPIROOT}/lib64:${MPIROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ -d /softs/gcc/11.3.0/lib64 ]]; then
    export LD_LIBRARY_PATH="/softs/gcc/11.3.0/lib64:/softs/gcc/11.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

# Array for mpirun -x (call after slurm_env_mpi_gcc11).
slurm_mpi_export_flags() {
  MPIRUN_EXPORT=(-x LD_LIBRARY_PATH -x PATH)
}
