#!/usr/bin/bash
#SBATCH --job-name=hmy-iris-5n
#SBATCH --partition=ncpushort
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
# Paths are absolute so logs land in slurm_script/log regardless of sbatch cwd.
#SBATCH -o /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_iris_%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_iris_%j.err

set -eo pipefail
cd /gpfs/users/delgaudio/exp_eval/Harmony
mkdir -p /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log

eval "$(~/bin/micromamba shell hook -s bash)"
micromamba activate /gpfs/users/delgaudio/local/gcc13
set -u

module purge
module load cmake/3.29.1 intel/20U4/mkl eigen/3.4.0
module unload gcc 2>/dev/null || true
module load openmpi/gcc-10.3.0/4.1.1

# Use GCC13 runtime libs (built with micromamba toolchain), not GCC10 from OpenMPI module.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=128
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Softs OpenMPI was not built with SLURM PMI support; srun direct launch fails PMIx handshake.
# Use mpirun with hosts from SLURM (--host is comma-separated).
_hosts=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | paste -sd,)
mpirun -np "${SLURM_NTASKS}" \
  --map-by ppr:1:node \
  --host "${_hosts}" \
  --bind-to none \
  ./build-harmony/bin/query \
  --benchmarks_path ./benchmarks \
  --dataset iris \
  --mode group \
  --group 2 \
  --team 2 \
  --block 5 \
  --k 10 \
  --nprobes 12 \
  --verbose
