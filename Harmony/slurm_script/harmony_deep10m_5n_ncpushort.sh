#!/usr/bin/bash
#SBATCH --job-name=hmy-deep10m-5n
#SBATCH --partition=ncpushort
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH -o /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_deep10m_%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_deep10m_%j.err

set -eo pipefail
cd /gpfs/users/delgaudio/exp_eval/Harmony
mkdir -p /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log

eval "$(~/bin/micromamba shell hook -s bash)"
micromamba activate /gpfs/users/delgaudio/local/gcc13
set -u

echo "=== Harmony DEEP run: HARMONY_K=${HARMONY_K:-10} HARMONY_NPROBE=${HARMONY_NPROBE:-192} ==="

module purge
module load cmake/3.29.1 intel/20U4/mkl eigen/3.4.0
module unload gcc 2>/dev/null || true
module load openmpi/gcc-10.3.0/4.1.1

# Use GCC13 runtime libs (build toolchain), not GCC10 libstdc++ from OpenMPI module.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# One MPI rank per node, each rank uses full node cores for OpenMP.
export OMP_NUM_THREADS=128
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# OpenMPI module here is not built with Slurm PMI; use mpirun inside allocation.
_hosts=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | paste -sd,)
mpirun -np "${SLURM_NTASKS}" \
  --map-by ppr:1:node \
  --host "${_hosts}" \
  --bind-to none \
  ./build-harmony/bin/query \
  --benchmarks_path /gpfs/scratch/delgaudio/vectors \
  --dataset DEEP \
  --input_format fvecs \
  --output_format bin \
  --mode group \
  --group "${HARMONY_GROUP:-4}" \
  --team "${HARMONY_TEAM:-3}" \
  --block "${HARMONY_BLOCK:-5}" \
  --k "${HARMONY_K:-10}" \
  --nlist "${HARMONY_NLIST:-4096}" \
  --nprobes "${HARMONY_NPROBE:-192}" \
  --cache \
  --verbose
