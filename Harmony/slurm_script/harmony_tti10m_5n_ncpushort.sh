#!/usr/bin/bash
#SBATCH --job-name=hmy-tti10m-5n
#SBATCH --partition=ncpushort
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH -o /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_tti10m_5n_%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_tti10m_5n_%j.err

set -eo pipefail
cd /gpfs/users/delgaudio/exp_eval/Harmony
mkdir -p /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log

eval "$(~/bin/micromamba shell hook -s bash)"
micromamba activate /gpfs/users/delgaudio/local/gcc13
set -u

echo "=== Harmony TTI 5n (IP): HARMONY_K=${HARMONY_K:-10} HARMONY_NPROBE=${HARMONY_NPROBE:-48} ==="

module purge
module load cmake/3.29.1 intel/20U4/mkl eigen/3.4.0
module unload gcc 2>/dev/null || true
module load openmpi/gcc-10.3.0/4.1.1

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=128
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

_hosts=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | paste -sd,)
mpirun -np "${SLURM_NTASKS}" \
  --map-by ppr:1:node \
  --host "${_hosts}" \
  --bind-to none \
  ./build-harmony/bin/query \
  --benchmarks_path /gpfs/scratch/delgaudio/vectors \
  --dataset TTI \
  --input_format fvecs \
  --output_format bin \
  --metric ip \
  --mode group \
  --group "${HARMONY_GROUP:-2}" \
  --team "${HARMONY_TEAM:-2}" \
  --block "${HARMONY_BLOCK:-5}" \
  --k "${HARMONY_K:-10}" \
  --nlist "${HARMONY_NLIST:-4096}" \
  --nprobes "${HARMONY_NPROBE:-48}" \
  --cache \
  --verbose
