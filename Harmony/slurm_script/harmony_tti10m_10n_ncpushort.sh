#!/usr/bin/bash
# 11 Slurm nodes => 11 MPI ranks (1 master + 10 workers). team=5 => teamSize=2, d=200 OK.
#SBATCH --job-name=hmy-tti11n
#SBATCH --partition=ncpushort
#SBATCH --nodes=11
#SBATCH --ntasks=11
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
# Default logs (overridden by submit_tti_k_nprobe_sweep.sh with k/nprobe in filename).
#SBATCH -o /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_tti10m_%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_tti10m_%j.err

set -eo pipefail
cd /gpfs/users/delgaudio/exp_eval/Harmony
mkdir -p /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log

eval "$(~/bin/micromamba shell hook -s bash)"
micromamba activate /gpfs/users/delgaudio/local/gcc13
set -u

echo "=== Harmony TTI 11n/10w (IP): K=${HARMONY_K:-10} nprobe=${HARMONY_NPROBE:-384} team=${HARMONY_TEAM:-5} group=${HARMONY_GROUP:-10} ==="

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
  --group "${HARMONY_GROUP:-10}" \
  --team "${HARMONY_TEAM:-5}" \
  --block "${HARMONY_BLOCK:-5}" \
  --k "${HARMONY_K:-10}" \
  --nlist "${HARMONY_NLIST:-4096}" \
  --nprobes "${HARMONY_NPROBE:-384}" \
  --cache \
  --verbose
