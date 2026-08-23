#!/usr/bin/bash
# Harmony DEEP 10M: 6 physical nodes, 7 MPI ranks (rank 0 coordinator + 6 workers).
# Slurm places 2 ranks on one node (coordinator colocated with a worker); 1 rank on each of the other 5 nodes.
# Topology: --mode group --group 4 --team 3 (6 workers).
#
# Env overrides: HARMONY_K, HARMONY_NPROBE, HARMONY_GROUP, HARMONY_TEAM, HARMONY_BLOCK, HARMONY_NLIST
#
#SBATCH --job-name=hmy-deep10m-6n
#SBATCH --partition=ncpushort
#SBATCH --nodes=6
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH -o /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_deep10m_6n_%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_deep10m_6n_%j.err

set -eo pipefail
cd /gpfs/users/delgaudio/exp_eval/Harmony
mkdir -p /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log

eval "$(~/bin/micromamba shell hook -s bash)"
micromamba activate /gpfs/users/delgaudio/local/gcc13
set -u

echo "=== Harmony DEEP 6n: NNODES=${SLURM_NNODES:-?} NTASKS=${SLURM_NTASKS:-?} NODELIST=${SLURM_JOB_NODELIST:-?} ==="
echo "=== HARMONY_K=${HARMONY_K:-10} HARMONY_NPROBE=${HARMONY_NPROBE:-32} group=${HARMONY_GROUP:-4} team=${HARMONY_TEAM:-3} ==="

module purge
module load cmake/3.29.1 intel/20U4/mkl eigen/3.4.0
module unload gcc 2>/dev/null || true
module load openmpi/gcc-10.3.0/4.1.1

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 7 ranks on 6 nodes (2 ranks on one node): 64 cores/task fits 128-core nodes.
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Do not use --map-by ppr:1:node: 7 ranks must fit on 6 nodes (coordinator shares a node).
mpirun -np "${SLURM_NTASKS}" \
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
  --nprobes "${HARMONY_NPROBE:-32}" \
  --cache \
  --verbose
