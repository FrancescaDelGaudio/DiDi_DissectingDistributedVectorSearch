#!/usr/bin/bash
# Harmony TTI 10M (MIPS/IP): 4 physical nodes, 5 MPI ranks (rank 0 coordinator + 4 workers).
# Slurm places 2 ranks on one node (coordinator colocated with a worker); 1 rank on each of the other 3 nodes.
# Topology: --mode group --group 2 --team 2 (4 workers).
#
# Env overrides: HARMONY_K, HARMONY_NPROBE, HARMONY_GROUP, HARMONY_TEAM, HARMONY_BLOCK, HARMONY_NLIST
#
#SBATCH --job-name=hmy-tti10m-4n
#SBATCH --partition=ncpushort
#SBATCH --nodes=4
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH -o /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_tti10m_4n_%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_tti10m_4n_%j.err

set -eo pipefail
cd /gpfs/users/delgaudio/exp_eval/Harmony
mkdir -p /gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log

eval "$(~/bin/micromamba shell hook -s bash)"
micromamba activate /gpfs/users/delgaudio/local/gcc13
set -u

echo "=== Harmony TTI 4n: NNODES=${SLURM_NNODES:-?} NTASKS=${SLURM_NTASKS:-?} NODELIST=${SLURM_JOB_NODELIST:-?} ==="
echo "=== HARMONY_K=${HARMONY_K:-10} HARMONY_NPROBE=${HARMONY_NPROBE:-48} group=${HARMONY_GROUP:-2} team=${HARMONY_TEAM:-2} ==="

module purge
module load cmake/3.29.1 intel/20U4/mkl eigen/3.4.0
module unload gcc 2>/dev/null || true
module load openmpi/gcc-10.3.0/4.1.1

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 5 ranks on 4 nodes (2 ranks on one node): 64 cores/task fits 128-core nodes.
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Do not use --map-by ppr:1:node: 5 ranks must fit on 4 nodes (coordinator shares a node).
mpirun -np "${SLURM_NTASKS}" \
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
