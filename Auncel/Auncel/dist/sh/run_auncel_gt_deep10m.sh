#!/usr/bin/env bash
#SBATCH --job-name=auncel-deep-gt
#SBATCH --partition=ncpum
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o /gpfs/scratch/delgaudio/vectors/DEEP/logs/auncel_gt_%A_%a.out
#SBATCH -e /gpfs/scratch/delgaudio/vectors/DEEP/logs/auncel_gt_%A_%a.err

# Brute-force per-shard GT (10k queries, k=100) on 2.5M vectors each.
# Prereq: bash sh/run_auncel_prepare_deep10m.sh

set -euo pipefail

DIST="${AUNCEL_DIST_ROOT:-/gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist}"
# shellcheck source=sh/auncel_deep10m_env.sh
source "${DIST}/sh/auncel_deep10m_env.sh"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

module load intel/20U4/mkl 2>/dev/null || true
export LD_LIBRARY_PATH="${MKLROOT:-}/lib/intel64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== gt shard=${SHARD_ID} host=$(hostname) ==="
cd "${DIST}"

"${DIST}/gt" \
  "${AUNCEL_SHARDS}/${SHARD_ID}db.fbin" \
  "${AUNCEL_QUERY}" \
  "${AUNCEL_K}"

ls -la "${AUNCEL_SHARDS}/${SHARD_ID}"gti.ivecs "${AUNCEL_SHARDS}/${SHARD_ID}"gtd.fvecs
echo "Done shard ${SHARD_ID}"
