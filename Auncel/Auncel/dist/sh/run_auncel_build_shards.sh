#!/usr/bin/env bash
#SBATCH --job-name=auncel-build
#SBATCH --partition=ncpushort
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /gpfs/scratch/delgaudio/vectors/toy/logs/auncel_build_%A_%a.out
#SBATCH -e /gpfs/scratch/delgaudio/vectors/toy/logs/auncel_build_%A_%a.err

# Offline indexing per shard (random_100k, 50 queries, k=10).
#
# Prereq:
#   bash sh/run_auncel_prepare_100k.sh
#
# Submit:
#   cd /gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist
#   sbatch sh/run_auncel_build_shards.sh

set -euo pipefail

DIST="${AUNCEL_DIST_ROOT:-/gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist}"
# shellcheck source=sh/auncel_100k_env.sh
source "${DIST}/sh/auncel_100k_env.sh"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
OUT_DIR="${AUNCEL_INDEX_DIR}/shard${SHARD_ID}"
PREFIX="${OUT_DIR}/shard"

mkdir -p "${AUNCEL_LOG_DIR}" "${OUT_DIR}"

module load intel/20U4/mkl 2>/dev/null || true
export LD_LIBRARY_PATH="${MKLROOT:-}/lib/intel64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "=== build_shard shard=${SHARD_ID} host=$(hostname) ==="
echo "dataset: random_100k  query=${AUNCEL_QUERY}  k=${AUNCEL_K}"
echo "index_key=${AUNCEL_INDEX_KEY}"
echo "out_prefix=${PREFIX}"

cd "$DIST"

"$DIST/build_shard" \
  --out-prefix "$PREFIX" \
  --db "${AUNCEL_SHARDS}/${SHARD_ID}db.fbin" \
  --query "$AUNCEL_QUERY" \
  --gt "${AUNCEL_SHARDS}/${SHARD_ID}gti.ivecs" \
  --gtd "${AUNCEL_SHARDS}/${SHARD_ID}gtd.fvecs" \
  --index-key "$AUNCEL_INDEX_KEY" \
  --omp-threads "$OMP_NUM_THREADS"

ls -la "$OUT_DIR"
echo "Done shard ${SHARD_ID}"
