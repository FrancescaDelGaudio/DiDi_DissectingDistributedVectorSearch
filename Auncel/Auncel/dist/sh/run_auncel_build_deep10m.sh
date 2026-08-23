#!/usr/bin/env bash
#SBATCH --job-name=auncel-deep-build
#SBATCH --partition=ncpum
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -o /gpfs/scratch/delgaudio/vectors/DEEP/logs/auncel_build_%A_%a.out
#SBATCH -e /gpfs/scratch/delgaudio/vectors/DEEP/logs/auncel_build_%A_%a.err

# Offline IVF1024 indexing + error-profile training per shard (DEEP10M / 4 workers).
#
# Prereq:
#   bash sh/run_auncel_prepare_deep10m.sh
#   sbatch sh/run_auncel_gt_deep10m.sh   # wait for all 4 tasks
#
# Submit:
#   cd /gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist
#   sbatch sh/run_auncel_build_deep10m.sh

set -euo pipefail

DIST="${AUNCEL_DIST_ROOT:-/gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist}"
# shellcheck source=sh/auncel_deep10m_env.sh
source "${DIST}/sh/auncel_deep10m_env.sh"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
OUT_DIR="${AUNCEL_INDEX_DIR}/shard${SHARD_ID}"
PREFIX="${OUT_DIR}/shard"

mkdir -p "${AUNCEL_LOG_DIR}" "${OUT_DIR}"

if [[ ! -f "${AUNCEL_SHARDS}/${SHARD_ID}gti.ivecs" ]]; then
  echo "ERROR: missing ${AUNCEL_SHARDS}/${SHARD_ID}gti.ivecs — run run_auncel_gt_deep10m.sh first" >&2
  exit 1
fi

module load intel/20U4/mkl 2>/dev/null || true
export LD_LIBRARY_PATH="${MKLROOT:-}/lib/intel64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CALIBRATION_RECALL="${AUNCEL_CALIBRATION_RECALL:-0.9}"
CALIBRATE_FLAG="--calibration-recall ${CALIBRATION_RECALL}"
if [[ "${AUNCEL_NO_CALIBRATE:-0}" == "1" ]]; then
  CALIBRATE_FLAG="--no-calibrate"
fi

echo "=== build_shard DEEP10M shard=${SHARD_ID} host=$(hostname) ==="
echo "db=${AUNCEL_SHARDS}/${SHARD_ID}db.fbin"
echo "query=${AUNCEL_QUERY}  k=${AUNCEL_K}  index=${AUNCEL_INDEX_KEY}"
echo "out_prefix=${PREFIX}"

cd "${DIST}"

"${DIST}/build_shard" \
  --out-prefix "${PREFIX}" \
  --db "${AUNCEL_SHARDS}/${SHARD_ID}db.fbin" \
  --query "${AUNCEL_QUERY}" \
  --gt "${AUNCEL_SHARDS}/${SHARD_ID}gti.ivecs" \
  --gtd "${AUNCEL_SHARDS}/${SHARD_ID}gtd.fvecs" \
  --index-key "${AUNCEL_INDEX_KEY}" \
  --omp-threads "${OMP_NUM_THREADS}" \
  ${CALIBRATE_FLAG}

ls -la "${OUT_DIR}"
echo "Done shard ${SHARD_ID}"
