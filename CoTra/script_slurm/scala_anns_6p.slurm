#!/bin/bash
# Distributed CoTra query — 6 partitions (scala_anns_6p).
#
# Requires index from scala_index_*_6p.slurm and build-6p/tests/scala_anns_6p
#
# Usage:
#   sbatch --export=ALL,COTRA_DATASET=deep,COTRA_RES_KNN=10 \
#     script_slurm/scala_anns_6p.slurm
#   bash script_slurm/submit_anns_sweep_6p.sh

#SBATCH -J cotra_anns6
#SBATCH -p ncpumshort
#SBATCH --exclusive
#SBATCH -N 6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 00:10:00
#SBATCH --mem=150G
#SBATCH -o /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/scala_anns_6p-%x-%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/scala_anns_6p-%x-%j.err

set -euo pipefail

COTRA_ROOT="/gpfs/users/delgaudio/exp_eval/CoTra"
cd "$COTRA_ROOT"
mkdir -p script_slurm/logs script_slurm/runtime

COTRA_IB_DRAIN_SEC="${COTRA_IB_DRAIN_SEC:-0}"
if [[ "${COTRA_IB_DRAIN_SEC}" -gt 0 ]]; then
  echo "==== IB drain sleep ${COTRA_IB_DRAIN_SEC}s (consecutive job) ===="
  sleep "${COTRA_IB_DRAIN_SEC}"
fi

# shellcheck disable=SC1091
source script_slurm/env_slurm.sh

export COTRA_MACHINE_NUM=6
export COTRA_CONF_DIR="${COTRA_CONF_DIR:-${COTRA_ROOT}/script_slurm/runtime}"

COTRA_DATASET="${COTRA_DATASET:-deep}"
COTRA_RES_KNN="${COTRA_RES_KNN:-10}"
COTRA_MILLION="${COTRA_MILLION:-10}"
COTRA_THREADS="${COTRA_THREADS:-16}"
COTRA_R="${COTRA_R:-48}"
COTRA_L="${COTRA_L:-500}"
COTRA_B="${COTRA_B:-100}"
COTRA_M="${COTRA_M:-120}"

case "$COTRA_DATASET" in
  deep)
    COTRA_DIST_FN="${COTRA_DIST_FN:-l2}"
    COTRA_DATA_PATH="${COTRA_DATA_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.fbin}"
    COTRA_QUERY_PATH="${COTRA_QUERY_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin}"
    COTRA_GT_PATH="${COTRA_GT_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep10M_10kqueries_k100_cotra.bin}"
    COTRA_INDEX_OUT="${COTRA_INDEX_OUT:-/gpfs/scratch/delgaudio/cotra/index/deep10M_6p}"
    ;;
  tti)
    COTRA_DIST_FN="${COTRA_DIST_FN:-mips}"
    COTRA_DATA_PATH="${COTRA_DATA_PATH:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2i10M.fbin}"
    COTRA_QUERY_PATH="${COTRA_QUERY_PATH:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2iquery10K.fbin}"
    COTRA_GT_PATH="${COTRA_GT_PATH:-/gpfs/scratch/delgaudio/vectors/TTI/computedGT_tti10M_10kqueries_k100_cotra.bin}"
    COTRA_INDEX_OUT="${COTRA_INDEX_OUT:-/gpfs/scratch/delgaudio/cotra/index/tti10M_6p}"
    ;;
  *)
    echo "FATAL: unknown COTRA_DATASET=$COTRA_DATASET" >&2
    exit 1
    ;;
esac

COTRA_INDEX_PREFIX="${COTRA_INDEX_PREFIX:-${COTRA_INDEX_OUT}/merged_index}"
BIN="${COTRA_ROOT}/build-6p/tests/scala_anns_6p"

echo "==== scala_anns_6p start $(date -Is) job=${SLURM_JOB_ID} ===="
echo "dataset=${COTRA_DATASET} dist_fn=${COTRA_DIST_FN} res_knn=${COTRA_RES_KNN} MACHINE_NUM=6"
echo "INDEX_PREFIX=${COTRA_INDEX_PREFIX} BIN=${BIN}"

test -x "$BIN" || { echo "FATAL: run build_cotra_6p.sh first" >&2; exit 1; }
for f in "$COTRA_DATA_PATH" "$COTRA_QUERY_PATH" "$COTRA_GT_PATH"; do
  test -f "$f" || { echo "FATAL: missing $f" >&2; exit 1; }
done
for i in 0 1 2 3 4 5; do
  test -f "${COTRA_INDEX_PREFIX}${i}_final_scala.index" || {
    echo "FATAL: missing ${COTRA_INDEX_PREFIX}${i}_final_scala.index" >&2
    exit 1
  }
done
test -f "${COTRA_INDEX_PREFIX}_top.index" || {
  echo "FATAL: missing ${COTRA_INDEX_PREFIX}_top.index" >&2
  exit 1
}

source script_slurm/generate_cluster_conf.sh
generate_cluster_conf
bash script_slurm/start_memcached_leader.sh

srun --ntasks-per-node=1 --cpus-per-task=1 \
  bash "${COTRA_ROOT}/script_slurm/verify_ip_config.sh"

srun --ntasks-per-node=1 --cpus-per-task="${COTRA_THREADS}" --kill-on-bad-exit=1 \
  "$BIN" \
  -d "${COTRA_IB_DEV:-mlx5_0}" \
  --config_file "$CONF_FILE" \
  --app_type scala_v3 \
  --graph_type scalagraph_v3 \
  --data_type float \
  --dist_fn "$COTRA_DIST_FN" \
  --data_path "$COTRA_DATA_PATH" \
  --query_path "$COTRA_QUERY_PATH" \
  --gt_path "$COTRA_GT_PATH" \
  --index_path_prefix "$COTRA_INDEX_PREFIX" \
  --res_knn "$COTRA_RES_KNN" \
  -R "$COTRA_R" -L "$COTRA_L" -B "$COTRA_B" -M "$COTRA_M" \
  -T "$COTRA_THREADS" \
  --scala_v3 \
  -s "$COTRA_MILLION" \
  -t "$COTRA_THREADS"

echo "==== scala_anns_6p done $(date -Is) ===="
