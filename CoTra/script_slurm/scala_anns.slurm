#!/bin/bash
# Distributed CoTra query (scala_anns, ScalaGraph V3).
#
# Prerequisites:
#   - Index already built (scala_index) with same dist_fn / data_path / -s / R,L,B,M
#   - MACHINE_NUM=4 in include/rdma/rdma_config.h (matches #SBATCH -N 4)
#   - build/tests/scala_anns
#
# Usage (single run):
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   sbatch --export=ALL,COTRA_DATASET=deep,COTRA_RES_KNN=10 \
#     script_slurm/scala_anns.slurm
#
# Submit all four experiments (DEEP L2 + TTI MIPS, k=10 and k=100):
#   bash script_slurm/submit_anns_sweep.sh
#
# Env:
#   COTRA_DATASET   deep | tti  (selects paths and dist_fn unless overridden)
#   COTRA_RES_KNN   10 | 100 | ...  (must be <= k in GT file; default 10)
#   COTRA_DIST_FN   override l2 / mips
#   COTRA_INDEX_PREFIX, COTRA_DATA_PATH, COTRA_QUERY_PATH, COTRA_GT_PATH

#SBATCH -J cotra_anns
#SBATCH -p ncpumshort
#SBATCH --exclusive
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 00:10:00
#SBATCH --mem=150G
#SBATCH -o /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/scala_anns-%x-%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/scala_anns-%x-%j.err

set -euo pipefail

COTRA_ROOT="/gpfs/users/delgaudio/exp_eval/CoTra"
cd "$COTRA_ROOT"
mkdir -p script_slurm/logs script_slurm/runtime

# Let IB/memcached drain when this job follows another on the same nodes.
COTRA_IB_DRAIN_SEC="${COTRA_IB_DRAIN_SEC:-0}"
if [[ "${COTRA_IB_DRAIN_SEC}" -gt 0 ]]; then
  echo "==== IB drain sleep ${COTRA_IB_DRAIN_SEC}s (consecutive job) ===="
  sleep "${COTRA_IB_DRAIN_SEC}"
fi

# shellcheck disable=SC1091
source script_slurm/env_slurm.sh

export COTRA_MACHINE_NUM=4
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
    COTRA_INDEX_OUT="${COTRA_INDEX_OUT:-/gpfs/scratch/delgaudio/cotra/index/deep10M}"
    ;;
  tti)
    COTRA_DIST_FN="${COTRA_DIST_FN:-mips}"
    COTRA_DATA_PATH="${COTRA_DATA_PATH:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2i10M.fbin}"
    COTRA_QUERY_PATH="${COTRA_QUERY_PATH:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2iquery10K.fbin}"
    COTRA_GT_PATH="${COTRA_GT_PATH:-/gpfs/scratch/delgaudio/vectors/TTI/computedGT_tti10M_10kqueries_k100_cotra.bin}"
    COTRA_INDEX_OUT="${COTRA_INDEX_OUT:-/gpfs/scratch/delgaudio/cotra/index/tti10M}"
    ;;
  *)
    echo "FATAL: unknown COTRA_DATASET=$COTRA_DATASET (use deep or tti)" >&2
    exit 1
    ;;
esac

COTRA_INDEX_PREFIX="${COTRA_INDEX_PREFIX:-${COTRA_INDEX_OUT}/merged_index}"

FAISS_LIB="/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/third_party/faiss/build-gcc10/faiss"
export LD_LIBRARY_PATH="${FAISS_LIB}:${LD_LIBRARY_PATH:-}"

BIN="${COTRA_ROOT}/build/tests/scala_anns"

echo "==== scala_anns start $(date -Is) job=${SLURM_JOB_ID} ===="
echo "dataset=${COTRA_DATASET} dist_fn=${COTRA_DIST_FN} res_knn=${COTRA_RES_KNN}"
echo "nodes=${SLURM_JOB_NODELIST}"
echo "DATA=${COTRA_DATA_PATH}"
echo "QUERY=${COTRA_QUERY_PATH}"
echo "GT=${COTRA_GT_PATH}"
echo "INDEX_PREFIX=${COTRA_INDEX_PREFIX}"
echo "IB_DEV=${COTRA_IB_DEV:-mlx5_0}"

test -x "$BIN" || {
  echo "FATAL: scala_anns not found. Run: bash script_slurm/build_cotra.sh" >&2
  exit 1
}
for f in "$COTRA_DATA_PATH" "$COTRA_QUERY_PATH" "$COTRA_GT_PATH"; do
  test -f "$f" || {
    echo "FATAL: missing file: $f" >&2
    exit 1
  }
done
for i in 0 1 2 3; do
  test -f "${COTRA_INDEX_PREFIX}${i}_final_scala.index" || {
    echo "FATAL: index missing: ${COTRA_INDEX_PREFIX}${i}_final_scala.index" >&2
    echo "  Run scala_index for dataset=${COTRA_DATASET} first." >&2
    exit 1
  }
done
test -f "${COTRA_INDEX_PREFIX}_top.index" || {
  echo "FATAL: top index missing: ${COTRA_INDEX_PREFIX}_top.index" >&2
  exit 1
}

# shellcheck disable=SC1091
source script_slurm/generate_cluster_conf.sh
generate_cluster_conf

bash script_slurm/start_memcached_leader.sh

echo
echo "==== verify IP mapping ===="
srun --ntasks-per-node=1 --cpus-per-task=1 \
  bash "${COTRA_ROOT}/script_slurm/verify_ip_config.sh"

echo
echo "==== launching scala_anns (recall@${COTRA_RES_KNN}${COTRA_EF:+, single ef=${COTRA_EF}}${COTRA_EFS:+, ef list=${COTRA_EFS}}${COTRA_EF:-}${COTRA_EFS:-}; default=21-ef sweep) ===="
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
  -R "$COTRA_R" \
  -L "$COTRA_L" \
  -B "$COTRA_B" \
  -M "$COTRA_M" \
  -T "$COTRA_THREADS" \
  --scala_v3 \
  -s "$COTRA_MILLION" \
  -t "$COTRA_THREADS"

echo "==== scala_anns done $(date -Is) ===="
echo "Log: script_slurm/logs/scala_anns-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "Columns per ef line: ef  recall  avg_lat_us  qps"
