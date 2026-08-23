#!/usr/bin/env bash
#SBATCH --job-name=auncel-search
#SBATCH --partition=ncpushort
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -o /gpfs/scratch/delgaudio/vectors/toy/logs/auncel_search_%j.out
#SBATCH -e /gpfs/scratch/delgaudio/vectors/toy/logs/auncel_search_%j.err

# Distributed search on random_100k (50 queries, k=10, 2 workers + master).
#
# Prereq:
#   bash sh/run_auncel_prepare_100k.sh
#   sbatch sh/run_auncel_build_shards.sh
#
# Submit:
#   AUNCEL_RECALL=0.9 sbatch sh/run_auncel_search_3node.sh

set -euo pipefail

DIST="${AUNCEL_DIST_ROOT:-/gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist}"
# shellcheck source=sh/auncel_100k_env.sh
source "${DIST}/sh/auncel_100k_env.sh"
LAUNCHER="${DIST}/sh/auncel_worker_launch.sh"
LOG_DIR="${AUNCEL_LOG_DIR}"
RESULTS="${AUNCEL_DATA_DIR}/auncel_results_${SLURM_JOB_ID}"
COORD="${AUNCEL_DATA_DIR}/auncel_coord_${SLURM_JOB_ID}"

GLOBAL_RECALL="${AUNCEL_RECALL:-0.9}"
HYPERPARAM_ID="${AUNCEL_HP_ID:-1}"

mkdir -p "$LOG_DIR" "$RESULTS" "$COORD"
rm -f "${COORD}/ip0" "${COORD}/ip1"

for s in 0 1; do
  if [[ ! -f "${AUNCEL_INDEX_DIR}/shard${s}/shard.faissindex" ]]; then
    echo "ERROR: missing ${AUNCEL_INDEX_DIR}/shard${s}/shard.faissindex" >&2
    echo "  Run: bash sh/run_auncel_prepare_100k.sh && sbatch sh/run_auncel_build_shards.sh" >&2
    exit 1
  fi
done

module load intel/20U4/mkl 2>/dev/null || true
export LD_LIBRARY_PATH="${MKLROOT:-}/lib/intel64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export AUNCEL_EXIT_AFTER_SEARCH=1

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
W0_NODE="${NODES[1]}"
W1_NODE="${NODES[2]}"

echo "=== Auncel search (random_100k) ==="
echo "master: ${NODES[0]} ($(hostname))"
echo "workers: ${W0_NODE} ${W1_NODE}"
echo "base: ${AUNCEL_BASE}"
echo "query: ${AUNCEL_QUERY}"
echo "global_gt: ${AUNCEL_GT}"
echo "index_dir: ${AUNCEL_INDEX_DIR}"
echo "global_recall: ${GLOBAL_RECALL}  k: ${AUNCEL_K}  shard_offset: ${AUNCEL_SHARD_OFFSET}"
echo "hyperparam_id: ${HYPERPARAM_ID}"
echo "results: ${RESULTS}"

chmod +x "$LAUNCHER"

launch_worker() {
  local shard="$1"
  local node="$2"
  local log="${LOG_DIR}/auncel_search_${SLURM_JOB_ID}_w${shard}.log"
  srun --nodes=1 --nodelist="${node}" --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash "${LAUNCHER}" "${shard}" "${COORD}" "${DIST}" \
    "${AUNCEL_INDEX_DIR}/shard${shard}/shard" \
    "${AUNCEL_QUERY}" \
    "${AUNCEL_SHARDS}/${shard}gti.ivecs" \
    "${AUNCEL_SHARDS}/${shard}gtd.fvecs" \
    "${GLOBAL_RECALL}" \
    "${HYPERPARAM_ID}" \
    > "${log}" 2>&1 &
  echo $!
}

SRUN_W0=$(launch_worker 0 "${W0_NODE}")
SRUN_W1=$(launch_worker 1 "${W1_NODE}")

echo "==> Waiting for worker IPs"
for _ in $(seq 1 300); do
  [[ -f "${COORD}/ip0" && -f "${COORD}/ip1" ]] && break
  sleep 1
done

if [[ ! -f "${COORD}/ip0" || ! -f "${COORD}/ip1" ]]; then
  echo "ERROR: workers did not start" >&2
  tail -30 "${LOG_DIR}/auncel_search_${SLURM_JOB_ID}_w0.log" >&2 || true
  tail -30 "${LOG_DIR}/auncel_search_${SLURM_JOB_ID}_w1.log" >&2 || true
  exit 1
fi

IP0="$(cat "${COORD}/ip0")"
IP1="$(cat "${COORD}/ip1")"
echo "worker0: ${IP0}:3456  worker1: ${IP1}:3456"
sleep 2

cd "${DIST}"

SEARCH_T0=$(date +%s.%N)
echo "==> master search (prebuilt index)"
./master search "${RESULTS}" --global-recall "${GLOBAL_RECALL}" --k "${AUNCEL_K}" \
  "${IP0}:3456" "${IP1}:3456"
SEARCH_T1=$(date +%s.%N)

echo "==> reduce"
REDUCE_T0=$(date +%s.%N)
./reduce "${RESULTS}" 2 0 "${AUNCEL_SHARD_OFFSET}"
REDUCE_T1=$(date +%s.%N)

echo "==> global recall (merged vs ${AUNCEL_GT})"
./eval_results --gt "${AUNCEL_GT}" --result "${RESULTS}"

SEARCH_WALL=$(awk -v t0="$SEARCH_T0" -v t1="$SEARCH_T1" 'BEGIN{printf "%.3f", t1-t0}')
REDUCE_WALL=$(awk -v t0="$REDUCE_T0" -v t1="$REDUCE_T1" 'BEGIN{printf "%.3f", t1-t0}')
TOTAL_WALL=$(awk -v a="$SEARCH_WALL" -v b="$REDUCE_WALL" 'BEGIN{printf "%.3f", a+b}')
echo "Batch wall time: master_search=${SEARCH_WALL}s  reduce=${REDUCE_WALL}s  total=${TOTAL_WALL}s"

ls -la "${RESULTS}"

wait "${SRUN_W0}" 2>/dev/null || true
wait "${SRUN_W1}" 2>/dev/null || true

echo "Done: ${RESULTS}/merged_idx.ivecs"
