#!/usr/bin/env bash
#SBATCH --job-name=auncel-deep-search
#SBATCH --partition=ncpum
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /gpfs/scratch/delgaudio/vectors/DEEP/logs/auncel_search_%j.out
#SBATCH -e /gpfs/scratch/delgaudio/vectors/DEEP/logs/auncel_search_%j.err

# Distributed search: 1 master + 4 workers, DEEP10M, k=100, §4.5 calibration.
#
# Prereq: build finished for all 4 shards.
#
# Submit:
#   AUNCEL_RECALL=0.9 AUNCEL_HP_ID=0 sbatch sh/run_auncel_search_deep10m.sh
#   AUNCEL_HP_ID=0 uses per-shard (a,b) from build_shard calibration.

set -euo pipefail

DIST="${AUNCEL_DIST_ROOT:-/gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist}"
# shellcheck source=sh/auncel_deep10m_env.sh
source "${DIST}/sh/auncel_deep10m_env.sh"

LAUNCHER="${DIST}/sh/auncel_worker_launch.sh"
LOG_DIR="${AUNCEL_LOG_DIR}"
RESULTS="${AUNCEL_DEEP_DIR}/auncel_results_${SLURM_JOB_ID}"
COORD="${AUNCEL_DEEP_DIR}/auncel_coord_${SLURM_JOB_ID}"

GLOBAL_RECALL="${AUNCEL_RECALL:-0.9}"
HYPERPARAM_ID="${AUNCEL_HP_ID:-0}"

mkdir -p "${LOG_DIR}" "${RESULTS}" "${COORD}"
for s in $(seq 0 $((AUNCEL_NUM_SHARDS - 1))); do
  rm -f "${COORD}/ip${s}"
done

for s in $(seq 0 $((AUNCEL_NUM_SHARDS - 1))); do
  if [[ ! -f "${AUNCEL_INDEX_DIR}/shard${s}/shard.faissindex" ]]; then
    echo "ERROR: missing ${AUNCEL_INDEX_DIR}/shard${s}/shard.faissindex" >&2
    exit 1
  fi
done

module load intel/20U4/mkl 2>/dev/null || true
export LD_LIBRARY_PATH="${MKLROOT:-}/lib/intel64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export AUNCEL_EXIT_AFTER_SEARCH=1

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_NODE="${NODES[0]}"

echo "=== Auncel search DEEP10M (${AUNCEL_NUM_SHARDS} workers) ==="
echo "master: ${MASTER_NODE} ($(hostname))"
echo "workers: ${NODES[*]:1}"
echo "base: ${AUNCEL_BASE}"
echo "query: ${AUNCEL_QUERY}"
echo "global_gt: ${AUNCEL_GT}"
echo "index_dir: ${AUNCEL_INDEX_DIR}"
echo "global_recall: ${GLOBAL_RECALL}  k: ${AUNCEL_K}  hyperparam_id: ${HYPERPARAM_ID}"
echo "shard_offsets: ${AUNCEL_SHARD_OFFSETS}"
echo "results: ${RESULTS}"

chmod +x "${LAUNCHER}"

declare -a SRUN_PIDS=()
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
  SRUN_PIDS+=("$!")
}

for s in $(seq 0 $((AUNCEL_NUM_SHARDS - 1))); do
  launch_worker "${s}" "${NODES[$((s + 1))]}"
done

echo "==> Waiting for worker IPs"
for _ in $(seq 1 600); do
  ready=1
  for s in $(seq 0 $((AUNCEL_NUM_SHARDS - 1))); do
    [[ -f "${COORD}/ip${s}" ]] || ready=0
  done
  [[ "${ready}" -eq 1 ]] && break
  sleep 2
done

for s in $(seq 0 $((AUNCEL_NUM_SHARDS - 1))); do
  if [[ ! -f "${COORD}/ip${s}" ]]; then
    echo "ERROR: worker ${s} did not publish IP" >&2
    tail -30 "${LOG_DIR}/auncel_search_${SLURM_JOB_ID}_w${s}.log" >&2 || true
    exit 1
  fi
done

declare -a ENDPOINTS=()
for s in $(seq 0 $((AUNCEL_NUM_SHARDS - 1))); do
  ip="$(cat "${COORD}/ip${s}")"
  echo "worker${s}: ${ip}:3456"
  ENDPOINTS+=("${ip}:3456")
done
sleep 3

cd "${DIST}"

SEARCH_T0=$(date +%s.%N)
echo "==> master search"
./master search "${RESULTS}" \
  --global-recall "${GLOBAL_RECALL}" \
  --k "${AUNCEL_K}" \
  "${ENDPOINTS[@]}"
SEARCH_T1=$(date +%s.%N)

echo "==> reduce"
REDUCE_T0=$(date +%s.%N)
# shellcheck disable=SC2206
OFFSETS=( ${AUNCEL_SHARD_OFFSETS} )
./reduce "${RESULTS}" "${AUNCEL_NUM_SHARDS}" "${OFFSETS[@]}"
REDUCE_T1=$(date +%s.%N)

echo "==> global recall vs ${AUNCEL_GT}"
./eval_results --gt "${AUNCEL_GT}" --result "${RESULTS}"

SEARCH_WALL=$(awk -v t0="$SEARCH_T0" -v t1="$SEARCH_T1" 'BEGIN{printf "%.3f", t1-t0}')
REDUCE_WALL=$(awk -v t0="$REDUCE_T0" -v t1="$REDUCE_T1" 'BEGIN{printf "%.3f", t1-t0}')
TOTAL_WALL=$(awk -v a="$SEARCH_WALL" -v b="$REDUCE_WALL" 'BEGIN{printf "%.3f", a+b}')
echo "Batch wall time: master_search=${SEARCH_WALL}s  reduce=${REDUCE_WALL}s  total=${TOTAL_WALL}s"

ls -la "${RESULTS}"

for pid in "${SRUN_PIDS[@]}"; do
  wait "${pid}" 2>/dev/null || true
done

echo "Done: ${RESULTS}/merged_idx.ivecs"
