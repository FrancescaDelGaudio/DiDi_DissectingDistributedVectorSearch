#!/usr/bin/bash
# Submit Pyramid 4-node query sweep (K x sender_ef) for DEEP or TTI, recall@k = 10 or 100.
#
# Layout: 4 physical nodes, 6 MPI ranks (sender/receiver colocated on slaves 0 and 1).
# Grid: ef in {50,100,200} x K in {1,10,20,50,70,80,100} → 21 jobs per invocation.
#
# Usage:
#   ./submit_4node_ef_k_sweep.sh deep 10
#   ./submit_4node_ef_k_sweep.sh deep 100
#   ./submit_4node_ef_k_sweep.sh tti 10
#   ./submit_4node_ef_k_sweep.sh tti 100
#   ./submit_4node_ef_k_sweep.sh deep 10 --dry-run

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/build"
LOG_DIR="/gpfs/scratch/delgaudio/dhnsw/logs"

DATASET="${1:-}"
RECALL_K="${2:-}"
DRY_RUN=0
if [[ "${3:-}" == "--dry-run" || "${2:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  [[ "${2:-}" == "--dry-run" ]] && RECALL_K="${3:-}"
fi

if [[ -z "$DATASET" || -z "$RECALL_K" ]]; then
  echo "Usage: $0 {deep|tti} {10|100} [--dry-run]"
  exit 1
fi

if [[ "$RECALL_K" != "10" && "$RECALL_K" != "100" ]]; then
  echo "FATAL: RECALL_K must be 10 or 100 (got: $RECALL_K)"
  exit 1
fi

NUM_SUBHNSW=4
NODES=4
TASKS=6
SENDER_EF_VALUES=(50 100 200)
WAKEUP_K_VALUES=(1 10 20 50 70 80 100)

case "$DATASET" in
  deep)
    SBATCH_SCRIPT="$SCRIPT_DIR/sbatch_bug_deep_4node_sweep.sh"
    DATA="/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid"
    OUT="$DATA/index_4shard_m10k"
    DHNSW="$BUILD/bug_4_k${RECALL_K}"
    CPUS_PER_TASK=8
    JOB_PREFIX="dhnsw-4n-deep"
    ;;
  tti)
    SBATCH_SCRIPT="$SCRIPT_DIR/sbatch_bug_mips_4node_sweep.sh"
    DATA="/gpfs/scratch/delgaudio/vectors/TTI"
    OUT="$DATA/index_4shard_m10k_mips"
    DHNSW="$BUILD/bug_mips_4_k${RECALL_K}"
    CPUS_PER_TASK=32
    JOB_PREFIX="dhnsw-4n-tti"
    ;;
  *)
    echo "FATAL: DATASET must be 'deep' or 'tti' (got: $DATASET)"
    exit 1
    ;;
esac

if [[ "$RECALL_K" == "10" ]]; then
  GT="$DATA/gt_k10.lshbox"
else
  GT="$DATA/gt.lshbox"
fi

mkdir -p "$LOG_DIR"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "FATAL: missing $SBATCH_SCRIPT"
  exit 1
fi

job_count=0

for sender_ef in "${SENDER_EF_VALUES[@]}"; do
  for wakeup_k in "${WAKEUP_K_VALUES[@]}"; do
    job_name="${JOB_PREFIX}-K${wakeup_k}-ef${sender_ef}-k${RECALL_K}"
    log_base="${LOG_DIR}/${job_name}_%j"

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "sbatch --exclusive --job-name=${job_name} -N ${NODES} -n ${TASKS} --cpus-per-task=${CPUS_PER_TASK} -o ${log_base}.log -e ${log_base}.log --export=ALL,DHNSW=${DHNSW},DATA=${DATA},OUT=${OUT},GT=${GT},NUM_SUBHNSW=${NUM_SUBHNSW},WAKEUP_RATE=${wakeup_k},RECALL_K=${RECALL_K},MODE_CODE=0,SENDER_EF=${sender_ef},SLAVE_EF=${sender_ef},PRINT_RESULTS=0 ${SBATCH_SCRIPT}"
    else
      sbatch \
        --exclusive \
        --job-name="${job_name}" \
        -N "${NODES}" \
        -n "${TASKS}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        -o "${log_base}.log" \
        -e "${log_base}.log" \
        --export=ALL,DHNSW="${DHNSW}",DATA="${DATA}",OUT="${OUT}",GT="${GT}",NUM_SUBHNSW="${NUM_SUBHNSW}",WAKEUP_RATE="${wakeup_k}",RECALL_K="${RECALL_K}",MODE_CODE=0,SENDER_EF="${sender_ef}",SLAVE_EF="${sender_ef}",PRINT_RESULTS=0 \
        "${SBATCH_SCRIPT}"
    fi
    job_count=$((job_count + 1))
  done
done

echo "Done. ${job_count} jobs for ${DATASET} recall@k=${RECALL_K} (ef x K grid, 4 nodes)."
