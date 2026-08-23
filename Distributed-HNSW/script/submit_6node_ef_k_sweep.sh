#!/usr/bin/bash
# Submit Pyramid 6-node query sweep (K x sender_ef) for DEEP or TTI, recall@k = 10 or 100.
#
# Layout: 6 physical nodes, 8 MPI ranks (6 slaves + sender/receiver on slaves 0 and 1).
# Grid: ef in {50,100,200} x K in {1,10,20,50,70,80,100} → 21 jobs per invocation.
#
# Usage:
#   ./submit_6node_ef_k_sweep.sh deep 10
#   ./submit_6node_ef_k_sweep.sh deep 100
#   ./submit_6node_ef_k_sweep.sh tti 10
#   ./submit_6node_ef_k_sweep.sh tti 100
#   ./submit_6node_ef_k_sweep.sh all          # all 4 campaigns (84 jobs)
#   ./submit_6node_ef_k_sweep.sh deep 10 --dry-run

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

if [[ -z "$DATASET" ]]; then
  echo "Usage: $0 {deep|tti|all} {10|100} [--dry-run]"
  echo "       $0 all   # submits deep 10/100 + tti 10/100"
  exit 1
fi

SENDER_EF_VALUES=(50 100 200)
WAKEUP_K_VALUES=(1 10 20 50 70 80 100)

submit_one() {
  local dataset=$1 recall_k=$2
  local NUM_SUBHNSW=6 NODES=6 TASKS=8
  local SBATCH_SCRIPT DATA OUT DHNSW CPUS_PER_TASK JOB_PREFIX GT

  case "$dataset" in
    deep)
      SBATCH_SCRIPT="$SCRIPT_DIR/sbatch_bug_deep_6node_sweep.sh"
      DATA="/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid"
      OUT="$DATA/index_6shard_m10k"
      DHNSW="$BUILD/bug_6_k${recall_k}"
      CPUS_PER_TASK=8
      JOB_PREFIX="dhnsw-6n-deep"
      ;;
    tti)
      SBATCH_SCRIPT="$SCRIPT_DIR/sbatch_bug_mips_6node_sweep.sh"
      DATA="/gpfs/scratch/delgaudio/vectors/TTI"
      OUT="$DATA/index_6shard_m10k_mips"
      DHNSW="$BUILD/bug_mips_6_k${recall_k}"
      CPUS_PER_TASK=32
      JOB_PREFIX="dhnsw-6n-tti"
      ;;
    *)
      echo "FATAL: unknown dataset: $dataset"
      exit 1
      ;;
  esac

  if [[ "$recall_k" == "10" ]]; then
    GT="$DATA/gt_k10.lshbox"
  else
    GT="$DATA/gt.lshbox"
  fi

  if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "FATAL: missing $SBATCH_SCRIPT"
    exit 1
  fi

  local job_count=0
  for sender_ef in "${SENDER_EF_VALUES[@]}"; do
    for wakeup_k in "${WAKEUP_K_VALUES[@]}"; do
      local job_name="${JOB_PREFIX}-K${wakeup_k}-ef${sender_ef}-k${recall_k}"
      local log_base="${LOG_DIR}/${job_name}_%j"

      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "sbatch --exclusive --job-name=${job_name} -N ${NODES} -n ${TASKS} --cpus-per-task=${CPUS_PER_TASK} -o ${log_base}.log -e ${log_base}.log --export=ALL,DHNSW=${DHNSW},DATA=${DATA},OUT=${OUT},GT=${GT},NUM_SUBHNSW=${NUM_SUBHNSW},WAKEUP_RATE=${wakeup_k},RECALL_K=${recall_k},MODE_CODE=0,SENDER_EF=${sender_ef},SLAVE_EF=${sender_ef},PRINT_RESULTS=0 ${SBATCH_SCRIPT}"
      else
        sbatch \
          --exclusive \
          --job-name="${job_name}" \
          -N "${NODES}" \
          -n "${TASKS}" \
          --cpus-per-task="${CPUS_PER_TASK}" \
          -o "${log_base}.log" \
          -e "${log_base}.log" \
          --export=ALL,DHNSW="${DHNSW}",DATA="${DATA}",OUT="${OUT}",GT="${GT}",NUM_SUBHNSW="${NUM_SUBHNSW}",WAKEUP_RATE="${wakeup_k}",RECALL_K="${recall_k}",MODE_CODE=0,SENDER_EF="${sender_ef}",SLAVE_EF="${sender_ef}",PRINT_RESULTS=0 \
          "${SBATCH_SCRIPT}"
      fi
      job_count=$((job_count + 1))
    done
  done
  echo "Done. ${job_count} jobs for ${dataset} recall@k=${recall_k} (6 nodes, ef x K grid)."
}

mkdir -p "$LOG_DIR"

if [[ "$DATASET" == "all" ]]; then
  for ds in deep tti; do
    for rk in 10 100; do
      submit_one "$ds" "$rk"
    done
  done
  echo "Submitted all 4 campaigns (84 jobs total, or dry-run lines above)."
else
  if [[ -z "$RECALL_K" ]]; then
    echo "Usage: $0 {deep|tti} {10|100} [--dry-run]"
    exit 1
  fi
  if [[ "$RECALL_K" != "10" && "$RECALL_K" != "100" ]]; then
    echo "FATAL: RECALL_K must be 10 or 100 (got: $RECALL_K)"
    exit 1
  fi
  submit_one "$DATASET" "$RECALL_K"
fi
