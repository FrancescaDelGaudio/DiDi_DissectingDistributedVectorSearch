#!/usr/bin/bash
# Submit TTI10M MIPS query sweep (high K): wakeup_rate K in {50,70,80,100} x recall@k in {10,100}.
# 8 jobs per shard config (4 or 6); 16 jobs if SHARDS=all.
#
# Same layout as submit_bug_mips_tti_sweep.sh; job/log names use suffix -hiK.
#
# Usage:
#   ./submit_bug_mips_tti_sweep_hiK.sh 4
#   ./submit_bug_mips_tti_sweep_hiK.sh 6
#   ./submit_bug_mips_tti_sweep_hiK.sh all
#   ./submit_bug_mips_tti_sweep_hiK.sh 4 --dry-run

set -eo pipefail

SBATCH_SCRIPT="/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/script/sbatch_bug_mips_tti.sh"
BUILD="/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/build"
DATA="/gpfs/scratch/delgaudio/vectors/TTI"
LOG_DIR="/gpfs/scratch/delgaudio/dhnsw/logs"

SHARDS="${1:-}"
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" || "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  [[ "${1:-}" == "--dry-run" ]] && SHARDS="${2:-all}"
fi

if [[ -z "$SHARDS" ]]; then
  echo "Usage: $0 {4|6|all} [--dry-run]"
  exit 1
fi

WAKEUP_K_VALUES=(50 70 80 100)
RECALL_K_VALUES=(10 100)

mkdir -p "$LOG_DIR"

submit_shard() {
  local shards=$1
  local out tasks nodes binary_prefix

  if [[ "$shards" == "4" ]]; then
    out="$DATA/index_4shard_m10k_mips"
    tasks=6
    nodes=6
    binary_prefix="bug_mips_4_k"
  elif [[ "$shards" == "6" ]]; then
    out="$DATA/index_6shard_m10k_mips"
    tasks=8
    nodes=8
    binary_prefix="bug_mips_6_k"
  else
    echo "FATAL: unknown shard count: $shards"
    exit 1
  fi

  for recall_k in "${RECALL_K_VALUES[@]}"; do
    for wakeup_k in "${WAKEUP_K_VALUES[@]}"; do
      local dhnsw gt job_name log_base
      dhnsw="$BUILD/${binary_prefix}${recall_k}"
      if [[ "$recall_k" == "10" ]]; then
        gt="$DATA/gt_k10.lshbox"
      else
        gt="$DATA/gt.lshbox"
      fi
      job_name="dhnsw-mips-${shards}s-K${wakeup_k}-k${recall_k}-hiK"
      log_base="${LOG_DIR}/${job_name}_%j"

      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "sbatch --exclusive --job-name=${job_name} -N ${nodes} -n ${tasks} --ntasks-per-node=1 --cpus-per-task=32 -o ${log_base}.log -e ${log_base}.log --export=ALL,DHNSW=${dhnsw},DATA=${DATA},OUT=${out},GT=${gt},NUM_SUBHNSW=${shards},WAKEUP_RATE=${wakeup_k},RECALL_K=${recall_k},MODE_CODE=0,PRINT_RESULTS=0 ${SBATCH_SCRIPT}"
      else
        sbatch \
          --exclusive \
          --job-name="${job_name}" \
          -N "${nodes}" \
          -n "${tasks}" \
          --ntasks-per-node=1 \
          --cpus-per-task=32 \
          -o "${log_base}.log" \
          -e "${log_base}.log" \
          --export=ALL,DHNSW="${dhnsw}",DATA="${DATA}",OUT="${out}",GT="${gt}",NUM_SUBHNSW="${shards}",WAKEUP_RATE="${wakeup_k}",RECALL_K="${recall_k}",MODE_CODE=0,PRINT_RESULTS=0 \
          "${SBATCH_SCRIPT}"
      fi
    done
  done
}

if [[ "$SHARDS" == "all" ]]; then
  submit_shard 4
  submit_shard 6
  echo "Done. Submitted $(( 2 * ${#WAKEUP_K_VALUES[@]} * ${#RECALL_K_VALUES[@]} )) jobs (K in {50,70,80,100})."
else
  submit_shard "$SHARDS"
  echo "Done. Submitted $(( ${#WAKEUP_K_VALUES[@]} * ${#RECALL_K_VALUES[@]} )) jobs (K in {50,70,80,100})."
fi
