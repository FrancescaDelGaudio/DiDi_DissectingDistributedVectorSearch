#!/usr/bin/bash
# Submit bug_mips MODE_CODE=1 (build slave indices) on TTI10M MIPS.
#
# One exclusive node per MPI rank:
#   4 shard → 6 nodes, 6 ranks
#   6 shard → 8 nodes, 8 ranks
#
# Prerequisites (per shard config):
#   - single_machine_mips finished
#   - gt_k10.lshbox / gt.lshbox present under $DATA
#   - bug_mips_{4,6}_k{10,100} built
#
# Usage:
#   ./submit_bug_mips_tti_build.sh 4
#   ./submit_bug_mips_tti_build.sh 6
#   ./submit_bug_mips_tti_build.sh all
#   ./submit_bug_mips_tti_build.sh 4 --dry-run
#   RECALL_K=10 ./submit_bug_mips_tti_build.sh all

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

RECALL_K="${RECALL_K:-100}"
MODE_CODE="${MODE_CODE:-1}"
PRINT_RESULTS="${PRINT_RESULTS:-0}"
WAKEUP_RATE="${WAKEUP_RATE:-10}"

if [[ -z "$SHARDS" ]]; then
  echo "Usage: $0 {4|6|all} [--dry-run]"
  exit 1
fi

if [[ "$RECALL_K" == "10" ]]; then
  GT="$DATA/gt_k10.lshbox"
else
  GT="$DATA/gt.lshbox"
fi

mkdir -p "$LOG_DIR"

submit_shard() {
  local shards=$1
  local out tasks nodes dhnsw job_name log_base

  if [[ "$shards" == "4" ]]; then
    out="$DATA/index_4shard_m10k_mips"
    tasks=6
    nodes=6
    dhnsw="$BUILD/bug_mips_4_k${RECALL_K}"
  elif [[ "$shards" == "6" ]]; then
    out="$DATA/index_6shard_m10k_mips"
    tasks=8
    nodes=8
    dhnsw="$BUILD/bug_mips_6_k${RECALL_K}"
  else
    echo "FATAL: unknown shard count: $shards"
    exit 1
  fi

  job_name="dhnsw-bug-mips-mc1-${shards}s-${nodes}n"
  log_base="${LOG_DIR}/${job_name}_%j"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "sbatch --exclusive --job-name=${job_name} -N ${nodes} -n ${tasks} --ntasks-per-node=1 --cpus-per-task=32 -o ${log_base}.log -e ${log_base}.log --export=ALL,DHNSW=${dhnsw},DATA=${DATA},OUT=${out},GT=${GT},NUM_SUBHNSW=${shards},RECALL_K=${RECALL_K},MODE_CODE=${MODE_CODE},WAKEUP_RATE=${WAKEUP_RATE},PRINT_RESULTS=${PRINT_RESULTS} ${SBATCH_SCRIPT}"
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
      --export=ALL,DHNSW="${dhnsw}",DATA="${DATA}",OUT="${out}",GT="${GT}",NUM_SUBHNSW="${shards}",RECALL_K="${RECALL_K}",MODE_CODE="${MODE_CODE}",WAKEUP_RATE="${WAKEUP_RATE}",PRINT_RESULTS="${PRINT_RESULTS}" \
      "${SBATCH_SCRIPT}"
  fi
}

if [[ "$SHARDS" == "all" ]]; then
  submit_shard 4
  submit_shard 6
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry-run: 2 jobs (4-shard + 6-shard)."
  else
    echo "Submitted 2 jobs (4-shard + 6-shard) with MODE_CODE=${MODE_CODE}, RECALL_K=${RECALL_K}."
  fi
else
  submit_shard "$SHARDS"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "Submitted 1 job for ${SHARDS} shards (MODE_CODE=${MODE_CODE}, RECALL_K=${RECALL_K})."
  fi
fi
