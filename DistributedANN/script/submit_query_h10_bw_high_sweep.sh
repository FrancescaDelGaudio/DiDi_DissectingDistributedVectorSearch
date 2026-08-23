#!/usr/bin/bash
# H=10 BW sweep on index_deep10m_k32 (same grid as recall@10 table).
# One Slurm job per BW: 256, 512, 768, 1024, 1536, 2048.
#
# Params match prior k=10 runs:
#   H=10, k_head=200, ef_head=400
#   L=400 at BW=256; L=BW for BW>=512
#
# Usage:
#   ./submit_query_h10_bw_high_sweep.sh --k 100
#   ./submit_query_h10_bw_high_sweep.sh --k 10
#   ./submit_query_h10_bw_high_sweep.sh --k 100 --dry-run

set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SBATCH_SCRIPT="${ROOT}/script/run_query.sbatch"
LOG_DIR="/gpfs/scratch/delgaudio/dann/logs/query_bw_h_sweep"

BW_VALUES=(256 512 768 1024 1536 2048)
H="10"
K_HEAD="200"
EF_HEAD="400"
WARMUP="10"
NUM_QUERIES="0"

DATASET="deep"
PARTITIONS="4"
RECALL_K="100"
DRY_RUN=0

usage() {
  echo "Usage: $0 [--k 10|100] [--dry-run]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --k)       RECALL_K="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

if [[ "$RECALL_K" != "10" && "$RECALL_K" != "100" ]]; then
  echo "FATAL: --k must be 10 or 100"
  exit 1
fi

INDEX_DIR="/gpfs/scratch/delgaudio/dann/index_deep10m_k32"
QUERY_PATH="/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin"
GT_PATH="/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep10M_10kqueries_k100_faiss.bin"
NODES="4"

for f in "${INDEX_DIR}/nodes.bin" "${GT_PATH}" "${QUERY_PATH}"; do
  [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done

mkdir -p "${LOG_DIR}"

l_for_bw() {
  local bw="$1"
  if [[ "$bw" -le 256 ]]; then
    echo 400
  else
    echo "$bw"
  fi
}

echo "=== H=10 high-BW sweep (index_deep10m_k32) ==="
echo "recall_k=${RECALL_K}  H=${H}  k_head=${K_HEAD} ef_head=${EF_HEAD}"
echo "BW: ${BW_VALUES[*]}"
echo "jobs: ${#BW_VALUES[@]}  nodes/job: ${NODES}"
echo "log_dir: ${LOG_DIR}"
echo

submit_one() {
  local bw="$1"
  local l
  l="$(l_for_bw "$bw")"
  local job_name="dann-k32-bw${bw}-h10-k${RECALL_K}"
  local log_base="${LOG_DIR}/${job_name}_%j"

  local export_vars="ALL"
  export_vars="${export_vars},DATASET=${DATASET},PARTITIONS=${PARTITIONS}"
  export_vars="${export_vars},INDEX_DIR=${INDEX_DIR}"
  export_vars="${export_vars},QUERY_PATH=${QUERY_PATH},GT_PATH=${GT_PATH}"
  export_vars="${export_vars},K=${RECALL_K},L=${l},BW=${bw},H=${H}"
  export_vars="${export_vars},K_HEAD=${K_HEAD},EF_HEAD=${EF_HEAD}"
  export_vars="${export_vars},WARMUP=${WARMUP},NUM_QUERIES=${NUM_QUERIES}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ${job_name}  L=${l}  BW=${bw}  H=${H}  k=${RECALL_K}"
    return
  fi

  sbatch \
    --job-name="${job_name}" \
    -p ncpushort \
    -N "${NODES}" \
    -n "${NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --exclusive \
    -o "${log_base}.log" \
    -e "${log_base}.log" \
    --export="${export_vars}" \
    "${SBATCH_SCRIPT}"
}

for bw in "${BW_VALUES[@]}"; do
  submit_one "${bw}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry-run complete."
else
  echo "Submitted ${#BW_VALUES[@]} jobs."
  echo "Grep: grep SWEEP_SUMMARY ${LOG_DIR}/dann-k32-bw*-h10-k${RECALL_K}_*.log"
  squeue -u "$(whoami)" -n "dann-k32-bw" 2>/dev/null | head -10 || true
fi
