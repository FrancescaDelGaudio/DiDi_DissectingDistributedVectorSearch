#!/usr/bin/bash
# Full H=10 BW sweep on index_tti10m_k32_w6 (MIPS / build_ip, 6 physical nodes).
# Same grid as W=4 TTI recall curves:
#   k=10  — BW 8..2048  (12 jobs)
#   k=100 — BW 16..3072 (13 jobs)
#
# Usage:
#   ./submit_query_tti_w6_h10_bw_sweep.sh
#   ./submit_query_tti_w6_h10_bw_sweep.sh --k 10
#   ./submit_query_tti_w6_h10_bw_sweep.sh --dry-run

set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SBATCH_SCRIPT="${ROOT}/script/run_query.sbatch"
LOG_DIR="/gpfs/scratch/delgaudio/dann/logs/query_bw_h_sweep/tti_w6"

H="10"
K_HEAD="200"
EF_HEAD="400"
WARMUP="10"
NUM_QUERIES="0"
NODES="6"
DRY_RUN=0
RECALL_K="all"

INDEX_DIR="/gpfs/scratch/delgaudio/dann/index_tti10m_k32_w6"
QUERY_PATH="/gpfs/scratch/delgaudio/vectors/TTI/yangt2iquery10K.fbin"
GT_PATH="/gpfs/scratch/delgaudio/vectors/TTI/computedGT_tti10M_10kqueries_k100_faiss.bin"
BUILD_DIR="${ROOT}/build_ip"

usage() {
  echo "Usage: $0 [--k 10|100|all] [--dry-run]"
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

if [[ "$RECALL_K" != "10" && "$RECALL_K" != "100" && "$RECALL_K" != "all" ]]; then
  echo "FATAL: --k must be 10, 100, or all"
  exit 1
fi

for f in "${INDEX_DIR}/nodes.bin" "${GT_PATH}" "${QUERY_PATH}" "${BUILD_DIR}/bin/run_query"; do
  [[ -e "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done
mkdir -p "${LOG_DIR}"

l_for_bw() {
  local bw="$1"
  if [[ "$bw" -le 256 ]]; then echo 400; else echo "$bw"; fi
}

submit_one() {
  local recall_k="$1" bw="$2" band_tag="$3"
  local l job_name log_base export_vars
  l="$(l_for_bw "$bw")"
  job_name="dann-tti-w6-k32-bw${bw}-h10-k${recall_k}"
  [[ -n "$band_tag" ]] && job_name="${job_name}-${band_tag}"
  log_base="${LOG_DIR}/${job_name}_%j"

  export_vars="ALL"
  export_vars="${export_vars},DATASET=tti,PARTITIONS=6"
  export_vars="${export_vars},BUILD_DIR=${BUILD_DIR}"
  export_vars="${export_vars},INDEX_DIR=${INDEX_DIR}"
  export_vars="${export_vars},QUERY_PATH=${QUERY_PATH},GT_PATH=${GT_PATH}"
  export_vars="${export_vars},K=${recall_k},L=${l},BW=${bw},H=${H}"
  export_vars="${export_vars},K_HEAD=${K_HEAD},EF_HEAD=${EF_HEAD}"
  export_vars="${export_vars},WARMUP=${WARMUP},NUM_QUERIES=${NUM_QUERIES}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] k=${recall_k} BW=${bw} L=${l} H=${H} (${band_tag:-main})"
    return
  fi

  sbatch \
    --job-name="${job_name}" \
    -p ncpushort \
    -N "${NODES}" -n "${NODES}" \
    --ntasks-per-node=1 --cpus-per-task=8 --exclusive \
    -o "${log_base}.log" -e "${log_base}.log" \
    --export="${export_vars}" \
    "${SBATCH_SCRIPT}"
}

run_k10() {
  echo "=== TTI W=6 k=10: low BW ==="
  for bw in 8 16 32 64 128 192; do submit_one 10 "$bw" k10-low; done
  echo "=== TTI W=6 k=10: high BW ==="
  for bw in 256 512 768 1024 1536 2048; do submit_one 10 "$bw" k10-high; done
}

run_k100() {
  echo "=== TTI W=6 k=100: low BW ==="
  for bw in 16 32 64 128 192; do submit_one 100 "$bw" k100-low; done
  echo "=== TTI W=6 k=100: mid/high BW ==="
  for bw in 256 512 768 1024 1536 2048; do submit_one 100 "$bw" k100-mid; done
  echo "=== TTI W=6 k=100: extra high BW (toward 0.99) ==="
  for bw in 2560 3072; do submit_one 100 "$bw" k100-high; done
}

echo "=== TTI W=6 H=10 BW sweep (index_tti10m_k32_w6, MIPS) ==="
echo "recall_k=${RECALL_K}  H=${H}  k_head=${K_HEAD} ef_head=${EF_HEAD}"
echo "INDEX_DIR=${INDEX_DIR}"
echo "log_dir=${LOG_DIR}"
echo

TOTAL=0
case "$RECALL_K" in
  10)  run_k10;  TOTAL=12 ;;
  100) run_k100; TOTAL=13 ;;
  all) run_k10; run_k100; TOTAL=25 ;;
esac

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry-run: ${TOTAL} jobs."
else
  echo "Submitted ${TOTAL} jobs."
  echo "Grep: grep SWEEP_SUMMARY ${LOG_DIR}/dann-tti-w6-k32-bw*-h10-k*.log"
  squeue -u "$(whoami)" -n "dann-tti-w6-k32-bw" 2>/dev/null | head -20 || true
fi
