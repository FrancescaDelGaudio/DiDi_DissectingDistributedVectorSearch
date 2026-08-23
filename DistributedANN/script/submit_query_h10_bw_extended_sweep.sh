#!/usr/bin/bash
# Extended H=10 BW sweeps on index_deep10m_k32.
#
# Bands (all H=10, k_head=200, ef_head=400, L=400 if BW<=256 else L=BW):
#   k10-low   BW 8 16 32 64 128 192     — fill recall@10 gap 0.7–0.95
#   k100-low  BW 16 32 64 128 192       — push recall@100 toward 0.7
#   k100-high BW 2560 3072              — push recall@100 toward 0.99
#
# Usage:
#   ./submit_query_h10_bw_extended_sweep.sh           # submit all three bands
#   ./submit_query_h10_bw_extended_sweep.sh k10-low
#   ./submit_query_h10_bw_extended_sweep.sh k100-low
#   ./submit_query_h10_bw_extended_sweep.sh k100-high
#   ./submit_query_h10_bw_extended_sweep.sh --dry-run k10-low

set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SBATCH_SCRIPT="${ROOT}/script/run_query.sbatch"
LOG_DIR="/gpfs/scratch/delgaudio/dann/logs/query_bw_h_sweep"

H="10"
K_HEAD="200"
EF_HEAD="400"
WARMUP="10"
NUM_QUERIES="0"
NODES="4"
DRY_RUN=0

INDEX_DIR="/gpfs/scratch/delgaudio/dann/index_deep10m_k32"
QUERY_PATH="/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin"
GT_PATH="/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep10M_10kqueries_k100_faiss.bin"

usage() {
  echo "Usage: $0 [--dry-run] [k10-low|k100-low|k100-high|all]"
  exit 1
}

BANDS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    k10-low|k100-low|k100-high|all) BANDS+=("$1"); shift ;;
    -h|--help) usage ;;
    *) echo "Unknown: $1"; usage ;;
  esac
done
[[ ${#BANDS[@]} -eq 0 ]] && BANDS=(all)
[[ " ${BANDS[*]} " == *" all "* ]] && BANDS=(k10-low k100-low k100-high)

for f in "${INDEX_DIR}/nodes.bin" "${GT_PATH}" "${QUERY_PATH}"; do
  [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
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
  job_name="dann-k32-bw${bw}-h10-k${recall_k}-${band_tag}"
  log_base="${LOG_DIR}/${job_name}_%j"

  export_vars="ALL,DATASET=deep,PARTITIONS=4"
  export_vars="${export_vars},INDEX_DIR=${INDEX_DIR}"
  export_vars="${export_vars},QUERY_PATH=${QUERY_PATH},GT_PATH=${GT_PATH}"
  export_vars="${export_vars},K=${recall_k},L=${l},BW=${bw},H=${H}"
  export_vars="${export_vars},K_HEAD=${K_HEAD},EF_HEAD=${EF_HEAD}"
  export_vars="${export_vars},WARMUP=${WARMUP},NUM_QUERIES=${NUM_QUERIES}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] k=${recall_k} BW=${bw} L=${l} H=${H} (${band_tag})"
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

run_band() {
  local band="$1" recall_k bw_list
  case "$band" in
    k10-low)
      recall_k=10
      bw_list=(8 16 32 64 128 192)
      ;;
    k100-low)
      recall_k=100
      bw_list=(16 32 64 128 192)
      ;;
    k100-high)
      recall_k=100
      bw_list=(2560 3072)
      ;;
    *) echo "Unknown band: $band"; return 1 ;;
  esac
  echo "=== band=${band}  k=${recall_k}  BW: ${bw_list[*]} ==="
  for bw in "${bw_list[@]}"; do
    submit_one "${recall_k}" "${bw}" "${band}"
  done
}

TOTAL=0
for band in "${BANDS[@]}"; do
  run_band "$band"
  case "$band" in
    k10-low)  TOTAL=$((TOTAL + 6)) ;;
    k100-low) TOTAL=$((TOTAL + 5)) ;;
    k100-high) TOTAL=$((TOTAL + 2)) ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry-run: ${TOTAL} jobs."
else
  echo "Submitted ${TOTAL} jobs. Logs: ${LOG_DIR}/"
  echo "Grep: grep SWEEP_SUMMARY ${LOG_DIR}/dann-k32-bw*-h10-k*_k*.log"
  squeue -u "$(whoami)" -n "dann-k32-bw" 2>/dev/null | head -15 || true
fi
