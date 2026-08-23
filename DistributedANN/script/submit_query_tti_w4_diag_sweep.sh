#!/usr/bin/bash
# TTI W=4 diagnostic query sweep — decide query-tuning vs re-indexing.
#
# Three bands (24 jobs total, k=10 + k=100):
#   h-sweep   H=10,16,24,32  BW=3072  kh=200 ef=400
#   head-sweep H=32  BW=3072  (kh,ef) in {(200,400),(400,400),(400,800),(800,1600)}
#   bw-sweep  H=32  kh=400 ef=800  BW=1024,2048,3072,4096
#
# Decision rule after results:
#   recall@10 >= 0.70 or recall@100 >= 0.70 → try full W=6 sweep before re-index
#   else → fix IP replication + re-index TTI
#
# Usage:
#   ./submit_query_tti_w4_diag_sweep.sh
#   ./submit_query_tti_w4_diag_sweep.sh --dry-run

set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SBATCH_SCRIPT="${ROOT}/script/run_query.sbatch"
LOG_DIR="/gpfs/scratch/delgaudio/dann/logs/query_diag/tti_w4"

NODES="4"
WARMUP="10"
NUM_QUERIES="0"
DRY_RUN=0

INDEX_DIR="/gpfs/scratch/delgaudio/dann/index_tti10m_k32"
QUERY_PATH="/gpfs/scratch/delgaudio/vectors/TTI/yangt2iquery10K.fbin"
GT_PATH="/gpfs/scratch/delgaudio/vectors/TTI/computedGT_tti10M_10kqueries_k100_faiss.bin"
BUILD_DIR="${ROOT}/build_ip"

usage() {
  echo "Usage: $0 [--dry-run]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown: $1"; usage ;;
  esac
done

for f in "${INDEX_DIR}/nodes.bin" "${GT_PATH}" "${QUERY_PATH}" "${BUILD_DIR}/bin/run_query"; do
  [[ -e "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done
mkdir -p "${LOG_DIR}"

l_for_bw() {
  local bw="$1"
  if [[ "$bw" -le 256 ]]; then echo 400; else echo "$bw"; fi
}

submit_one() {
  local recall_k="$1" h="$2" bw="$3" k_head="$4" ef_head="$5" tag="$6"
  local l job_name log_base export_vars
  l="$(l_for_bw "$bw")"
  job_name="dann-tti-w4-diag-${tag}-h${h}-bw${bw}-kh${k_head}-ef${ef_head}-k${recall_k}"
  log_base="${LOG_DIR}/${job_name}_%j"

  export_vars="ALL,DATASET=tti,PARTITIONS=4"
  export_vars="${export_vars},BUILD_DIR=${BUILD_DIR}"
  export_vars="${export_vars},INDEX_DIR=${INDEX_DIR}"
  export_vars="${export_vars},QUERY_PATH=${QUERY_PATH},GT_PATH=${GT_PATH}"
  export_vars="${export_vars},K=${recall_k},L=${l},BW=${bw},H=${h}"
  export_vars="${export_vars},K_HEAD=${k_head},EF_HEAD=${ef_head}"
  export_vars="${export_vars},WARMUP=${WARMUP},NUM_QUERIES=${NUM_QUERIES}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] k=${recall_k} H=${h} BW=${bw} kh=${k_head} ef=${ef_head} (${tag})"
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

TOTAL=0
echo "=== TTI W=4 diagnostic sweep (index_tti10m_k32) ==="
echo "log_dir=${LOG_DIR}"
echo

echo "--- Band 1: H sweep (BW=3072, kh=200, ef=400) ---"
for h in 10 16 24 32; do
  for k in 10 100; do
    submit_one "$k" "$h" 3072 200 400 h-sweep
    TOTAL=$((TOTAL + 1))
  done
done

echo "--- Band 2: head sweep (H=32, BW=3072) ---"
for pair in "200 400" "400 400" "400 800" "800 1600"; do
  read -r kh ef <<< "$pair"
  for k in 10 100; do
    submit_one "$k" 32 3072 "$kh" "$ef" head-sweep
    TOTAL=$((TOTAL + 1))
  done
done

echo "--- Band 3: BW sweep (H=32, kh=400, ef=800) ---"
for bw in 1024 2048 3072 4096; do
  for k in 10 100; do
    submit_one "$k" 32 "$bw" 400 800 bw-sweep
    TOTAL=$((TOTAL + 1))
  done
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry-run: ${TOTAL} jobs."
else
  echo "Submitted ${TOTAL} jobs."
  echo "Grep: grep SWEEP_SUMMARY ${LOG_DIR}/dann-tti-w4-diag-*.log"
  squeue -u "$(whoami)" -n "dann-tti-w4-diag" 2>/dev/null | head -15 || true
fi
