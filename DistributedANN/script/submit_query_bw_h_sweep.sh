#!/usr/bin/bash
# Mini 2D query sweep: BW x H (one Slurm job per grid point).
#
# Default grid (tune toward higher recall):
#   BW in {256, 512}  x  H in {5, 8, 10}  => 6 jobs
#   L = max(400, BW), k_head=200, ef_head=400
#
# Usage:
#   ./submit_query_bw_h_sweep.sh --partitions 4 --k 10
#   ./submit_query_bw_h_sweep.sh --partitions 4 --k 10 --dry-run

set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SBATCH_SCRIPT="${ROOT}/script/run_query.sbatch"
LOG_DIR="/gpfs/scratch/delgaudio/dann/logs/query_bw_h_sweep"

BW_VALUES=(256 512)
H_VALUES=(5 8 10)

DATASET="deep"
PARTITIONS="4"
RECALL_K="10"
DRY_RUN=0

usage() {
  echo "Usage: $0 [--dataset deep|tti] [--partitions 4|6] [--k 10|100] [--dry-run]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)    DATASET="$2"; shift 2 ;;
    --partitions) PARTITIONS="$2"; shift 2 ;;
    --k)          RECALL_K="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

if [[ "$PARTITIONS" != "4" && "$PARTITIONS" != "6" ]]; then
  echo "FATAL: --partitions must be 4 or 6 (got ${PARTITIONS})"
  exit 1
fi
if [[ "$RECALL_K" != "10" && "$RECALL_K" != "100" ]]; then
  echo "FATAL: --k must be 10 or 100 (got ${RECALL_K})"
  exit 1
fi
if [[ "$DATASET" != "deep" && "$DATASET" != "tti" ]]; then
  echo "FATAL: --dataset must be deep or tti (got ${DATASET})"
  exit 1
fi

QUERY_PATH="/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin"
GT_PATH="/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep10M_10kqueries_k100_faiss.bin"
NODES="${PARTITIONS}"
K_HEAD="200"
EF_HEAD="400"
WARMUP="10"
NUM_QUERIES="0"

case "${DATASET}-${PARTITIONS}" in
  deep-4)
    INDEX_DIR="/gpfs/scratch/delgaudio/dann/index_deep10m_k32"
    ;;
  deep-6)
    INDEX_DIR="/gpfs/scratch/delgaudio/dann/index_deep10m_k32_w6"
    ;;
  tti-4)
    INDEX_DIR="/gpfs/scratch/delgaudio/dann/index_tti10m_k32"
    QUERY_PATH="/gpfs/scratch/delgaudio/vectors/TTI/yangt2iquery10K.fbin"
    GT_PATH="/gpfs/scratch/delgaudio/vectors/TTI/computedGT_tti10M_10kqueries_k100_faiss.bin"
    ;;
  tti-6)
    INDEX_DIR="/gpfs/scratch/delgaudio/dann/index_tti10m_k32_w6"
    QUERY_PATH="/gpfs/scratch/delgaudio/vectors/TTI/yangt2iquery10K.fbin"
    GT_PATH="/gpfs/scratch/delgaudio/vectors/TTI/computedGT_tti10M_10kqueries_k100_faiss.bin"
    ;;
  *)
    echo "FATAL: unsupported dataset/partitions combo: ${DATASET}-${PARTITIONS}"
    exit 1
    ;;
esac

if [[ ! -f "${INDEX_DIR}/nodes.bin" ]]; then
  echo "FATAL: index not found at ${INDEX_DIR}/nodes.bin"
  exit 1
fi
for f in "${GT_PATH}" "${QUERY_PATH}"; do
  [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done

mkdir -p "${LOG_DIR}"

pick_L() {
  local bw="$1"
  if (( bw > 400 )); then echo "${bw}"; else echo "400"; fi
}

echo "=== DistributedANN BW x H sweep submit ==="
echo "dataset=${DATASET}  partitions=${PARTITIONS}  recall_k=${RECALL_K}"
echo "index=${INDEX_DIR}"
echo "BW: ${BW_VALUES[*]}  H: ${H_VALUES[*]}"
echo "k_head=${K_HEAD} ef_head=${EF_HEAD}  L=max(400,BW)"
echo "jobs: $(( ${#BW_VALUES[@]} * ${#H_VALUES[@]} ))  nodes/job: ${NODES}"
echo "log_dir: ${LOG_DIR}"
echo

submit_one() {
  local bw="$1"
  local h="$2"
  local l
  l="$(pick_L "${bw}")"
  local job_name="dann-${DATASET}${PARTITIONS}p-bw${bw}-h${h}-k${RECALL_K}"
  local log_base="${LOG_DIR}/${job_name}_%j"

  local export_vars="ALL"
  export_vars="${export_vars},DATASET=${DATASET}"
  export_vars="${export_vars},PARTITIONS=${PARTITIONS}"
  export_vars="${export_vars},INDEX_DIR=${INDEX_DIR}"
  export_vars="${export_vars},QUERY_PATH=${QUERY_PATH}"
  export_vars="${export_vars},GT_PATH=${GT_PATH}"
  export_vars="${export_vars},K=${RECALL_K},L=${l},BW=${bw},H=${h}"
  export_vars="${export_vars},K_HEAD=${K_HEAD},EF_HEAD=${EF_HEAD}"
  export_vars="${export_vars},WARMUP=${WARMUP},NUM_QUERIES=${NUM_QUERIES}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ${job_name}  L=${l}  BW=${bw}  H=${h}"
    echo "  sbatch ... --export=${export_vars} ${SBATCH_SCRIPT}"
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
  for h in "${H_VALUES[@]}"; do
    submit_one "${bw}" "${h}"
  done
done

NJ=$(( ${#BW_VALUES[@]} * ${#H_VALUES[@]} ))
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "Dry-run complete (${NJ} jobs)."
else
  echo
  echo "Submitted ${NJ} jobs. Logs: ${LOG_DIR}/"
  echo "Grep: grep SWEEP_SUMMARY ${LOG_DIR}/dann-${DATASET}${PARTITIONS}p-bw*-k${RECALL_K}_*.log"
  squeue -u "$(whoami)" -n "dann-${DATASET}${PARTITIONS}p-bw" 2>/dev/null | head -10 || true
fi
