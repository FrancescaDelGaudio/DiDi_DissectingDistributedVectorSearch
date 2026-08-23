#!/usr/bin/bash
# Submit DistributedANN DEEP/TTI query sweep over beam width BW.
# One Slurm job per BW value; each job runs all 10K queries and logs recall + time.
#
# BW grid (paper-style 32..256 plus 192): 32, 64, 128, 192, 256
#
# Slurm layout: one MPI rank per physical node (W = partitions), exclusive.
#
# Usage:
#   ./submit_query_bw_sweep.sh --partitions 4 --k 10
#   ./submit_query_bw_sweep.sh --partitions 4 --k 100
#   ./submit_query_bw_sweep.sh --partitions 6 --k 10      # after 6-shard index exists
#   ./submit_query_bw_sweep.sh --dataset tti --partitions 4 --k 10   # when TTI index ready
#   ./submit_query_bw_sweep.sh --partitions 4 --k 10 --dry-run
#
# Examples (full matrix you described):
#   ./submit_query_bw_sweep.sh --partitions 4 --k 10
#   ./submit_query_bw_sweep.sh --partitions 4 --k 100
#   ./submit_query_bw_sweep.sh --partitions 6 --k 10
#   ./submit_query_bw_sweep.sh --partitions 6 --k 100

set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SBATCH_SCRIPT="${ROOT}/script/run_query.sbatch"
LOG_DIR="/gpfs/scratch/delgaudio/dann/logs/query_bw_sweep"

BW_VALUES=(32 64 128 192 256)

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

# --- Paths per dataset / shard count ----------------------------------------
QUERY_PATH="/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin"
GT_PATH="/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep10M_10kqueries_k100_faiss.bin"
NODES="${PARTITIONS}"
L="200"
H="5"
K_HEAD="100"
EF_HEAD="200"
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
  echo "       Build indexing for ${DATASET} W=${PARTITIONS} first."
  exit 1
fi
if [[ ! -f "${GT_PATH}" ]]; then
  echo "FATAL: ground truth not found: ${GT_PATH}"
  exit 1
fi
if [[ ! -f "${QUERY_PATH}" ]]; then
  echo "FATAL: query file not found: ${QUERY_PATH}"
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "=== DistributedANN BW sweep submit ==="
echo "dataset=${DATASET}  partitions=${PARTITIONS}  recall_k=${RECALL_K}"
echo "index=${INDEX_DIR}"
echo "queries=${QUERY_PATH}  gt=${GT_PATH}"
echo "BW values: ${BW_VALUES[*]}"
echo "jobs: ${#BW_VALUES[@]}  nodes/job: ${NODES}"
echo "log_dir: ${LOG_DIR}"
echo

submit_one() {
  local bw="$1"
  local job_name="dann-${DATASET}${PARTITIONS}p-bw${bw}-k${RECALL_K}"
  local log_base="${LOG_DIR}/${job_name}_%j"

  local export_vars="ALL"
  export_vars="${export_vars},DATASET=${DATASET}"
  export_vars="${export_vars},PARTITIONS=${PARTITIONS}"
  export_vars="${export_vars},INDEX_DIR=${INDEX_DIR}"
  export_vars="${export_vars},QUERY_PATH=${QUERY_PATH}"
  export_vars="${export_vars},GT_PATH=${GT_PATH}"
  export_vars="${export_vars},K=${RECALL_K},L=${L},BW=${bw},H=${H}"
  export_vars="${export_vars},K_HEAD=${K_HEAD},EF_HEAD=${EF_HEAD}"
  export_vars="${export_vars},WARMUP=${WARMUP},NUM_QUERIES=${NUM_QUERIES}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] sbatch --job-name=${job_name} -p ncpushort -N ${NODES} -n ${NODES} \\"
    echo "  --ntasks-per-node=1 --cpus-per-task=8 --exclusive \\"
    echo "  -o ${log_base}.log -e ${log_base}.log \\"
    echo "  --export=${export_vars} ${SBATCH_SCRIPT}"
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
  echo
  echo "Dry-run complete (${#BW_VALUES[@]} jobs)."
else
  echo
  echo "Submitted ${#BW_VALUES[@]} jobs. Logs: ${LOG_DIR}/"
  echo "Grep results: grep SWEEP_SUMMARY ${LOG_DIR}/dann-${DATASET}${PARTITIONS}p-bw*-k${RECALL_K}_*.log"
  squeue -u "$(whoami)" -n "dann-${DATASET}${PARTITIONS}p-bw" 2>/dev/null | head -8 || true
fi
