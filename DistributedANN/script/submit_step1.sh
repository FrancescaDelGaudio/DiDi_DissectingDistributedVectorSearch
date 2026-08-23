#!/usr/bin/bash
# Submit DistributedANN Step 1 (k-means + epsilon-replication).
#
# Usage:
#   ./submit_step1.sh [nodes] [--dry-run]
#
# Examples:
#   ./submit_step1.sh 4
#   ./submit_step1.sh 6 --dry-run
#   K=64 EPSILON=0.3 ./submit_step1.sh 4

set -eo pipefail

SBATCH_SCRIPT="$(dirname "$0")/build_step1.sbatch"

NODES="${1:-4}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" || "${1:-}" == "--dry-run" ]] && DRY_RUN=1
[[ "${1:-}" == "--dry-run" ]] && NODES=4

LOG_DIR="/gpfs/scratch/delgaudio/dann/logs"
mkdir -p "${LOG_DIR}"

K="${K:-32}"
EPSILON="${EPSILON:-0.2}"
MAX_REPLICAS="${MAX_REPLICAS:-3}"
KMEANS_ITER="${KMEANS_ITER:-20}"
BASE_PATH="${BASE_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid/base.fvecs}"
INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_deep10m_k${K}}"

JOB_NAME="dann-step1-${NODES}n-k${K}"
LOG_BASE="${LOG_DIR}/${JOB_NAME}_%j"

EXPORT_VARS="ALL,K=${K},EPSILON=${EPSILON},MAX_REPLICAS=${MAX_REPLICAS}"
EXPORT_VARS="${EXPORT_VARS},KMEANS_ITER=${KMEANS_ITER}"
EXPORT_VARS="${EXPORT_VARS},BASE_PATH=${BASE_PATH}"
EXPORT_VARS="${EXPORT_VARS},INDEX_DIR=${INDEX_DIR}"

CMD="sbatch
  --job-name=${JOB_NAME}
  -N ${NODES}
  -n ${NODES}
  --ntasks-per-node=1
  --cpus-per-task=8
  --exclusive
  -o ${LOG_BASE}.log
  -e ${LOG_BASE}.log
  --export=${EXPORT_VARS}
  ${SBATCH_SCRIPT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] $CMD"
else
    # shellcheck disable=SC2086
    sbatch \
        --job-name="${JOB_NAME}" \
        -N "${NODES}" \
        -n "${NODES}" \
        --ntasks-per-node=1 \
        --cpus-per-task=8 \
        --exclusive \
        -o "${LOG_BASE}.log" \
        -e "${LOG_BASE}.log" \
        --export="${EXPORT_VARS}" \
        "${SBATCH_SCRIPT}"
    echo "Submitted step 1 job (${NODES} nodes, K=${K}, epsilon=${EPSILON})."
fi
