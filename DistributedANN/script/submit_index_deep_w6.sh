#!/usr/bin/bash
# Full DEEP 10M indexing on 6 physical nodes (W=6).
# Same params as index_deep10m_k32 (K=32, random head, L2).
#
# Usage:
#   ./script/submit_index_deep_w6.sh
#   K=32 ./script/submit_index_deep_w6.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}"

export BASE_PATH="${BASE_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid/base.fvecs}"
export INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_deep10m_k32_w6}"
export BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
export K="${K:-32}"
export EPSILON="${EPSILON:-0.2}"
export MAX_REPLICAS="${MAX_REPLICAS:-3}"
export KMEANS_ITER="${KMEANS_ITER:-20}"
export KMEANS_TOL="${KMEANS_TOL:-1e-4}"
export R="${R:-72}"
export L="${L:-100}"
export ALPHA="${ALPHA:-1.2}"
export PASSES="${PASSES:-2}"
export M="${M:-16}"
export TRAIN_SIZE="${TRAIN_SIZE:-100000}"
export ITERS="${ITERS:-25}"
export HEAD_FRAC="${HEAD_FRAC:-0.05}"
export HNSW_M="${HNSW_M:-16}"
export EFC="${EFC:-200}"
export METHOD="${METHOD:-random}"
export METRIC="${METRIC:-l2}"

echo "### DEEP 10M indexing — 6 nodes (W=6)"
echo "### INDEX_DIR=${INDEX_DIR}"
echo "### K=${K} METHOD=${METHOD} METRIC=${METRIC} BUILD_DIR=${BUILD_DIR}"

JOB_ID="$(
  sbatch \
    --job-name=dann-index-w6 \
    -N 6 \
    -n 6 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --exclusive \
    -o "${LOG_DIR}/dann-index-w6_%j.log" \
    -e "${LOG_DIR}/dann-index-w6_%j.log" \
    --export=ALL \
    "${ROOT}/script/build_all.sbatch" \
    | awk '{print $NF}'
)"

echo "### Submitted job ${JOB_ID}"
squeue -j "${JOB_ID}" 2>/dev/null || true
