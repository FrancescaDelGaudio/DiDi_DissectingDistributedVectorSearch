#!/usr/bin/bash
# Full TTI 10M MIPS indexing on 6 physical nodes (W=6).
# Same params as index_tti10m_k32 / index_deep10m_k32_w6.
#
# Usage:
#   ./script/submit_index_tti_w6.sh
#   K=32 ./script/submit_index_tti_w6.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_IP="${ROOT}/build_ip"
LOG_DIR="${ROOT}"

export BASE_PATH="${BASE_PATH:-/gpfs/scratch/delgaudio/vectors/TTI/base.fvecs}"
export INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_tti10m_k32_w6_topr}"
export BUILD_DIR="${BUILD_DIR:-${BUILD_IP}}"
export METRIC="${METRIC:-ip}"
export METHOD="${METHOD:-random}"
export K="${K:-32}"
export EPSILON="${EPSILON:-0.2}"
export MAX_REPLICAS="${MAX_REPLICAS:-3}"
export REPLICA_R="${REPLICA_R:-300}"
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

echo "### TTI 10M MIPS indexing — 6 nodes (W=6)"
echo "### INDEX_DIR=${INDEX_DIR}"
echo "### K=${K} METHOD=${METHOD} METRIC=${METRIC} replica_r=${REPLICA_R} BUILD_DIR=${BUILD_DIR}"

if [ ! -x "${BUILD_DIR}/bin/build_index" ]; then
    echo "### Configuring build_ip..."
    module purge 2>/dev/null || true
    module load gcc/10.3.0 openmpi/gcc-10.3.0/4.1.1 2>/dev/null || true
    cmake -S "${ROOT}" -B "${BUILD_IP}" -DCMAKE_BUILD_TYPE=Release
    (cd "${BUILD_IP}" && make -j "$(nproc 2>/dev/null || echo 8)")
fi

JOB_ID="$(
  sbatch \
    --job-name=dann-index-tti-w6 \
    -N 6 \
    -n 6 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --exclusive \
    -o "${LOG_DIR}/dann-index-tti-w6_%j.log" \
    -e "${LOG_DIR}/dann-index-tti-w6_%j.log" \
    --export=ALL \
    "${ROOT}/script/build_all.sbatch" \
    | awk '{print $NF}'
)"

echo "### Submitted job ${JOB_ID}"
squeue -j "${JOB_ID}" 2>/dev/null || true
