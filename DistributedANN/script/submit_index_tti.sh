#!/usr/bin/bash
# ===========================================================================
# Submit full TTI 10M indexing with MIPS (inner product, unit-normalised vectors).
# Uses build_ip/ so L2 query binaries in build/ stay untouched.
#
# Usage:
#   ./submit_index_tti.sh
#   K=32 INDEX_DIR=/path/to/idx sbatch --export=ALL script/build_all.sbatch
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${SCRIPT_DIR}/.."
BUILD_IP="${REPO}/build_ip"

export BASE_PATH="${BASE_PATH:-/gpfs/scratch/delgaudio/vectors/TTI/base.fvecs}"
export INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_tti10m_k${K:-32}_topr}"
export METRIC="${METRIC:-ip}"
export METHOD="${METHOD:-random}"
export BUILD_DIR="${BUILD_DIR:-${BUILD_IP}}"
export K="${K:-32}"
export EPSILON="${EPSILON:-0.2}"
export MAX_REPLICAS="${MAX_REPLICAS:-3}"
export REPLICA_R="${REPLICA_R:-300}"
export R="${R:-72}"
export L="${L:-100}"
export ALPHA="${ALPHA:-1.2}"
export HEAD_FRAC="${HEAD_FRAC:-0.05}"
export HNSW_M="${HNSW_M:-16}"
export EFC="${EFC:-200}"
export M="${M:-16}"
export TRAIN_SIZE="${TRAIN_SIZE:-100000}"
export ITERS="${ITERS:-25}"

echo "### TTI MIPS indexing"
echo "###   BASE_PATH=${BASE_PATH}"
echo "###   INDEX_DIR=${INDEX_DIR}"
echo "###   K=${K} R=${R} eps=${EPSILON} replica_r=${REPLICA_R} METHOD=${METHOD}"
echo "###   BUILD_DIR=${BUILD_DIR}"

if [ ! -x "${BUILD_DIR}/bin/build_index" ]; then
    echo "### Configuring build_ip..."
    module purge 2>/dev/null || true
    module load gcc/10.3.0 openmpi/gcc-10.3.0/4.1.1 2>/dev/null || true
    cmake -S "${REPO}" -B "${BUILD_IP}" -DCMAKE_BUILD_TYPE=Release
    (cd "${BUILD_IP}" && make -j "$(nproc 2>/dev/null || echo 8)")
fi

JOB_ID="$(sbatch --export=ALL "${SCRIPT_DIR}/build_all.sbatch" | awk '{print $NF}')"
echo "### Submitted job ${JOB_ID}"
squeue -j "${JOB_ID}" 2>/dev/null || true
