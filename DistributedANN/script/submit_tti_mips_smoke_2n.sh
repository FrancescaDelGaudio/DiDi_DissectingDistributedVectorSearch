#!/usr/bin/bash
# Submit TTI MIPS smoke test (1000 vectors, 100 queries) on 2 Slurm nodes.
#
# Usage:
#   ./script/submit_tti_mips_smoke_2n.sh
#   SKIP_INDEX=1 ./script/submit_tti_mips_smoke_2n.sh   # query-only if index exists
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="/gpfs/scratch/delgaudio/dann/logs/tti_smoke_2n"
mkdir -p "${LOG_DIR}"

DATA_DIR="${DATA_DIR:-/gpfs/scratch/delgaudio/dann/tti_smoke_1k}"
if [ ! -f "${DATA_DIR}/base_1k.fvecs" ]; then
    echo "Preparing smoke data in ${DATA_DIR} ..."
    python3 "${ROOT}/script/prepare_tti_smoke.py" --out-dir "${DATA_DIR}"
fi

BUILD_IP="${ROOT}/build_ip"
if [ ! -x "${BUILD_IP}/bin/build_index" ]; then
    echo "ERROR: build_ip binaries missing. Run:"
    echo "  module load gcc/10.3.0 openmpi/gcc-10.3.0/4.1.1"
    echo "  cd ${BUILD_IP} && cmake .. && make -j"
    exit 1
fi

echo "### Submitting TTI MIPS smoke (2 nodes, W=2, METRIC=ip)"
echo "### DATA_DIR  = ${DATA_DIR}"
echo "### INDEX_DIR = ${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_tti_smoke_1k_2n}"
echo "### Logs      = ${LOG_DIR}/"

JOB_ID="$(
    sbatch --export=ALL \
        "${ROOT}/script/tti_mips_smoke_2n.sbatch" \
        | awk '{print $NF}'
)"

echo "### Submitted job ${JOB_ID}"
squeue -j "${JOB_ID}" 2>/dev/null || true
