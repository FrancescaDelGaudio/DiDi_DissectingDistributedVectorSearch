#!/usr/bin/env bash
# Launch one Auncel worker (prebuilt bundle or online) and publish IPv4 for master.
set -euo pipefail

SHARD="${1:?shard id (0 or 1)}"
COORD_DIR="${2:?coord dir on GPFS}"
DIST="${3:?path to dist/}"
LOAD_PREFIX="${4:?bundle prefix (dir/shard)}"
QUERY="${5:?query fbin}"
GTI="${6:?gti.ivecs}"
GTD="${7:?gtd.fvecs}"
REQUIRED_RECALL="${8:-0.9}"
HYPERPARAM_ID="${9:-1}"

module load intel/20U4/mkl 2>/dev/null || true
export LD_LIBRARY_PATH="${MKLROOT:-}/lib/intel64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

IP="$(hostname -i | awk '{print $1}')"
echo "$IP" > "${COORD_DIR}/ip${SHARD}"
echo "[shard ${SHARD}] host=$(hostname) ip=${IP} port=3456 load=${LOAD_PREFIX}" >&2

cd "$DIST"
exec ./worker --load "$LOAD_PREFIX" --port 3456 \
  --query "$QUERY" --gt "$GTI" --gtd "$GTD" \
  --required-recall "$REQUIRED_RECALL" --hyperparam-id "$HYPERPARAM_ID" \
  --omp-threads "$OMP_NUM_THREADS"
