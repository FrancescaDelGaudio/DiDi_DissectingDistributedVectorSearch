#!/usr/bin/env bash
# One-time prep on login node: split base, per-shard GT, slice global GT to 50 queries.
#
#   cd /gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist
#   bash sh/run_auncel_prepare_100k.sh
#
# Prereq: random_100k.fbin, random_100k.query.50.fbin, random_100k.ground_truth.bin

set -euo pipefail

DIST="${AUNCEL_DIST_ROOT:-/gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist}"
# shellcheck source=sh/auncel_100k_env.sh
source "${DIST}/sh/auncel_100k_env.sh"
mkdir -p "${AUNCEL_LOG_DIR}" "${AUNCEL_SHARDS}"

module load intel/20U4/mkl 2>/dev/null || true
export LD_LIBRARY_PATH="${MKLROOT:-}/lib/intel64:${LD_LIBRARY_PATH:-}"

for f in "${AUNCEL_BASE}" "${AUNCEL_QUERY}" "${AUNCEL_GT_FULL}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

# Infer nq from query file header.
read -r NQ DIM < <(python3 -c "
import struct
with open('${AUNCEL_QUERY}','rb') as f:
    n,d=struct.unpack('<II', f.read(8))
print(n, d)
")

read -r GT_NQ GT_K < <(python3 -c "
import struct
with open('${AUNCEL_GT_FULL}','rb') as f:
    n,k=struct.unpack('<II', f.read(8))
print(n, k)
")

echo "=== Auncel 100k prepare ==="
echo "base: ${AUNCEL_BASE}"
echo "queries: ${AUNCEL_QUERY}  (nq=${NQ} dim=${DIM})"
echo "GT full: ${AUNCEL_GT_FULL}  (nq=${GT_NQ} k=${GT_K})"
echo "shards: ${AUNCEL_SHARDS}"

if [[ "${GT_K}" != "${AUNCEL_K}" ]]; then
  echo "WARNING: AUNCEL_K=${AUNCEL_K} but GT k=${GT_K}; using k=${GT_K}" >&2
  AUNCEL_K="${GT_K}"
fi

if [[ "${GT_NQ}" -lt "${NQ}" ]]; then
  echo "ERROR: GT has fewer queries (${GT_NQ}) than query file (${NQ})" >&2
  exit 1
fi

echo "==> slice global GT to first ${NQ} queries"
python3 "${DIST}/scripts/slice_gt_bin.py" \
  --input "${AUNCEL_GT_FULL}" \
  --output "${AUNCEL_GT}" \
  --num-queries "${NQ}"

echo "==> split base into 2 shards"
cd "${DIST}"
./split "${AUNCEL_BASE}" "${AUNCEL_SHARDS}/" 0

echo "==> per-shard brute-force GT (k=${AUNCEL_K})"
./gt "${AUNCEL_SHARDS}/0db.fbin" "${AUNCEL_QUERY}" "${AUNCEL_K}"
./gt "${AUNCEL_SHARDS}/1db.fbin" "${AUNCEL_QUERY}" "${AUNCEL_K}"

ls -la "${AUNCEL_SHARDS}"
echo "Done. Next: sbatch sh/run_auncel_build_shards.sh"
