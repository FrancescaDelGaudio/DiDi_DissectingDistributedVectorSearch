#!/usr/bin/env bash
# One-time prep: 4-way split of DEEP10M base (login node).
#
# Per-shard GT (brute-force, slow) is a separate Slurm job:
#   sbatch sh/run_auncel_gt_deep10m.sh
#
# Usage:
#   cd /gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist
#   bash sh/run_auncel_prepare_deep10m.sh

set -euo pipefail

DIST="${AUNCEL_DIST_ROOT:-/gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist}"
# shellcheck source=sh/auncel_deep10m_env.sh
source "${DIST}/sh/auncel_deep10m_env.sh"

mkdir -p "${AUNCEL_LOG_DIR}" "${AUNCEL_SHARDS}"

for f in "${AUNCEL_BASE}" "${AUNCEL_QUERY}" "${AUNCEL_GT}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

python3 << PY
import struct
from pathlib import Path

def hdr_fbin(p):
    with open(p,'rb') as f:
        return struct.unpack('<II', f.read(8))

def hdr_gt(p):
    with open(p,'rb') as f:
        return struct.unpack('<II', f.read(8))

base = Path("${AUNCEL_BASE}")
query = Path("${AUNCEL_QUERY}")
gt = Path("${AUNCEL_GT}")

nb, d = hdr_fbin(base)
nq, dq = hdr_fbin(query)
nq_gt, k = hdr_gt(gt)

print("=== DEEP10M input check ===")
print(f"base:  n={nb} d={d}")
print(f"query: n={nq} d={dq}")
print(f"GT:    nq={nq_gt} k={k}")
assert d == dq, f"dim mismatch base={d} query={dq}"
assert nq == nq_gt, f"nq mismatch query={nq} gt={nq_gt}"
assert k == ${AUNCEL_K}, f"expected k=${AUNCEL_K}, gt has k={k}"
assert nb % ${AUNCEL_NUM_SHARDS} == 0
assert nq % 10 == 0, "Error_sys requires nq divisible by 10"
print("Coherence OK — .fbin/.bin used as-is (no fvecs conversion).")
print(f"Note: fvecs copies exist under ${AUNCEL_DEEP_DIR}/origin/ but are NOT needed.")
PY

echo "==> split base into ${AUNCEL_NUM_SHARDS} shards"
python3 "${DIST}/scripts/split_fbin.py" \
  --input "${AUNCEL_BASE}" \
  --output-dir "${AUNCEL_SHARDS}" \
  --num-shards "${AUNCEL_NUM_SHARDS}"

ls -la "${AUNCEL_SHARDS}"
echo "Done split. Next: sbatch sh/run_auncel_gt_deep10m.sh"
