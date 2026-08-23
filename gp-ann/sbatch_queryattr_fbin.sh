#!/bin/bash
# QueryAttribution for one partition file (GP/OGP clusters format).
#
# Required env:
#   BASE_FBIN, QUERY_FBIN, GT_BIN, PARTITION_FILE, OUT_PREFIX, QUERYATTR_BIN
# Optional:
#   GT_K (default 100), PART_METHOD (default GP), REQUESTED_SHARDS (default: wc -l PARTITION_FILE)
#
# Writes:
#   ${OUT_PREFIX}.{routes,searches,pareto}
#   ${PARTITION_FILE}.routing_index (+ sidecar)

#SBATCH -p ncpum
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH -t 1-00:00:00
#SBATCH --mem=120G
#SBATCH -o /gpfs/scratch/delgaudio/gp_ann/new_experiments_queryAttribution/logs/%x-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/gp_ann/new_experiments_queryAttribution/logs/%x-%j.log

set -euo pipefail

: "${BASE_FBIN:?BASE_FBIN required}"
: "${QUERY_FBIN:?QUERY_FBIN required}"
: "${GT_BIN:?GT_BIN required}"
: "${PARTITION_FILE:?PARTITION_FILE required}"
: "${OUT_PREFIX:?OUT_PREFIX required}"
: "${QUERYATTR_BIN:?QUERYATTR_BIN required}"

GT_K="${GT_K:-100}"
PART_METHOD="${PART_METHOD:-GP}"

if [[ -z "${REQUESTED_SHARDS:-}" ]]; then
  REQUESTED_SHARDS=$(wc -l < "$PARTITION_FILE")
fi
REQUESTED_SHARDS="${REQUESTED_SHARDS//[[:space:]]/}"

ROUTER_INDEX="${PARTITION_FILE}.routing_index"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

command -v module >/dev/null 2>&1 || {
  echo "FATAL: 'module' command not available in batch environment."
  exit 1
}

module purge
module load gcc/10.3.0
module load openmpi/gcc-10.3.0/4.1.1
module load tbb

export LD_LIBRARY_PATH="/softs/gcc/11.3.0/lib64:/softs/gcc/11.3.0/lib:/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for p in \
  /opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8 \
  /softs/oneapi/tbb/latest/lib/intel64/gcc4.8 \
  /softs/intel/oneapi/tbb/latest/lib/intel64/gcc4.8; do
  if [[ -d "$p" ]]; then
    export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH:-}"
  fi
done
hash -r

echo "==== query attribution start $(date -Is) jobid=${SLURM_JOB_ID:-?} node=$(hostname) ===="
echo "BASE_FBIN=$BASE_FBIN"
echo "QUERY_FBIN=$QUERY_FBIN"
echo "GT_BIN=$GT_BIN"
echo "PARTITION_FILE=$PARTITION_FILE"
echo "OUT_PREFIX=$OUT_PREFIX"
echo "ROUTER_INDEX=$ROUTER_INDEX"
echo "GT_K=$GT_K PART_METHOD=$PART_METHOD REQUESTED_SHARDS=$REQUESTED_SHARDS"
echo "BIN=$QUERYATTR_BIN"

test -x "$QUERYATTR_BIN" || { echo "FATAL: QueryAttribution not executable: $QUERYATTR_BIN"; exit 1; }
test -f "$BASE_FBIN" || { echo "FATAL: base not found: $BASE_FBIN"; exit 1; }
test -f "$QUERY_FBIN" || { echo "FATAL: queries not found: $QUERY_FBIN"; exit 1; }
test -f "$GT_BIN" || { echo "FATAL: GT not found: $GT_BIN"; exit 1; }
test -f "$PARTITION_FILE" || { echo "FATAL: partition not found: $PARTITION_FILE"; exit 1; }

mkdir -p "$(dirname "$OUT_PREFIX")"

srun -n 1 -N 1 --cpu-bind=cores "$QUERYATTR_BIN" \
  "$BASE_FBIN" \
  "$QUERY_FBIN" \
  "$GT_BIN" \
  "$GT_K" \
  "$PARTITION_FILE" \
  "$OUT_PREFIX" \
  "$PART_METHOD" \
  "$REQUESTED_SHARDS"

echo "==== done $(date -Is) ===="
ls -lh "${OUT_PREFIX}.routes" "${OUT_PREFIX}.searches" "${OUT_PREFIX}.pareto" 2>/dev/null || true
ls -lh "$ROUTER_INDEX" "${ROUTER_INDEX}_partition" 2>/dev/null || true
