#!/bin/bash
# QueryAttribution on one compute node (not login node).
# Default target: DEEP 10M base + public 10K queries + k=100 GT.
#
# Usage:
#   sbatch /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_queryattr_deep_10M.sh
#
# Optional overrides:
#   sbatch --export=ALL,BASE_FBIN=...,QUERY_FBIN=...,GT_BIN=...,PARTITION_FILE=...,OUT_PREFIX=...,K=100,REQUESTED_SHARDS=8 \
#     /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_queryattr_deep_10M.sh
#
# Outputs (with default OUT_PREFIX):
#   /gpfs/scratch/delgaudio/vectors/DEEP/base.10M.qa.routes
#   /gpfs/scratch/delgaudio/vectors/DEEP/base.10M.qa.searches
#   /gpfs/scratch/delgaudio/vectors/DEEP/base.10M.qa.pareto
# Plus router written by QueryAttribution:
#   ${PARTITION_FILE}.routing_index

#SBATCH -J queryattr_deep10M
#SBATCH -p ncpum
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 1-00:00:00
#SBATCH --mem=120G
#SBATCH --export=NONE
#SBATCH -o /gpfs/scratch/delgaudio/gp_ann/slurm_queryattr_10M-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/gp_ann/slurm_queryattr_10M-%j.log

set -euo pipefail

# Inputs / outputs
BASE_FBIN="${BASE_FBIN:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.fbin}"
QUERY_FBIN="${QUERY_FBIN:-/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin}"
GT_BIN="${GT_BIN:-/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep10M_10kqueries_k100_faiss.bin}"
PARTITION_FILE="${PARTITION_FILE:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP}"
OUT_PREFIX="${OUT_PREFIX:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.qa}"

# QueryAttribution params
K="${K:-100}"
PART_METHOD="${PART_METHOD:-GP}"
REQUESTED_SHARDS="${REQUESTED_SHARDS:-8}"

# Binary
QUERYATTR_BIN="${QUERYATTR_BIN:-/gpfs/users/delgaudio/exp_eval/gp-ann/build_clean10/QueryAttribution}"

# Runtime modules
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

export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
echo "K=$K PART_METHOD=$PART_METHOD REQUESTED_SHARDS=$REQUESTED_SHARDS"

test -x "$QUERYATTR_BIN" || { echo "FATAL: QueryAttribution binary not executable: $QUERYATTR_BIN"; exit 1; }
test -f "$BASE_FBIN" || { echo "FATAL: base file not found: $BASE_FBIN"; exit 1; }
test -f "$QUERY_FBIN" || { echo "FATAL: query file not found: $QUERY_FBIN"; exit 1; }
test -f "$GT_BIN" || { echo "FATAL: gt file not found: $GT_BIN"; exit 1; }
test -f "$PARTITION_FILE" || { echo "FATAL: partition file not found: $PARTITION_FILE"; exit 1; }

mkdir -p "$(dirname "$OUT_PREFIX")"

# Usage:
# QueryAttribution input-points queries ground-truth-file num_neighbors partition-file output-file partition_method requested-num-shards
srun -n 1 -N 1 --cpu-bind=cores "$QUERYATTR_BIN" \
  "$BASE_FBIN" \
  "$QUERY_FBIN" \
  "$GT_BIN" \
  "$K" \
  "$PARTITION_FILE" \
  "$OUT_PREFIX" \
  "$PART_METHOD" \
  "$REQUESTED_SHARDS"

echo "==== done $(date -Is) ===="
ls -lh "${OUT_PREFIX}.routes" "${OUT_PREFIX}.searches" "${OUT_PREFIX}.pareto" 2>/dev/null || true
ls -lh "${PARTITION_FILE}.routing_index" 2>/dev/null || true
