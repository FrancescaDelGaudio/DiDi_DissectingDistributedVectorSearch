#!/bin/bash
# QueryAttribution on DEEP 10M with overlapping partition (20%, cluster file from Partition).
#
# Prerequisite:
#   sbatch sbatch_partition_deep_10M_overlap20.sh
#   → /gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.o=0.2
#
# Usage:
#   sbatch /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_queryattr_deep_10M_overlap20.sh
#
# Outputs:
#   /gpfs/scratch/delgaudio/vectors/DEEP/base.10M.qa_o20.routes
#   /gpfs/scratch/delgaudio/vectors/DEEP/base.10M.qa_o20.searches
#   /gpfs/scratch/delgaudio/vectors/DEEP/base.10M.qa_o20.pareto
#   /gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.o=0.2.routing_index
#   /gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.o=0.2.routing_index_partition

#SBATCH -J qa_deep10M_o20
#SBATCH -p ncpum
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 1-00:00:00
#SBATCH --mem=120G
#SBATCH -o /gpfs/scratch/delgaudio/gp_ann/slurm_queryattr_10M_o20-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/gp_ann/slurm_queryattr_10M_o20-%j.log

set -euo pipefail

BASE_FBIN="${BASE_FBIN:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.fbin}"
QUERY_FBIN="${QUERY_FBIN:-/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin}"
GT_BIN="${GT_BIN:-/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep10M_10kqueries_k100_faiss.bin}"
PARTITION_FILE="${PARTITION_FILE:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.o=0.2}"
OUT_PREFIX="${OUT_PREFIX:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.qa_o20}"

K="${K:-100}"
PART_METHOD="${PART_METHOD:-GP}"
# OGP output for this run had 10 cluster rows (see partition log); must match ReadClusters size.
REQUESTED_SHARDS="${REQUESTED_SHARDS:-10}"

QUERYATTR_BIN="${QUERYATTR_BIN:-/gpfs/users/delgaudio/exp_eval/gp-ann/release_l2/QueryAttribution}"

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

ROUTER_INDEX="${PARTITION_FILE}.routing_index"

echo "==== query attribution (overlap 20%) start $(date -Is) jobid=${SLURM_JOB_ID:-?} node=$(hostname) ===="
echo "BASE_FBIN=$BASE_FBIN"
echo "QUERY_FBIN=$QUERY_FBIN"
echo "GT_BIN=$GT_BIN"
echo "PARTITION_FILE=$PARTITION_FILE"
echo "OUT_PREFIX=$OUT_PREFIX"
echo "ROUTER_INDEX=$ROUTER_INDEX"
echo "K=$K PART_METHOD=$PART_METHOD REQUESTED_SHARDS=$REQUESTED_SHARDS"
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
  "$K" \
  "$PARTITION_FILE" \
  "$OUT_PREFIX" \
  "$PART_METHOD" \
  "$REQUESTED_SHARDS"

echo "==== done $(date -Is) ===="
ls -lh "${OUT_PREFIX}.routes" "${OUT_PREFIX}.searches" "${OUT_PREFIX}.pareto" 2>/dev/null || true
ls -lh "$ROUTER_INDEX" "${ROUTER_INDEX}_partition" 2>/dev/null || true
