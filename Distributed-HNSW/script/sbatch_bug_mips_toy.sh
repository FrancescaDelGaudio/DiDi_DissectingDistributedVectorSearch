#!/bin/bash
# bug_mips on TTI toy — 2 shards, 2 physical nodes (4 MPI ranks).
#
# MPI layout (SIZEWORKER=2 in bug_mips build):
#   rank 0,1 = slaves (one sub-HNSW per shard)
#   rank 2   = sender (meta-HNSW + wakeup)
#   rank 3   = receiver (merge + recall)
#
# Slurm: 2 nodes x 2 tasks/node = 4 ranks across 2 physical nodes.
#
# mode_code=1 → build hnsw_slave* (step 17) + queries
# mode_code=0 → load indices + queries only
#
# Usage:
#   sbatch script/sbatch_bug_mips_toy.sh
#   MODE_CODE=0 sbatch script/sbatch_bug_mips_toy.sh
#   sbatch --partition=ncpumshort script/sbatch_bug_mips_toy.sh

#SBATCH -J dhnsw-bug-mips-toy
#SBATCH -p ncpumshort
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH -t 0:15:00
#SBATCH --mem=8G
#SBATCH -o /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log

set -euo pipefail

DHNSW="${DHNSW:-/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/build/bug_mips}"
DATA="${DATA:-/gpfs/scratch/delgaudio/vectors/toy/generated_deep_files}"
BASE="${BASE:-$DATA/tti_100_base.fvecs}"
QUERY="${QUERY:-$DATA/tti_100_query.fvecs}"
GT="${GT:-$DATA/tti_100_gt_k10.lshbox}"
OUT="${OUT:-$DATA/index_tti_100_mips}"
MODE_CODE="${MODE_CODE:-0}"
NUM_CENTROID="${NUM_CENTROID:-20}"
NUM_SUBHNSW="${NUM_SUBHNSW:-2}"
DIM="${DIM:-200}"
SENDER_EF="${SENDER_EF:-100}"
SLAVE_EF="${SLAVE_EF:-100}"
WAKEUP_RATE="${WAKEUP_RATE:-10}"
PRINT_RESULTS="${PRINT_RESULTS:-1}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/gpfs/scratch/delgaudio/dhnsw/logs}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

command -v module >/dev/null 2>&1 || {
  echo "FATAL: 'module' command not available"
  exit 1
}

module purge
module load gcc/10.3.0 openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$SLURM_LOG_DIR" "$OUT/subset"

echo "==== bug_mips toy start $(date -Is) jobid=${SLURM_JOB_ID:-?} ===="
echo "nodes=${SLURM_JOB_NODELIST:-?} ntasks=${SLURM_NTASKS:-?}"
echo "DHNSW=$DHNSW OUT=$OUT MODE_CODE=$MODE_CODE WAKEUP_RATE=$WAKEUP_RATE"

test -x "$DHNSW" || { echo "FATAL: bug_mips not executable: $DHNSW"; exit 1; }
test -f "$OUT/hnsw_meta" || { echo "FATAL: run single_machine_mips first ($OUT/hnsw_meta)"; exit 1; }
test -f "$QUERY" || { echo "FATAL: missing $QUERY"; exit 1; }
test -f "$GT" || { echo "FATAL: missing $GT"; exit 1; }

if [[ "$MODE_CODE" == "0" || "$MODE_CODE" == "10" ]]; then
  for i in 0 1; do
    test -f "$OUT/hnsw_slave$i" || { echo "FATAL: missing $OUT/hnsw_slave$i (run MODE_CODE=1 first)"; exit 1; }
    test -f "$OUT/subset/slave$i" || { echo "FATAL: missing $OUT/subset/slave$i"; exit 1; }
  done
fi

: "${SLURM_NTASKS:?SLURM_NTASKS not set}"
: "${SLURM_NNODES:?SLURM_NNODES not set}"
_expected_tasks=$((NUM_SUBHNSW + 2))
if [[ "$SLURM_NTASKS" -ne "$_expected_tasks" ]]; then
  echo "FATAL: SLURM_NTASKS=$SLURM_NTASKS expected $_expected_tasks (num_subhnsw + sender + receiver)"
  exit 1
fi
_slots_per_node=$((SLURM_NTASKS / SLURM_NNODES))
if [[ $((SLURM_NTASKS % SLURM_NNODES)) -ne 0 ]]; then
  echo "FATAL: SLURM_NTASKS must divide SLURM_NNODES evenly"
  exit 1
fi

_hosts=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | awk -v s="${_slots_per_node}" '{printf "%s:%s,", $1, s}' | sed 's/,$//')
_map_by="slot"
echo "mpirun=$(command -v mpirun) hosts=${_hosts} map-by=${_map_by}"

mpirun -np "${SLURM_NTASKS}" \
  --host "${_hosts}" \
  --map-by "${_map_by}" \
  --bind-to none \
  --report-bindings \
  "$DHNSW" \
  --dim "$DIM" \
  --num_centroid "$NUM_CENTROID" \
  --num_subhnsw "$NUM_SUBHNSW" \
  --wakeup_rate "$WAKEUP_RATE" \
  -b "$BASE" \
  -q "$QUERY" \
  -g "$GT" \
  --hnsw_dir "$OUT" \
  --subset_dir "$OUT/subset" \
  --centroids_file "$OUT/centroids" \
  --cluster_file "$OUT/clusters" \
  --mode_code "$MODE_CODE" \
  --sender_ef "$SENDER_EF" \
  --slave_ef "$SLAVE_EF" \
  --print_results "$PRINT_RESULTS"

echo "==== bug_mips toy done $(date -Is) exit=$? ===="
