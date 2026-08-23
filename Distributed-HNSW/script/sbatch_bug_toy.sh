#!/bin/bash
# Distributed-HNSW MPI benchmark (bug) — toy dataset on 2 compute nodes.
#
# MPI layout (SIZEWORKER=2 in macro.h → 4 ranks total):
#   rank 0,1 = slaves (shard HNSW search)
#   rank 2   = sender (meta-HNSW + wakeup)
#   rank 3   = receiver (merge + recall)
#
# Prerequisites:
#   - single_machine step-1 indexing done → $OUT/{centroids,clusters,hnsw_meta,...}
#   - for --mode_code 0: $OUT/hnsw_slave{0,1} and $OUT/subset/slave{0,1}
#   - for --mode_code 1: builds slave indices then runs queries
#
# Usage:
#   sbatch script/sbatch_bug_toy.sh
#   MODE_CODE=0 PRINT_RESULTS=1 sbatch script/sbatch_bug_toy.sh
#
# Optional env overrides:
#   DHNSW, DATA, OUT, MODE_CODE, SENDER_EF, SLAVE_EF, PRINT_RESULTS, SLURM_LOG_DIR

#SBATCH -J dhnsw-bug-toy
#SBATCH -p ncpum
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH -t 0:30:00
#SBATCH --mem=8G
#SBATCH -o /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log

set -euo pipefail

DHNSW="${DHNSW:-/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/build/bug}"
DATA="${DATA:-/gpfs/scratch/delgaudio/vectors/toy/generated_deep_files/pyramid}"
OUT="${OUT:-/gpfs/scratch/delgaudio/vectors/toy/generated_deep_files/pyramid/index}"
MODE_CODE="${MODE_CODE:-0}"
SENDER_EF="${SENDER_EF:-100}"
SLAVE_EF="${SLAVE_EF:-100}"
PRINT_RESULTS="${PRINT_RESULTS:-1}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/gpfs/scratch/delgaudio/dhnsw/logs}"

NUM_CENTROID="${NUM_CENTROID:-20}"
NUM_SUBHNSW="${NUM_SUBHNSW:-2}"
DIM="${DIM:-32}"

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

export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$SLURM_LOG_DIR" "$OUT/subset"

echo "==== bug start $(date -Is) jobid=${SLURM_JOB_ID:-?} ===="
echo "nodes=${SLURM_JOB_NODELIST:-?} ntasks=${SLURM_NTASKS:-?}"
echo "DHNSW=$DHNSW"
echo "DATA=$DATA"
echo "OUT=$OUT"
echo "MODE_CODE=$MODE_CODE SENDER_EF=$SENDER_EF SLAVE_EF=$SLAVE_EF PRINT_RESULTS=$PRINT_RESULTS"

test -f "$DHNSW" && test -x "$DHNSW" || { echo "FATAL: bug binary not found or not executable: $DHNSW"; exit 1; }
test -f "$DATA/base.fvecs" || { echo "FATAL: missing $DATA/base.fvecs"; exit 1; }
test -f "$DATA/query.fvecs" || { echo "FATAL: missing $DATA/query.fvecs"; exit 1; }
test -f "$DATA/gt.lshbox" || { echo "FATAL: missing $DATA/gt.lshbox"; exit 1; }
test -f "$OUT/centroids" || { echo "FATAL: missing $OUT/centroids (run single_machine first)"; exit 1; }
test -f "$OUT/clusters" || { echo "FATAL: missing $OUT/clusters (run single_machine first)"; exit 1; }
test -f "$OUT/hnsw_meta" || { echo "FATAL: missing $OUT/hnsw_meta (run single_machine first)"; exit 1; }

if [[ "$MODE_CODE" == "0" || "$MODE_CODE" == "10" ]]; then
  test -f "$OUT/hnsw_slave0" || { echo "FATAL: missing $OUT/hnsw_slave0 (run with MODE_CODE=1 first)"; exit 1; }
  test -f "$OUT/hnsw_slave1" || { echo "FATAL: missing $OUT/hnsw_slave1 (run with MODE_CODE=1 first)"; exit 1; }
  test -f "$OUT/subset/slave0" || { echo "FATAL: missing $OUT/subset/slave0"; exit 1; }
  test -f "$OUT/subset/slave1" || { echo "FATAL: missing $OUT/subset/slave1"; exit 1; }
fi

ldd "$DHNSW" | sed -n '/not found/p' || true

# OpenMPI module is not built with Slurm PMI; launch with mpirun inside allocation.
# Default: 2 nodes x 2 ranks (efficient). Alternative: -N 4 -n 4 --ntasks-per-node=1
# (one rank per node; script adapts slot count automatically).
: "${SLURM_NTASKS:?SLURM_NTASKS not set}"
: "${SLURM_NNODES:?SLURM_NNODES not set}"
_expected_tasks=$((NUM_SUBHNSW + 2))
if [[ "$SLURM_NTASKS" -ne "$_expected_tasks" ]]; then
  echo "FATAL: SLURM_NTASKS=$SLURM_NTASKS but expected $_expected_tasks (num_subhnsw + sender + receiver)"
  exit 1
fi
_slots_per_node=$((SLURM_NTASKS / SLURM_NNODES))
if [[ $((SLURM_NTASKS % SLURM_NNODES)) -ne 0 ]]; then
  echo "FATAL: SLURM_NTASKS ($SLURM_NTASKS) must be divisible by SLURM_NNODES ($SLURM_NNODES)"
  exit 1
fi

_hosts=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | awk -v s="${_slots_per_node}" '{printf "%s:%s,", $1, s}' | sed 's/,$//')
if [[ "$_slots_per_node" -eq 1 ]]; then
  _map_by="ppr:1:node"
else
  _map_by="slot"
fi
echo "mpirun=$(command -v mpirun) hosts=${_hosts} map-by=${_map_by}"

mpirun -np "${SLURM_NTASKS}" \
  --host "${_hosts}" \
  --map-by "${_map_by}" \
  --bind-to none \
  "$DHNSW" \
  --dim "$DIM" \
  --num_centroid "$NUM_CENTROID" \
  --num_subhnsw "$NUM_SUBHNSW" \
  -b "$DATA/base.fvecs" \
  -q "$DATA/query.fvecs" \
  -g "$DATA/gt.lshbox" \
  --hnsw_dir "$OUT" \
  --subset_dir "$OUT/subset" \
  --centroids_file "$OUT/centroids" \
  --cluster_file "$OUT/clusters" \
  --mode_code "$MODE_CODE" \
  --sender_ef "$SENDER_EF" \
  --slave_ef "$SLAVE_EF" \
  --print_results "$PRINT_RESULTS"

echo "==== bug done $(date -Is) exit=$? ===="
