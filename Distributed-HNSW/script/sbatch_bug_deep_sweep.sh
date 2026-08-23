#!/bin/bash
# Distributed-HNSW MPI query sweep (bug, mode_code=0) — DEEP10M L2.
#
# MPI layout (NUM_SUBHNSW slaves + sender + receiver):
#   ranks 0..N-1     = slaves
#   rank N           = sender (meta-HNSW + wakeup)
#   rank N+1         = receiver (merge + recall)
#
# Prerequisites:
#   - single_machine done → $OUT/{centroids,clusters,hnsw_meta,...}
#   - bug --mode_code 1 done → $OUT/hnsw_slave{0..N-1} and $OUT/subset/slave{0..N-1}
#
# Env (set by submit_deep_k_sweep.sh or manually):
#   DHNSW, DATA, OUT, GT, NUM_SUBHNSW, WAKEUP_RATE, RECALL_K
#   MODE_CODE (default 0), SENDER_EF, SLAVE_EF, PRINT_RESULTS
#
# Usage:
#   sbatch script/sbatch_bug_deep_sweep.sh
#   DHNSW=.../bug_4_k10 WAKEUP_RATE=10 GT=.../gt_k10.lshbox \
#     NUM_SUBHNSW=4 OUT=.../index_4shard_m10k sbatch -N 2 -n 6 --ntasks-per-node=3 \
#     script/sbatch_bug_deep_sweep.sh

#SBATCH -J dhnsw-bug-deep
#SBATCH -p ncpum
#SBATCH -N 2
#SBATCH -n 6
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH -t 4:00:00
#SBATCH --mem=64G
#SBATCH -o /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log

set -euo pipefail

BUILD="${BUILD:-/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/build}"
DATA="${DATA:-/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid}"
OUT="${OUT:-/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid/index_4shard_m10k}"
GT="${GT:-$DATA/gt.lshbox}"
NUM_SUBHNSW="${NUM_SUBHNSW:-4}"
WAKEUP_RATE="${WAKEUP_RATE:-10}"
RECALL_K="${RECALL_K:-100}"
MODE_CODE="${MODE_CODE:-0}"
SENDER_EF="${SENDER_EF:-100}"
SLAVE_EF="${SLAVE_EF:-100}"
PRINT_RESULTS="${PRINT_RESULTS:-0}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/gpfs/scratch/delgaudio/dhnsw/logs}"

NUM_CENTROID="${NUM_CENTROID:-10000}"
DIM="${DIM:-96}"

if [[ -n "${DHNSW:-}" ]]; then
  :
elif [[ "$NUM_SUBHNSW" == "4" && "$RECALL_K" == "10" ]]; then
  DHNSW="$BUILD/bug_4_k10"
elif [[ "$NUM_SUBHNSW" == "4" && "$RECALL_K" == "100" ]]; then
  DHNSW="$BUILD/bug_4_k100"
elif [[ "$NUM_SUBHNSW" == "6" && "$RECALL_K" == "10" ]]; then
  DHNSW="$BUILD/bug_6_k10"
elif [[ "$NUM_SUBHNSW" == "6" && "$RECALL_K" == "100" ]]; then
  DHNSW="$BUILD/bug_6_k100"
else
  echo "FATAL: set DHNSW explicitly for NUM_SUBHNSW=$NUM_SUBHNSW RECALL_K=$RECALL_K"
  exit 1
fi

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

echo "==== bug deep sweep start $(date -Is) jobid=${SLURM_JOB_ID:-?} ===="
echo "nodes=${SLURM_JOB_NODELIST:-?} ntasks=${SLURM_NTASKS:-?}"
echo "DHNSW=$DHNSW"
echo "DATA=$DATA"
echo "OUT=$OUT"
echo "GT=$GT"
echo "NUM_SUBHNSW=$NUM_SUBHNSW WAKEUP_RATE=$WAKEUP_RATE RECALL_K=$RECALL_K"
echo "MODE_CODE=$MODE_CODE SENDER_EF=$SENDER_EF SLAVE_EF=$SLAVE_EF PRINT_RESULTS=$PRINT_RESULTS"

test -f "$DHNSW" && test -x "$DHNSW" || { echo "FATAL: bug binary not found or not executable: $DHNSW"; exit 1; }
test -f "$DATA/base.fvecs" || { echo "FATAL: missing $DATA/base.fvecs"; exit 1; }
test -f "$DATA/query.fvecs" || { echo "FATAL: missing $DATA/query.fvecs"; exit 1; }
test -f "$GT" || { echo "FATAL: missing ground truth: $GT"; exit 1; }
test -f "$OUT/centroids" || { echo "FATAL: missing $OUT/centroids (run single_machine first)"; exit 1; }
test -f "$OUT/clusters" || { echo "FATAL: missing $OUT/clusters (run single_machine first)"; exit 1; }
test -f "$OUT/hnsw_meta" || { echo "FATAL: missing $OUT/hnsw_meta (run single_machine first)"; exit 1; }

if [[ "$MODE_CODE" == "0" || "$MODE_CODE" == "10" ]]; then
  for ((i = 0; i < NUM_SUBHNSW; i++)); do
    test -f "$OUT/hnsw_slave${i}" || { echo "FATAL: missing $OUT/hnsw_slave${i} (run MODE_CODE=1 first)"; exit 1; }
    test -f "$OUT/subset/slave${i}" || { echo "FATAL: missing $OUT/subset/slave${i}"; exit 1; }
  done
fi

ldd "$DHNSW" | sed -n '/not found/p' || true

: "${SLURM_NTASKS:?SLURM_NTASKS not set}"
: "${SLURM_NNODES:?SLURM_NNODES not set}"
_expected_tasks=$((NUM_SUBHNSW + 2))
if [[ "$SLURM_NTASKS" -ne "$_expected_tasks" ]]; then
  echo "FATAL: SLURM_NTASKS=$SLURM_NTASKS but expected $_expected_tasks (num_subhnsw + sender + receiver)"
  exit 1
fi
if [[ "$SLURM_NNODES" -ne "$SLURM_NTASKS" ]]; then
  echo "FATAL: expected one MPI rank per node (SLURM_NNODES=$SLURM_NNODES SLURM_NTASKS=$SLURM_NTASKS)"
  exit 1
fi

_hosts=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | awk '{printf "%s:1,", $1}' | sed 's/,$//')
_map_by="ppr:1:node"
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
  -g "$GT" \
  --hnsw_dir "$OUT" \
  --subset_dir "$OUT/subset" \
  --centroids_file "$OUT/centroids" \
  --cluster_file "$OUT/clusters" \
  --mode_code "$MODE_CODE" \
  --sender_ef "$SENDER_EF" \
  --slave_ef "$SLAVE_EF" \
  --wakeup_rate "$WAKEUP_RATE" \
  --print_results "$PRINT_RESULTS"

echo "==== bug deep sweep done $(date -Is) exit=$? ===="
