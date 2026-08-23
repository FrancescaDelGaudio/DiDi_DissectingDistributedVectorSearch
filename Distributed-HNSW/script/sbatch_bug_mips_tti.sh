#!/bin/bash
# bug_mips on TTI10M MIPS — MPI build / query (mode_code=1 or 0).
#
# MPI layout (NUM_SUBHNSW slaves + sender + receiver):
#   ranks 0..N-1     = slaves (one sub-HNSW per shard)
#   rank N           = sender (meta-HNSW + wakeup)
#   rank N+1         = receiver (merge + recall)
#
# mode_code=1 → build hnsw_slave* + subset lists, then run queries
# mode_code=0 → load existing hnsw_slave* and run queries only
#
# Prerequisites:
#   - single_machine_mips → $OUT/{centroids,clusters,hnsw_meta,map,partition_map}
#   - $DATA/{base.fvecs,query.fvecs} and ground-truth .lshbox
#
# Env (set by submit_bug_mips_tti_build.sh or manually):
#   DHNSW, DATA, OUT, GT, NUM_SUBHNSW, RECALL_K, MODE_CODE
#   WAKEUP_RATE, SENDER_EF, SLAVE_EF, PRINT_RESULTS
#
# Usage (4 shard, 6 exclusive nodes, build slaves):
#   DHNSW=.../bug_mips_4_k100 MODE_CODE=1 NUM_SUBHNSW=4 \
#     OUT=.../index_4shard_m10k_mips sbatch --exclusive -N 6 -n 6 --ntasks-per-node=1 \
#     script/sbatch_bug_mips_tti.sh

#SBATCH -J dhnsw-bug-mips-tti
#SBATCH -p ncpum
#SBATCH -N 6
#SBATCH -n 6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 8:00:00
#SBATCH --mem=128G
#SBATCH -o /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log

set -euo pipefail

BUILD="${BUILD:-/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/build}"
DATA="${DATA:-/gpfs/scratch/delgaudio/vectors/TTI}"
OUT="${OUT:-/gpfs/scratch/delgaudio/vectors/TTI/index_4shard_m10k_mips}"
GT="${GT:-$DATA/gt.lshbox}"
NUM_SUBHNSW="${NUM_SUBHNSW:-4}"
RECALL_K="${RECALL_K:-100}"
MODE_CODE="${MODE_CODE:-1}"
WAKEUP_RATE="${WAKEUP_RATE:-10}"
SENDER_EF="${SENDER_EF:-100}"
SLAVE_EF="${SLAVE_EF:-100}"
PRINT_RESULTS="${PRINT_RESULTS:-0}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/gpfs/scratch/delgaudio/dhnsw/logs}"

NUM_CENTROID="${NUM_CENTROID:-10000}"
DIM="${DIM:-200}"

if [[ -n "${DHNSW:-}" ]]; then
  :
elif [[ "$NUM_SUBHNSW" == "4" && "$RECALL_K" == "10" ]]; then
  DHNSW="$BUILD/bug_mips_4_k10"
elif [[ "$NUM_SUBHNSW" == "4" && "$RECALL_K" == "100" ]]; then
  DHNSW="$BUILD/bug_mips_4_k100"
elif [[ "$NUM_SUBHNSW" == "6" && "$RECALL_K" == "10" ]]; then
  DHNSW="$BUILD/bug_mips_6_k10"
elif [[ "$NUM_SUBHNSW" == "6" && "$RECALL_K" == "100" ]]; then
  DHNSW="$BUILD/bug_mips_6_k100"
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

echo "==== bug_mips TTI start $(date -Is) jobid=${SLURM_JOB_ID:-?} ===="
echo "nodes=${SLURM_JOB_NODELIST:-?} ntasks=${SLURM_NTASKS:-?}"
echo "DHNSW=$DHNSW"
echo "DATA=$DATA"
echo "OUT=$OUT"
echo "GT=$GT"
echo "NUM_SUBHNSW=$NUM_SUBHNSW RECALL_K=$RECALL_K WAKEUP_RATE=$WAKEUP_RATE"
echo "MODE_CODE=$MODE_CODE SENDER_EF=$SENDER_EF SLAVE_EF=$SLAVE_EF PRINT_RESULTS=$PRINT_RESULTS"

test -f "$DHNSW" && test -x "$DHNSW" || { echo "FATAL: bug_mips binary not found or not executable: $DHNSW"; exit 1; }
test -f "$DATA/base.fvecs" || { echo "FATAL: missing $DATA/base.fvecs"; exit 1; }
test -f "$DATA/query.fvecs" || { echo "FATAL: missing $DATA/query.fvecs"; exit 1; }
test -f "$GT" || { echo "FATAL: missing ground truth: $GT"; exit 1; }
test -f "$OUT/centroids" || { echo "FATAL: missing $OUT/centroids (run single_machine_mips first)"; exit 1; }
test -f "$OUT/clusters" || { echo "FATAL: missing $OUT/clusters (run single_machine_mips first)"; exit 1; }
test -f "$OUT/hnsw_meta" || { echo "FATAL: missing $OUT/hnsw_meta (run single_machine_mips first)"; exit 1; }
test -f "$OUT/map" || { echo "FATAL: missing $OUT/map (run single_machine_mips first)"; exit 1; }

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
  --report-bindings \
  "$DHNSW" \
  --dim "$DIM" \
  --num_centroid "$NUM_CENTROID" \
  --num_subhnsw "$NUM_SUBHNSW" \
  --wakeup_rate "$WAKEUP_RATE" \
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
  --print_results "$PRINT_RESULTS"

echo "==== bug_mips TTI done $(date -Is) exit=$? ===="
