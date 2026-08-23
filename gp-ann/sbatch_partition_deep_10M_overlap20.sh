#!/bin/bash
# Overlapping partition (20%) of DEEP base 10M on one compute node.
#
# Usage (from gp-ann repo):
#   sbatch /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_partition_deep_10M_overlap20.sh
#
# Output (clusters format, one row per shard):
#   /gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.o=0.2
#
# Optional fixed-machines-mode (keeps exactly K final shards, checks per-shard memory budget):
#   PARTITION_MODE=fixed-machines-mode MEMORY_BUDGET_GB=110 K=4 sbatch ...
#
# Next steps: QueryAttribution with PARTITION_FILE set to that file, then
# DistributedBench (after cover-based LoadShardPointSet is implemented).

#SBATCH -J part_10M_o20
#SBATCH -p ncpum
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 4:00:00
#SBATCH --mem=64G
#SBATCH -o /gpfs/scratch/delgaudio/gp_ann/slurm_partition_10M_o20-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/gp_ann/slurm_partition_10M_o20-%j.log

set -euo pipefail

DATASET_FBIN="${DATASET_FBIN:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.fbin}"
OUT_PREFIX="${OUT_PREFIX:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition}"
K="${K:-8}"
PARTITION_METHOD="${PARTITION_METHOD:-GP}"
KAMINPAR_MODE="${KAMINPAR_MODE:-default}"
OVERLAP="${OVERLAP:-0.2}"
PARTITION_MODE="${PARTITION_MODE:-}"
MEMORY_BUDGET_GB="${MEMORY_BUDGET_GB:-110}"

PARTITION_BIN="${PARTITION_BIN:-/gpfs/users/delgaudio/exp_eval/gp-ann/release_l2/Partition}"

PART_FILE="${OUT_PREFIX}.k=${K}.${PARTITION_METHOD}.o=${OVERLAP}"

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

# Partition in release_l2 is built with gcc/11.3 (needs GLIBCXX_3.4.29); tbb/gcc10 stay after.
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

echo "==== partition DEEP 10M overlap=${OVERLAP} start $(date -Is) jobid=${SLURM_JOB_ID:-?} node=$(hostname) ===="
echo "Input:  $DATASET_FBIN"
echo "Output: $PART_FILE"
echo "K=$K method=$PARTITION_METHOD kaminpar=$KAMINPAR_MODE overlap=$OVERLAP"
if [[ -n "$PARTITION_MODE" ]]; then
  MEMORY_BUDGET_BYTES=$((MEMORY_BUDGET_GB * 1024 * 1024 * 1024))
  echo "PARTITION_MODE=$PARTITION_MODE MEMORY_BUDGET_GB=$MEMORY_BUDGET_GB (${MEMORY_BUDGET_BYTES} bytes)"
fi
echo "BIN=$PARTITION_BIN"

test -f "$DATASET_FBIN" || { echo "FATAL: input not found: $DATASET_FBIN"; exit 1; }
test -x "$PARTITION_BIN" || { echo "FATAL: not executable: $PARTITION_BIN"; exit 1; }
ldd "$PARTITION_BIN" | sed -n '/not found/p' || true
if ldd "$PARTITION_BIN" 2>&1 | grep -q 'GLIBCXX_3.4.29 not found'; then
  echo "FATAL: libstdc++ too old — prepend /softs/gcc/11.3.0/lib64 (rebuild Partition with loaded modules)."
  exit 1
fi
ldd "$PARTITION_BIN" | sed -n '/libtbb\.so\.12/p' >/dev/null || {
  echo "FATAL: libtbb.so.12 not resolvable on this node."
  exit 1
}

mkdir -p "$(dirname "$OUT_PREFIX")"

# ./Partition input output K GP default 0.2 [fixed-machines-mode MEMORY_BUDGET_BYTES]
PARTITION_ARGS=(
  "$DATASET_FBIN"
  "$OUT_PREFIX"
  "$K"
  "$PARTITION_METHOD"
  "$KAMINPAR_MODE"
  "$OVERLAP"
)
if [[ -n "$PARTITION_MODE" ]]; then
  PARTITION_ARGS+=("$PARTITION_MODE" "$MEMORY_BUDGET_BYTES")
fi
srun -n 1 -N 1 --cpu-bind=cores "$PARTITION_BIN" "${PARTITION_ARGS[@]}"

echo "==== done $(date -Is) ===="
ls -lh "$PART_FILE"
head -2 "$PART_FILE" || true
