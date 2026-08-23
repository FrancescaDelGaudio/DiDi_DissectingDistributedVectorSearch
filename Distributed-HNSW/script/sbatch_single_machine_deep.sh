#!/bin/bash
# Distributed-HNSW step 1: single_machine indexing on DEEP10M (L2).
#
# Produces in $OUT:
#   centroids, clusters, map, partition_map, partition, hnsw_meta
#
# Prerequisites:
#   - converted fvecs in $DATA (step 0: convert_to_dhnsw.py)
#   - single_machine built in $BUILD
#
# Usage:
#   sbatch script/sbatch_single_machine_deep.sh
#   NUM_SUBHNSW=4 NUM_CENTROID=10000 sbatch script/sbatch_single_machine_deep.sh
#
# Optional env overrides:
#   BUILD, DATA, OUT, DIM, NUM_CENTROID, NUM_SUBHNSW, SLURM_LOG_DIR

#SBATCH -J dhnsw-sm-deep
#SBATCH -p ncpum
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 8:00:00
#SBATCH --mem=128G
#SBATCH -o /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log

set -euo pipefail

BUILD="${BUILD:-/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/build/single_machine}"
DATA="${DATA:-/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid}"
OUT="${OUT:-/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid/index_4shard_m10k}"
DIM="${DIM:-96}"
NUM_CENTROID="${NUM_CENTROID:-10000}"
NUM_SUBHNSW="${NUM_SUBHNSW:-4}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/gpfs/scratch/delgaudio/dhnsw/logs}"

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

mkdir -p "$SLURM_LOG_DIR" "$OUT"

echo "==== single_machine start $(date -Is) jobid=${SLURM_JOB_ID:-?} node=$(hostname) ===="
echo "BUILD=$BUILD"
echo "DATA=$DATA"
echo "OUT=$OUT"
echo "DIM=$DIM NUM_CENTROID=$NUM_CENTROID NUM_SUBHNSW=$NUM_SUBHNSW"

test -f "$BUILD" && test -x "$BUILD" || { echo "FATAL: single_machine not found: $BUILD"; exit 1; }
test -f "$DATA/base.fvecs" || { echo "FATAL: missing $DATA/base.fvecs"; exit 1; }

ldd "$BUILD" | sed -n '/not found/p' || true

"$BUILD" \
  --dim "$DIM" \
  --num_centroid "$NUM_CENTROID" \
  --num_subhnsw "$NUM_SUBHNSW" \
  -b "$DATA/base.fvecs" \
  --out_dir "$OUT/"

echo "==== single_machine done $(date -Is) ===="
ls -la "$OUT"
