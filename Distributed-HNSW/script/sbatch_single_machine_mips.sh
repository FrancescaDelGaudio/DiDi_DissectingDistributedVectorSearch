#!/bin/bash
# Distributed-HNSW step 1 (MIPS): single_machine_mips indexing (Algorithm 5).
#
# Produces in $OUT:
#   centroids, clusters, map, partition_map, hnsw_meta
#
# Prerequisites:
#   - base vectors in fvecs format ($DATA/base.fvecs)
#   - single_machine_mips built with Faiss (script/build_faiss_gcc10.sh)
#
# Usage:
#   sbatch script/sbatch_single_machine_mips.sh
#   NUM_SUBHNSW=4 NUM_CENTROID=10000 MIPS_REPLICA_R=300 sbatch script/sbatch_single_machine_mips.sh
#
# Optional env overrides:
#   BUILD, DATA, OUT, DIM, NUM_CENTROID, NUM_SUBHNSW, MIPS_SAMPLE_N0, MIPS_REPLICA_R

#SBATCH -J dhnsw-sm-mips
#SBATCH -p ncpum
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 8:00:00
#SBATCH --mem=128G
#SBATCH -o /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/dhnsw/logs/%x-%j.log

set -euo pipefail

ROOT="${ROOT:-/gpfs/users/delgaudio/exp_eval/Distributed-HNSW}"
BUILD="${BUILD:-$ROOT/build/single_machine_mips}"
DATA="${DATA:-/gpfs/scratch/delgaudio/vectors/TTI}"
OUT="${OUT:-/gpfs/scratch/delgaudio/vectors/TTI/index_4shard_m10k_mips}"
DIM="${DIM:-200}"
NUM_CENTROID="${NUM_CENTROID:-10000}"
NUM_SUBHNSW="${NUM_SUBHNSW:-4}"
MIPS_SAMPLE_N0="${MIPS_SAMPLE_N0:-100000}"
MIPS_REPLICA_R="${MIPS_REPLICA_R:-300}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/gpfs/scratch/delgaudio/dhnsw/logs}"
FAISS_LIB="${FAISS_LIB:-$ROOT/third_party/faiss/build-gcc10/faiss/libfaiss.so}"
MKL_LIB="${MKL_LIB:-/gpfs/softs/intel/2020U4/mkl/lib/intel64_lin}"

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

export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib:${FAISS_LIB%/*}:${MKL_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$SLURM_LOG_DIR" "$OUT"

echo "==== single_machine_mips start $(date -Is) jobid=${SLURM_JOB_ID:-?} node=$(hostname) ===="
echo "BUILD=$BUILD"
echo "DATA=$DATA"
echo "OUT=$OUT"
echo "DIM=$DIM NUM_CENTROID=$NUM_CENTROID NUM_SUBHNSW=$NUM_SUBHNSW"
echo "MIPS_SAMPLE_N0=$MIPS_SAMPLE_N0 MIPS_REPLICA_R=$MIPS_REPLICA_R"

test -f "$BUILD" && test -x "$BUILD" || { echo "FATAL: single_machine_mips not found: $BUILD"; exit 1; }
test -f "$FAISS_LIB" || { echo "FATAL: libfaiss not found: $FAISS_LIB (run script/build_faiss_gcc10.sh)"; exit 1; }
test -f "$DATA/base.fvecs" || { echo "FATAL: missing $DATA/base.fvecs"; exit 1; }

ldd "$BUILD" | sed -n '/not found/p' || true

"$BUILD" \
  --dim "$DIM" \
  --num_centroid "$NUM_CENTROID" \
  --num_subhnsw "$NUM_SUBHNSW" \
  --mips_sample_n0 "$MIPS_SAMPLE_N0" \
  --mips_replica_r "$MIPS_REPLICA_R" \
  -b "$DATA/base.fvecs" \
  --out_dir "$OUT/"

echo "==== single_machine_mips done $(date -Is) ===="
ls -la "$OUT"
