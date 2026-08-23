#!/bin/bash
# Partition DEEP learn.350M on one compute node (not the login node).
# Output prefix includes "350M" in the filename stem; Partition appends: .k=<K>.<METHOD>
# Example result: /gpfs/scratch/delgaudio/gp_ann/deep_learn_350M.k=8.GP
#
# Usage: sbatch /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_partition_deep_350M.sh
# Adjust DATASET_FBIN, K, PARTITION_METHOD, modules, and --mem to your site.
#
# Memory: the Partition binary loads the full .fbin into RAM. For DEEP, dim=96, a 350M
# float32 base is on the order of 100+ GiB of vectors alone; request a fat node and
# raise --mem accordingly, or use a smaller slice if the job is killed (OOM).
#
# --- SLURM partitions (this cluster) — pick by RAM + wall time —
#  ncpu / ncpushort     : ncpu[001-042], 2 GB RAM per core,  MaxTime 2d / 2h (ncpushort: higher priority)
#  ncpulong             : ncpu[001-007] only,                MaxTime 7d, lower priority than ncpu
#  ncpulargeshort       : ncpu[001-007], MaxNodes=25,        MaxTime very short (see sinfo; not for 350M)
#  ncpum / ncpumshort   : ncpum[001-014], 4 GB RAM per core, MaxTime 2d / 2h — prefer ncpum for big RAM
#  ngpu*                : A100; Partition is CPU — use only if you know you need the node type
# For a heavy, long 350M run: do NOT use ncpushort/ncpumshort (2h is usually too little).
# Prefer: ncpum (up to 2d) for 4GB/core; ncpulong (7d) if you need a week; ncpushort only for tiny tests.

#SBATCH -J part_deep_350M
# Default: memory-friendlier queue and 2d wall (override with: sbatch -p ncpulong -t 7-00:00:00 this.sh)
#SBATCH -p ncpum
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 2-00:00:00
#SBATCH --export=NONE
# --- If you still OOM, try fewer CPUs on the same node, or #SBATCH --exclusive, or a larger-memory class ---
# #SBATCH --mem=0
#SBATCH --mem=180G
#SBATCH -o /gpfs/scratch/delgaudio/gp_ann/slurm_partition_350M-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/gp_ann/slurm_partition_350M-%j.log

set -euo pipefail

# === Edit paths and parameters ===
# Input: Yandex/DEEP learn sample (float binary)
DATASET_FBIN="${DATASET_FBIN:-/gpfs/scratch/delgaudio/vectors/DEEP/learn.350M.fbin}"

# Output prefix (no extension). Final partition file is:
#   ${OUT_PREFIX}.k=${K}.${PARTITION_METHOD}
OUT_PREFIX="${OUT_PREFIX:-/gpfs/scratch/delgaudio/gp_ann/deep_learn_350M}"

# Number of shards (clusters) for graph partitioning, e.g. 8 to match 8 nodes later
K="${K:-8}"
PARTITION_METHOD="${PARTITION_METHOD:-GP}"   # GP | KMeans | BalancedKMeans | ...
KAMINPAR_MODE="${KAMINPAR_MODE:-default}"   # default | strong

# Binary (GCC/OpenMP toolchain should match the build used for this Partition)
PARTITION_BIN="${PARTITION_BIN:-/gpfs/users/delgaudio/exp_eval/gp-ann/release_l2/Partition}"

# Optional: load the same compiler/MPI module stack you used to build, if the binary
# needs a newer libstdc++/tbb than the default on compute nodes.
# This block proactively sets runtime libs for Partition.
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

# Add common runtime locations for GCC10 + oneTBB.
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

echo "==== partition deep 350M start $(date -Is) jobid=${SLURM_JOB_ID:-?} node=$(hostname) ===="
echo "Input:  $DATASET_FBIN"
echo "Output: ${OUT_PREFIX}.k=${K}.${PARTITION_METHOD}"
test -f "$DATASET_FBIN" || { echo "FATAL: input not found: $DATASET_FBIN"; exit 1; }
test -x "$PARTITION_BIN" || { echo "FATAL: not executable: $PARTITION_BIN"; exit 1; }
ldd "$PARTITION_BIN" | sed -n '/not found/p' || true
ldd "$PARTITION_BIN" | sed -n '/libtbb\.so\.12/p' >/dev/null || {
  echo "FATAL: libtbb.so.12 not resolvable on this node. Load correct tbb/oneapi module or adjust LD_LIBRARY_PATH."
  exit 1
}

mkdir -p "$(dirname "$OUT_PREFIX")"
cd "$(dirname "$OUT_PREFIX")" || exit 1

# Usage: ./Partition input-points output-path num-clusters partitioning-method (default|strong) [overlap]
srun -n 1 -N 1 --cpu-bind=cores "$PARTITION_BIN" \
  "$DATASET_FBIN" \
  "$OUT_PREFIX" \
  "$K" \
  "$PARTITION_METHOD" \
  "$KAMINPAR_MODE"

echo "==== done $(date -Is) ===="
ls -lh "${OUT_PREFIX}.k=${K}.${PARTITION_METHOD}" 2>/dev/null || true
