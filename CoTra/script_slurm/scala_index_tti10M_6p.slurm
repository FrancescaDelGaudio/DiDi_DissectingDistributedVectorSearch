#!/bin/bash
# Distributed CoTra indexing — TTI yangt2i 10M, MIPS, 6 partitions / 6 nodes.
#
# Requires: bash script_slurm/build_cotra_6p.sh
# Usage: sbatch script_slurm/scala_index_tti10M_6p.slurm

#SBATCH -J cotra_idx6_t
#SBATCH -p ncpum
#SBATCH -N 6
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 08:00:00
#SBATCH --mem=150G
#SBATCH -o /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/scala_index_tti10M_6p-%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/scala_index_tti10M_6p-%j.err

set -euo pipefail

COTRA_ROOT="/gpfs/users/delgaudio/exp_eval/CoTra"
cd "$COTRA_ROOT"
mkdir -p script_slurm/logs script_slurm/runtime

# shellcheck disable=SC1091
source script_slurm/env_slurm.sh

export COTRA_MACHINE_NUM=6
export COTRA_CONF_DIR="${COTRA_CONF_DIR:-${COTRA_ROOT}/script_slurm/runtime}"

COTRA_DATA_PATH="${COTRA_DATA_PATH:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2i10M.fbin}"
COTRA_INDEX_OUT="${COTRA_INDEX_OUT:-/gpfs/scratch/delgaudio/cotra/index/tti10M_6p}"
COTRA_INDEX_PREFIX="${COTRA_INDEX_PREFIX:-${COTRA_INDEX_OUT}/merged_index}"
COTRA_MILLION="${COTRA_MILLION:-10}"
COTRA_THREADS="${COTRA_THREADS:-32}"
COTRA_R="${COTRA_R:-48}"
COTRA_L="${COTRA_L:-500}"
COTRA_B="${COTRA_B:-100}"
COTRA_M="${COTRA_M:-120}"

FAISS_LIB="/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/third_party/faiss/build-gcc10/faiss"
export LD_LIBRARY_PATH="${FAISS_LIB}:${LD_LIBRARY_PATH:-}"

BIN="${COTRA_ROOT}/build-6p/tests/scala_index_6p"

echo "==== scala_index_6p TTI ${COTRA_MILLION}M MIPS start $(date -Is) job=${SLURM_JOB_ID} ===="
echo "INDEX_PREFIX=${COTRA_INDEX_PREFIX} BIN=${BIN}"

test -x "$BIN" || { echo "FATAL: run build_cotra_6p.sh first" >&2; exit 1; }
test -f "$COTRA_DATA_PATH" || { echo "FATAL: missing $COTRA_DATA_PATH" >&2; exit 1; }
mkdir -p "$COTRA_INDEX_OUT"

source script_slurm/generate_cluster_conf.sh
generate_cluster_conf
bash script_slurm/start_memcached_leader.sh

srun --ntasks-per-node=1 --cpus-per-task=1 \
  bash "${COTRA_ROOT}/script_slurm/verify_ip_config.sh"

srun --ntasks-per-node=1 --cpus-per-task="${COTRA_THREADS}" --kill-on-bad-exit=1 \
  "$BIN" \
  -d "${COTRA_IB_DEV:-mlx5_0}" \
  --config_file "$CONF_FILE" \
  --graph_type scalagraph_v3 \
  --data_type float \
  --dist_fn mips \
  --data_path "$COTRA_DATA_PATH" \
  --index_path_prefix "$COTRA_INDEX_PREFIX" \
  -R "$COTRA_R" -L "$COTRA_L" -B "$COTRA_B" -M "$COTRA_M" \
  -T "$COTRA_THREADS" \
  --scala_v3 \
  -s "$COTRA_MILLION" \
  -t "$COTRA_THREADS"

echo "==== scala_index_6p TTI done $(date -Is) ===="
