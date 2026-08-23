#!/bin/bash
# Distributed CoTra indexing — DEEP 10M, L2, 6 partitions / 6 nodes.
#
# Requires build-6p binary (does NOT use build/tests/scala_index):
#   bash script_slurm/build_cotra_6p.sh
#
# Usage:
#   sbatch script_slurm/scala_index_deep10M_6p.slurm
#
# Output: /gpfs/scratch/delgaudio/cotra/index/deep10M_6p/merged_index{0..5}_*

#SBATCH -J cotra_idx6_d
#SBATCH -p ncpum
#SBATCH -N 6
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 08:00:00
#SBATCH --mem=150G
#SBATCH -o /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/scala_index_deep10M_6p-%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/scala_index_deep10M_6p-%j.err

set -euo pipefail

COTRA_ROOT="/gpfs/users/delgaudio/exp_eval/CoTra"
cd "$COTRA_ROOT"
mkdir -p script_slurm/logs script_slurm/runtime

# shellcheck disable=SC1091
source script_slurm/env_slurm.sh

export COTRA_MACHINE_NUM=6
export COTRA_CONF_DIR="${COTRA_CONF_DIR:-${COTRA_ROOT}/script_slurm/runtime}"

COTRA_DATA_PATH="${COTRA_DATA_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.fbin}"
COTRA_INDEX_OUT="${COTRA_INDEX_OUT:-/gpfs/scratch/delgaudio/cotra/index/deep10M_6p}"
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

echo "==== scala_index_6p DEEP ${COTRA_MILLION}M L2 start $(date -Is) job=${SLURM_JOB_ID} ===="
echo "nodes=${SLURM_JOB_NODELIST} MACHINE_NUM=${COTRA_MACHINE_NUM}"
echo "DATA=${COTRA_DATA_PATH}"
echo "INDEX_PREFIX=${COTRA_INDEX_PREFIX}"
echo "IB_DEV=${COTRA_IB_DEV:-mlx5_0}"
echo "BIN=${BIN}"

test -x "$BIN" || {
  echo "FATAL: $BIN not found. Run: bash script_slurm/build_cotra_6p.sh" >&2
  exit 1
}
test -f "$COTRA_DATA_PATH" || { echo "FATAL: missing $COTRA_DATA_PATH" >&2; exit 1; }
mkdir -p "$COTRA_INDEX_OUT"

source script_slurm/generate_cluster_conf.sh
generate_cluster_conf
bash script_slurm/start_memcached_leader.sh

echo
echo "==== verify IP mapping ===="
srun --ntasks-per-node=1 --cpus-per-task=1 \
  bash "${COTRA_ROOT}/script_slurm/verify_ip_config.sh"

echo
echo "==== launching scala_index_6p on 6 nodes ===="
srun --ntasks-per-node=1 --cpus-per-task="${COTRA_THREADS}" --kill-on-bad-exit=1 \
  "$BIN" \
  -d "${COTRA_IB_DEV:-mlx5_0}" \
  --config_file "$CONF_FILE" \
  --graph_type scalagraph_v3 \
  --data_type float \
  --dist_fn l2 \
  --data_path "$COTRA_DATA_PATH" \
  --index_path_prefix "$COTRA_INDEX_PREFIX" \
  -R "$COTRA_R" -L "$COTRA_L" -B "$COTRA_B" -M "$COTRA_M" \
  -T "$COTRA_THREADS" \
  --scala_v3 \
  -s "$COTRA_MILLION" \
  -t "$COTRA_THREADS"

echo
echo "==== index artifacts ===="
ls -lh "${COTRA_INDEX_PREFIX}"_top.index 2>/dev/null || true
for i in 0 1 2 3 4 5; do
  ls -lh "${COTRA_INDEX_PREFIX}${i}_final_scala.index" \
        "${COTRA_INDEX_PREFIX}${i}_final_scala.data" 2>/dev/null || true
done
echo "==== done $(date -Is) ===="
