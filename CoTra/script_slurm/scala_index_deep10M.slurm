#!/bin/bash
# Distributed CoTra indexing (ScalaGraph V3) on DEEP 10M, 4 partitions / 4 nodes.
#
# Prerequisites:
#   - MACHINE_NUM=4 in include/rdma/rdma_config.h (matches #SBATCH -N 4)
#   - build/tests/scala_index (bash script_slurm/build_cotra.sh)
#   - bash script_slurm/build_cotra_deps.sh (memcached on compute nodes)
#
# Usage (login node):
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   mkdir -p script_slurm/logs script_slurm/runtime
#   sbatch script_slurm/scala_index_deep10M.slurm
#
# Optional overrides at submit time:
#   sbatch --export=ALL,COTRA_INDEX_OUT=/gpfs/scratch/delgaudio/cotra/index/my_run \
#     script_slurm/scala_index_deep10M.slurm
#
# Outputs (under COTRA_INDEX_OUT, default .../cotra/index/deep10M):
#   merged_index{0..3}_final_scala.index / .data
#   merged_index_top.index
#   merged_index_temp_{pos2label,label2pos,id2part}.map
#   cluster-job<JOBID>.conf

#SBATCH -J cotra_idx10M
#SBATCH -p ncpum
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 08:00:00
#SBATCH --mem=150G
#SBATCH -o /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/scala_index_deep10M-%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/scala_index_deep10M-%j.err

set -euo pipefail

COTRA_ROOT="/gpfs/users/delgaudio/exp_eval/CoTra"
cd "$COTRA_ROOT"
mkdir -p script_slurm/logs script_slurm/runtime

# shellcheck disable=SC1091
source script_slurm/env_slurm.sh

export COTRA_MACHINE_NUM=4
export COTRA_CONF_DIR="${COTRA_CONF_DIR:-${COTRA_ROOT}/script_slurm/runtime}"

COTRA_DATA_PATH="${COTRA_DATA_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.fbin}"
COTRA_INDEX_OUT="${COTRA_INDEX_OUT:-/gpfs/scratch/delgaudio/cotra/index/deep10M}"
COTRA_INDEX_PREFIX="${COTRA_INDEX_PREFIX:-${COTRA_INDEX_OUT}/merged_index}"
COTRA_MILLION="${COTRA_MILLION:-10}"
COTRA_THREADS="${COTRA_THREADS:-32}"
COTRA_R="${COTRA_R:-48}"
COTRA_L="${COTRA_L:-500}"
COTRA_B="${COTRA_B:-100}"
COTRA_M="${COTRA_M:-120}"

FAISS_LIB="/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/third_party/faiss/build-gcc10/faiss"
export LD_LIBRARY_PATH="${FAISS_LIB}:${LD_LIBRARY_PATH:-}"

BIN="${COTRA_ROOT}/build/tests/scala_index"

echo "==== scala_index DEEP ${COTRA_MILLION}M start $(date -Is) job=${SLURM_JOB_ID} ===="
echo "nodes=${SLURM_JOB_NODELIST} ntasks=${SLURM_NTASKS:-?}"
echo "DATA=${COTRA_DATA_PATH}"
echo "INDEX_PREFIX=${COTRA_INDEX_PREFIX}"
echo "IB_DEV=${COTRA_IB_DEV:-mlx5_0}"
echo "BIN=${BIN}"

test -x "$BIN" || {
  echo "FATAL: scala_index not found. Run: bash script_slurm/build_cotra.sh" >&2
  exit 1
}
test -f "$COTRA_DATA_PATH" || {
  echo "FATAL: base dataset missing: $COTRA_DATA_PATH" >&2
  exit 1
}

mkdir -p "$COTRA_INDEX_OUT"

# shellcheck disable=SC1091
source script_slurm/generate_cluster_conf.sh
generate_cluster_conf

bash script_slurm/start_memcached_leader.sh

echo
echo "==== verify IP mapping (CoTra get_machine_id) ===="
srun --ntasks-per-node=1 --cpus-per-task=1 \
  bash "${COTRA_ROOT}/script_slurm/verify_ip_config.sh"

echo
echo "==== launching scala_index on 4 nodes ===="
srun --ntasks-per-node=1 --cpus-per-task="${COTRA_THREADS}" --kill-on-bad-exit=1 \
  "$BIN" \
  -d "${COTRA_IB_DEV:-mlx5_0}" \
  --config_file "$CONF_FILE" \
  --graph_type scalagraph_v3 \
  --data_type float \
  --dist_fn l2 \
  --data_path "$COTRA_DATA_PATH" \
  --index_path_prefix "$COTRA_INDEX_PREFIX" \
  -R "$COTRA_R" \
  -L "$COTRA_L" \
  -B "$COTRA_B" \
  -M "$COTRA_M" \
  -T "$COTRA_THREADS" \
  --scala_v3 \
  -s "$COTRA_MILLION" \
  -t "$COTRA_THREADS"

echo
echo "==== index artifacts ===="
ls -lh "${COTRA_INDEX_PREFIX}"_top.index 2>/dev/null || true
for i in 0 1 2 3; do
  ls -lh "${COTRA_INDEX_PREFIX}${i}_final_scala.index" \
        "${COTRA_INDEX_PREFIX}${i}_final_scala.data" 2>/dev/null || true
done
ls -lh "${COTRA_INDEX_PREFIX}"_temp_*.map 2>/dev/null || true

echo "==== scala_index done $(date -Is) exit=$? ===="
echo "CONF_FILE=$CONF_FILE"
echo "Use for scala_anns: --index_path_prefix $COTRA_INDEX_PREFIX --config_file <conf from this job>"
