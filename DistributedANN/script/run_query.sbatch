#!/usr/bin/bash
#SBATCH --job-name=dann-query
#SBATCH -p ncpushort
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.log

# ===========================================================================
# DistributedANN — distributed query answering (Algorithms 1 & 2).
# Requires a completed indexing run (INDEX_DIR with nodes.bin, pq, head).
#
# Submit (pass -N/-n to match index shard count W):
#   sbatch -N 4 -n 4 --export=ALL,... script/run_query.sbatch
#
# Override via environment, e.g.:
#   K=10 BW=128 H=5 NUM_QUERIES=0 sbatch --export=ALL,-N 4 -n 4 script/run_query.sbatch
# ===========================================================================

set -euo pipefail

module purge
module load gcc/10.3.0
module load openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_smoke_1m}"
QUERY_PATH="${QUERY_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin}"
GT_PATH="${GT_PATH:-/gpfs/scratch/delgaudio/dann/test/gt_1m_100q_k10.bin}"
BUILD_DIR="${BUILD_DIR:-/gpfs/users/delgaudio/exp_eval/DistributedANN/build}"

K="${K:-10}"
L="${L:-200}"
BW="${BW:-128}"
H="${H:-5}"
K_HEAD="${K_HEAD:-100}"
EF_HEAD="${EF_HEAD:-200}"
WARMUP="${WARMUP:-10}"
NUM_QUERIES="${NUM_QUERIES:-0}"
DATASET="${DATASET:-deep}"
PARTITIONS="${PARTITIONS:-${SLURM_NTASKS:-2}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

NT="${SLURM_NTASKS}"
_HOSTS=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | awk '{printf "%s:1,",$1}' | sed 's/,$//')
MPI=(mpirun -np "${NT}" --host "${_HOSTS}" --map-by ppr:1:node --bind-to none)

echo "=== DistributedANN query (distributed) ==="
echo "jobid       : ${SLURM_JOB_ID}"
echo "dataset     : ${DATASET}"
echo "partitions  : ${PARTITIONS} (W=${NT})"
echo "nodes       : ${SLURM_JOB_NODELIST}"
echo "ntasks      : ${NT}  omp=${OMP_NUM_THREADS}"
echo "launcher    : ${MPI[*]}"
echo "INDEX_DIR   : ${INDEX_DIR}"
echo "QUERY_PATH  : ${QUERY_PATH}"
echo "GT_PATH     : ${GT_PATH}"
echo "k=${K} L=${L} BW=${BW} H=${H} k_head=${K_HEAD} ef_head=${EF_HEAD}"
echo "num_queries : ${NUM_QUERIES} (0=all)  warmup=${WARMUP}"
echo "=========================================="

for f in "${INDEX_DIR}/nodes.bin" "${INDEX_DIR}/pq_codebook.bin" \
         "${INDEX_DIR}/head/head.hnsw" "${BUILD_DIR}/bin/run_query" \
         "${QUERY_PATH}"; do
  test -f "$f" || { echo "FATAL: missing $f"; exit 1; }
done
if [[ -n "${GT_PATH}" ]]; then
  test -f "${GT_PATH}" || { echo "FATAL: missing GT ${GT_PATH}"; exit 1; }
fi

echo
echo "=== PREFLIGHT: MPI wireup ==="
WSIZE=$("${MPI[@]}" "${BUILD_DIR}/bin/mpi_size_test" | sed -n 's/.*size=\([0-9]*\).*/\1/p' | head -1)
echo "MPI_COMM_WORLD size = ${WSIZE} (expected ${NT})"
if [[ "${WSIZE}" != "${NT}" ]]; then
  echo "FATAL: MPI wireup failed (size ${WSIZE} != ${NT})"
  exit 2
fi

GT_ARGS=()
if [[ -n "${GT_PATH}" ]]; then
  GT_ARGS=(--gt_path "${GT_PATH}")
fi

echo
echo "=== QUERY RUN ==="
T_START=$(date +%s)
"${MPI[@]}" "${BUILD_DIR}/bin/run_query" \
    --index_dir "${INDEX_DIR}" \
    --query_path "${QUERY_PATH}" \
    "${GT_ARGS[@]}" \
    --k "${K}" --L "${L}" --BW "${BW}" --H "${H}" \
    --k_head "${K_HEAD}" --ef_head "${EF_HEAD}" \
    --warmup "${WARMUP}" --num_queries "${NUM_QUERIES}"
T_END=$(date +%s)
WALL=$((T_END - T_START))
echo
echo "=== JOB WALL TIME (query phase) ==="
echo "  wall_s      : ${WALL}"
