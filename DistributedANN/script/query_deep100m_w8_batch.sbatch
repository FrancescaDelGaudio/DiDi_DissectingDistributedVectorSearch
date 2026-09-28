#!/usr/bin/bash
#SBATCH --job-name=dann-batch-d100m
#SBATCH -p ncpum
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH -t 8:00:00
#SBATCH -o /gpfs/scratch/delgaudio/dann/logs/query_batch/deep100m_w8/dann-batch-d100m_%j.log
#SBATCH -e /gpfs/scratch/delgaudio/dann/logs/query_batch/deep100m_w8/dann-batch-d100m_%j.log
#
# DA-batch on DEEP100M W=8 (branch da-batch, build_batch/). Same index, queries,
# GT and H/k_head/ef_head/L rule as the run_query H=10 sweeps.
# Grid: for each K, BW, THREADS, CHUNK -> one run_query_batch invocation.
#   THREADS_LIST  default "1 16"      (T=1 reference, T=16 = CoTra/Harmony setting)
#   CHUNK_LIST    default "0"         (0 = all nq/W queries in flight)
# Chunk mini-sweep: CHUNK_LIST="64 256 0" BW_LIST=128 THREADS_LIST=16 sbatch ...

set -euo pipefail
module purge
module load gcc/10.3.0
module load openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_deep100m_k256_w8}"
QUERY_PATH="${QUERY_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin}"
GT_PATH="${GT_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/GT.100M.bin}"
BUILD_DIR="${BUILD_DIR:-/gpfs/users/delgaudio/exp_eval_da_batch/DistributedANN/build_batch}"

H="${H:-10}"
K_HEAD="${K_HEAD:-200}"
EF_HEAD="${EF_HEAD:-400}"
WARMUP="${WARMUP:-10}"
NUM_QUERIES="${NUM_QUERIES:-0}"
read -r -a K_ARR       <<< "${K_LIST:-10}"
read -r -a BW_ARR      <<< "${BW_LIST:-32 128 512 1024}"
read -r -a THREADS_ARR <<< "${THREADS_LIST:-1 16}"
read -r -a CHUNK_ARR   <<< "${CHUNK_LIST:-0}"

NT="${SLURM_NTASKS}"
_HOSTS=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | awk '{printf "%s:1,",$1}' | sed 's/,$//')
MPI=(mpirun -np "${NT}" --host "${_HOSTS}" --map-by ppr:1:node --bind-to none)

echo "=== DA-batch DEEP100M W=${NT} ==="
echo "jobid=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
echo "K=${K_ARR[*]} BW=${BW_ARR[*]} THREADS=${THREADS_ARR[*]} CHUNK=${CHUNK_ARR[*]} H=${H} k_head=${K_HEAD} ef_head=${EF_HEAD}"
echo "BUILD_DIR=${BUILD_DIR}"

for f in "${INDEX_DIR}/nodes.bin" "${INDEX_DIR}/head/head.hnsw" "${INDEX_DIR}/pq_codebook.bin" \
         "${GT_PATH}" "${QUERY_PATH}" "${BUILD_DIR}/bin/run_query_batch" "${BUILD_DIR}/bin/mpi_size_test"; do
  [[ -e "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done
WSIZE=$("${MPI[@]}" "${BUILD_DIR}/bin/mpi_size_test" | sed -n 's/.*size=\([0-9]*\).*/\1/p' | head -1)
[[ "${WSIZE}" == "${NT}" ]] || { echo "FATAL: MPI wireup failed (${WSIZE} != ${NT})"; exit 2; }

for K in "${K_ARR[@]}"; do
  for BW in "${BW_ARR[@]}"; do
    if [[ "${BW}" -le 256 ]]; then L=400; else L="${BW}"; fi
    for T in "${THREADS_ARR[@]}"; do
      for C in "${CHUNK_ARR[@]}"; do
        echo
        echo "========== RUN K=${K} BW=${BW} L=${L} THREADS=${T} CHUNK=${C} =========="
        t0=$(date +%s)
        OMP_NUM_THREADS="${T}" "${MPI[@]}" -x OMP_NUM_THREADS "${BUILD_DIR}/bin/run_query_batch" \
            --index_dir "${INDEX_DIR}" --query_path "${QUERY_PATH}" --gt_path "${GT_PATH}" \
            --k "${K}" --L "${L}" --BW "${BW}" --H "${H}" \
            --k_head "${K_HEAD}" --ef_head "${EF_HEAD}" \
            --warmup "${WARMUP}" --num_queries "${NUM_QUERIES}" \
            --threads "${T}" --chunk "${C}"
        echo "WALL_K${K}_BW${BW}_T${T}_C${C}_s=$(( $(date +%s) - t0 ))"
      done
    done
  done
done
echo "=== Done. grep SWEEP_SUMMARY for results. ==="
