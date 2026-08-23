#!/usr/bin/bash
#SBATCH --job-name=dann-step4
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.log

# DistributedANN — Step 4: PQ codebook + per-vector PQ codes.
# Outputs: INDEX_DIR/pq_codebook.bin, INDEX_DIR/codes.bin

module purge
module load gcc/10.3.0
module load openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BASE_PATH="${BASE_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid/base.fvecs}"
INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_deep10m_k${K:-32}}"
BUILD_DIR="${BUILD_DIR:-/gpfs/users/delgaudio/exp_eval/DistributedANN/build}"

M="${M:-16}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
ITERS="${ITERS:-25}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== DistributedANN Step 4 (PQ) ==="
echo "INDEX_DIR  : ${INDEX_DIR}"
echo "m=${M}  train_size=${TRAIN_SIZE}  iters=${ITERS}"
echo "=================================="

mpirun -n "${SLURM_NTASKS}" \
    "${BUILD_DIR}/bin/build_pq" \
    --base_path "${BASE_PATH}" \
    --index_dir "${INDEX_DIR}" \
    --m "${M}" --train_size "${TRAIN_SIZE}" --iters "${ITERS}"
