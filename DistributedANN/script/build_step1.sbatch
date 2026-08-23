#!/usr/bin/bash
#SBATCH --job-name=dann-step1
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.log

# DistributedANN — Step 1: k-means clustering + epsilon-replication
# Overridable via --export or environment variables.

module purge
module load gcc/10.3.0
module load openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BASE_PATH="${BASE_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid/base.fvecs}"
INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_deep10m_k${K:-32}}"
BUILD_DIR="${BUILD_DIR:-/gpfs/users/delgaudio/exp_eval/DistributedANN/build}"

K="${K:-32}"
EPSILON="${EPSILON:-0.2}"
MAX_REPLICAS="${MAX_REPLICAS:-3}"
KMEANS_ITER="${KMEANS_ITER:-20}"
METRIC="${METRIC:-l2}"

echo "=== DistributedANN Step 1 ==="
echo "NODES         : ${SLURM_JOB_NUM_NODES}"
echo "NTASKS        : ${SLURM_NTASKS}"
echo "BASE_PATH     : ${BASE_PATH}"
echo "INDEX_DIR     : ${INDEX_DIR}"
echo "K             : ${K}"
echo "EPSILON       : ${EPSILON}"
echo "MAX_REPLICAS  : ${MAX_REPLICAS}"
echo "KMEANS_ITER   : ${KMEANS_ITER}"
echo "=============================="

mpirun -n "${SLURM_NTASKS}" \
    "${BUILD_DIR}/bin/build_index" \
    --base_path    "${BASE_PATH}"    \
    --index_dir    "${INDEX_DIR}"    \
    --K            "${K}"            \
    --kmeans_iter  "${KMEANS_ITER}"  \
    --kmeans_tol   "${KMEANS_TOL}"   \
    --epsilon      "${EPSILON}"      \
    --max_replicas "${MAX_REPLICAS}" \
    --metric       "${METRIC}"
