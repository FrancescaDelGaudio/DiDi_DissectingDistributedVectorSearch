#!/usr/bin/bash
#SBATCH --job-name=dann-step3
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.log

# DistributedANN — Step 3: global graph stitching (union of neighbour lists + RobustPrune).
# Prerequisite: Step 2 cluster graphs present in INDEX_DIR/clusters/.

module purge
module load gcc/10.3.0
module load openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BASE_PATH="${BASE_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid/base.fvecs}"
INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_deep10m_k${K:-32}}"
BUILD_DIR="${BUILD_DIR:-/gpfs/users/delgaudio/exp_eval/DistributedANN/build}"

R="${R:-72}"
ALPHA="${ALPHA:-1.2}"

echo "=== DistributedANN Step 3 ==="
echo "NODES     : ${SLURM_JOB_NUM_NODES}"
echo "BASE_PATH : ${BASE_PATH}"
echo "INDEX_DIR : ${INDEX_DIR}"
echo "R / alpha : ${R} / ${ALPHA}"
echo "============================="

mpirun -n "${SLURM_NTASKS}" \
    "${BUILD_DIR}/bin/build_stitch" \
    --base_path "${BASE_PATH}" \
    --index_dir "${INDEX_DIR}" \
    --R "${R}" --alpha "${ALPHA}"
