#!/usr/bin/bash
#SBATCH --job-name=dann-tti-smoke
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH -o /gpfs/scratch/delgaudio/dann/logs/tti_smoke_2n/%x_%j.log
#SBATCH -e /gpfs/scratch/delgaudio/dann/logs/tti_smoke_2n/%x_%j.log

# ===========================================================================
# TTI MIPS end-to-end smoke on 2 physical Slurm nodes.
# Same data as login-node test: 1000 base + 100 queries + IP GT.
# Indexing steps 1-6 + distributed query (W=2).
#
# Submit:  ./script/submit_tti_mips_smoke_2n.sh
# ===========================================================================

set -euo pipefail

ROOT="/gpfs/users/delgaudio/exp_eval/DistributedANN"
LOG_DIR="/gpfs/scratch/delgaudio/dann/logs/tti_smoke_2n"
mkdir -p "${LOG_DIR}"

export BUILD_DIR="${BUILD_DIR:-${ROOT}/build_ip}"
export DATA_DIR="${DATA_DIR:-/gpfs/scratch/delgaudio/dann/tti_smoke_1k}"
export INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_tti_smoke_1k_2n}"
export NT="${SLURM_NTASKS}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export METRIC="${METRIC:-ip}"
export SKIP_BUILD="${SKIP_BUILD:-1}"
export SKIP_DATA="${SKIP_DATA:-1}"
export SKIP_INDEX="${SKIP_INDEX:-0}"

echo "############################################################"
echo "# Slurm job ${SLURM_JOB_ID}"
echo "#   nodes : ${SLURM_JOB_NODELIST}"
echo "#   ntasks: ${SLURM_NTASKS}  omp=${OMP_NUM_THREADS}"
echo "#   INDEX : ${INDEX_DIR}"
echo "#   BUILD : ${BUILD_DIR}"
echo "############################################################"

cd "${ROOT}"
exec bash "${ROOT}/script/run_tti_mips_smoke.sh"
