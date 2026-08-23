#!/bin/bash
# Slurm job: build CoTra .conf from allocated nodes + start memcached on leader.
# Does NOT run scala_index yet — validates the cluster setup path.
#
# Usage (login node):
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   mkdir -p script_slurm/logs script_slurm/runtime
#   module purge && module load gcc/10.3.0
#   sbatch script_slurm/setup_cluster.slurm
#
# Must match compile-time MACHINE_NUM in include/rdma/rdma_config.h (default 4).

#SBATCH -J cotra_setup
#SBATCH -p ncpumshort
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH --mem=2G
#SBATCH -o /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/setup_cluster-%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/setup_cluster-%j.err

set -euo pipefail

COTRA_ROOT="/gpfs/users/delgaudio/exp_eval/CoTra"
cd "$COTRA_ROOT"
mkdir -p script_slurm/logs script_slurm/runtime

# shellcheck disable=SC1091
source script_slurm/env_slurm.sh
# shellcheck disable=SC1091
source script_slurm/generate_cluster_conf.sh

export COTRA_MACHINE_NUM=4

echo "==== setup_cluster start $(date -Is) job=${SLURM_JOB_ID} ===="
generate_cluster_conf

# Batch script runs on the first allocated node; start memcached there if it is the leader.
bash script_slurm/start_memcached_leader.sh

echo
echo "==== verify hostname + IP mapping on each node ===="
export CONF_FILE
srun --ntasks-per-node=1 --cpus-per-task=1 \
  bash "${COTRA_ROOT}/script_slurm/verify_node_conf.sh"
srun --ntasks-per-node=1 --cpus-per-task=1 \
  bash "${COTRA_ROOT}/script_slurm/verify_ip_config.sh"

echo "==== setup_cluster end $(date -Is) ===="
echo "Saved conf: ${CONF_FILE}"
echo "Use this CONF_FILE in scala_index / scala_anns with --config_file"
