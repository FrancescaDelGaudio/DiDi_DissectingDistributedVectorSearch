#!/bin/bash
# Smoke su 1 solo nodo fisico: 2 rank MPI (partizione k=2, dati 100k).
# Verifica moduli + binario release_l2 prima del job a 8 nodi.
#
# Uso:
#   sbatch /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_distributed_deep_10M_1node_test.sh
#
# Log:
#   /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_logs/dist_deep10m_1n_test-%j.out

#SBATCH -J dist_1n_smoke
#SBATCH -p ncpushort
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_logs/dist_deep10m_1n_test-%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_logs/dist_deep10m_1n_test-%j.out

set -euo pipefail

mkdir -p /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_logs

echo "==== 1-node smoke start $(date -Is) jobid=${SLURM_JOB_ID:-?} host=$(hostname) ===="

GP_ANN_ROOT="${GP_ANN_ROOT:-${SLURM_SUBMIT_DIR:-/gpfs/users/delgaudio/exp_eval/gp-ann}}"
# shellcheck disable=SC1091
source "${GP_ANN_ROOT}/scripts/slurm_env_mpi_gcc11.sh"
slurm_env_mpi_gcc11
slurm_mpi_export_flags

DATA_DIR="/gpfs/users/delgaudio/exp_eval/gp-ann/data/smoke_100k"
BENCH="/gpfs/users/delgaudio/exp_eval/gp-ann/release_l2/DistributedBench"

test -x "$BENCH"

echo "rank/host check:"
mpirun -n "${SLURM_NTASKS}" "${MPIRUN_EXPORT[@]}" \
  bash -lc 'echo "rank=${OMPI_COMM_WORLD_RANK} host=$(hostname)"'

echo "========== DistributedBench (smoke 100k, k=10) =========="
mpirun -n "${SLURM_NTASKS}" "${MPIRUN_EXPORT[@]}" \
  "$BENCH" \
  "$DATA_DIR/random_100k.fbin" \
  "$DATA_DIR/random_100k.query.50.fbin" \
  "$DATA_DIR/random_100k.query.50.ground_truth.bin" \
  10 \
  "$DATA_DIR/random_100k.partition.k=2.GP" \
  "$DATA_DIR/random_100k.partition.k=2.GP.routing_index"

echo "==== done $(date -Is) ===="
