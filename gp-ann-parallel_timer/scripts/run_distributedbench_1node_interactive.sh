#!/usr/bin/env bash
# Esegui DENTRO un allocation Slurm a 1 nodo (salloc), dopo aver compilato release_l2.
#
# Esempio sul login:
#   salloc -p ncpushort -N 1 -n 2 --ntasks-per-node=2 --cpus-per-task=4 -t 00:30:00
#   bash /gpfs/users/delgaudio/exp_eval/gp-ann/scripts/run_distributedbench_1node_interactive.sh
#
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Avvia prima: salloc -p ncpushort -N 1 -n 2 --ntasks-per-node=2 -t 00:30:00" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/slurm_env_mpi_gcc11.sh"
slurm_env_mpi_gcc11
slurm_mpi_export_flags

DATA_DIR="/gpfs/users/delgaudio/exp_eval/gp-ann/data/smoke_100k"
BENCH="/gpfs/users/delgaudio/exp_eval/gp-ann/release_l2/DistributedBench"

test -x "$BENCH"

mpirun -n "${SLURM_NTASKS:-2}" "${MPIRUN_EXPORT[@]}" \
  bash -lc 'echo "rank=${OMPI_COMM_WORLD_RANK} host=$(hostname)"'

exec mpirun -n "${SLURM_NTASKS:-2}" "${MPIRUN_EXPORT[@]}" \
  "$BENCH" \
  "$DATA_DIR/random_100k.fbin" \
  "$DATA_DIR/random_100k.query.50.fbin" \
  "$DATA_DIR/random_100k.query.50.ground_truth.bin" \
  10 \
  "$DATA_DIR/random_100k.partition.k=2.GP" \
  "$DATA_DIR/random_100k.partition.k=2.GP.routing_index"
