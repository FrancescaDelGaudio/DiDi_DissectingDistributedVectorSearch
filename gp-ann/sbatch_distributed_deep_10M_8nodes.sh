#!/bin/bash
# DistributedBench on DEEP 10M, 8 nodi (1 rank/nodo), binario release_l2.
#
# Uso:
#   sbatch /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_distributed_deep_10M_8nodes.sh
#
# Log:
#   /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_10M-<jobid>.log

#SBATCH -J dist_deep10m_8n
#SBATCH -p ncpushort
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_10M-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_10M-%j.log

set -euo pipefail

log() { echo "[$(date -Is)] $*" >&2; }

log "==== DEEP 10M (8 nodes, 8 ranks) jobid=${SLURM_JOB_ID:-?} ===="
log "NODELIST=${SLURM_NODELIST:-?} NNODES=${SLURM_NNODES:-?} NTASKS=${SLURM_NTASKS:-?}"

GP_ANN_ROOT="${GP_ANN_ROOT:-${SLURM_SUBMIT_DIR:-/gpfs/users/delgaudio/exp_eval/gp-ann}}"
log "GP_ANN_ROOT=${GP_ANN_ROOT}"
# shellcheck disable=SC1091
source "${GP_ANN_ROOT}/scripts/slurm_env_mpi_gcc11.sh"
log "loading MPI modules..."
slurm_env_mpi_gcc11
slurm_mpi_export_flags
log "mpirun=$(command -v mpirun) ($(mpirun --version 2>&1 | head -1))"

BENCH="${BENCH:-/gpfs/users/delgaudio/exp_eval/gp-ann/release_l2/DistributedBench}"
BASE="${BASE:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.fbin}"
QUERY="${QUERY:-/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin}"
GT="${GT:-/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep10M_10kqueries_k100_faiss.bin}"
PART="${PART:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.metis}"
ROUTER="${ROUTER:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.routing_index}"
K="${K:-10}"

test -x "$BENCH"
for f in "$BASE" "$QUERY" "$GT" "$PART" "$ROUTER"; do test -f "$f"; done
log "BENCH=$BENCH K=$K"
log "inputs OK; starting mpirun (lungo silenzio fino al recall)"

mpirun -n "${SLURM_NTASKS}" "${MPIRUN_EXPORT[@]}" \
  --map-by ppr:1:node \
  --bind-to none \
  "$BENCH" \
  "$BASE" \
  "$QUERY" \
  "$GT" \
  "$K" \
  "$PART" \
  "$ROUTER"

log "==== done ===="
