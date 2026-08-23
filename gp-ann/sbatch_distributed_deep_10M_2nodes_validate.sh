#!/bin/bash
# Stesso esperimento DEEP 10M dell'8-nodi, ma su 2 nodi fisici (8 rank MPI, 4/nodo).
# Serve a verificare moduli + binario + I/O prima di sbatch a 8 nodi.
#
# Uso:
#   sbatch /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_distributed_deep_10M_2nodes_validate.sh

#SBATCH -J dist_deep10m_2n
#SBATCH -p ncpushort
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_10M_2n_validate-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_10M_2n_validate-%j.log

set -euo pipefail

log() { echo "[$(date -Is)] $*" >&2; }

log "==== DEEP 10M validate (2 nodes, 8 ranks) jobid=${SLURM_JOB_ID:-?} ===="
log "NODELIST=${SLURM_NODELIST:-?}"

# Slurm copia lo script in /var/spool/slurmd/...: non usare dirname $0.
GP_ANN_ROOT="${GP_ANN_ROOT:-${SLURM_SUBMIT_DIR:-/gpfs/users/delgaudio/exp_eval/gp-ann}}"
log "GP_ANN_ROOT=${GP_ANN_ROOT}"
# shellcheck disable=SC1091
source "${GP_ANN_ROOT}/scripts/slurm_env_mpi_gcc11.sh"
log "loading MPI modules..."
slurm_env_mpi_gcc11
slurm_mpi_export_flags
log "mpirun=$(command -v mpirun) ($(mpirun --version 2>&1 | head -1))"

BENCH="/gpfs/users/delgaudio/exp_eval/gp-ann/release_l2/DistributedBench"
BASE="/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.fbin"
QUERY="/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin"
GT="/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep10M_10kqueries_k100_faiss.bin"
PART="/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.metis"
ROUTER="/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.routing_index"
K=100

test -x "$BENCH"
for f in "$BASE" "$QUERY" "$GT" "$PART" "$ROUTER"; do test -f "$f"; done
log "inputs OK; starting mpirun (load 10M + 10k queries: silenzioso fino al recall, anche 20-40 min)"

mpirun -n 8 "${MPIRUN_EXPORT[@]}" \
  --map-by ppr:4:node \
  --bind-to none \
  "$BENCH" "$BASE" "$QUERY" "$GT" "$K" "$PART" "$ROUTER"

log "==== done ===="
