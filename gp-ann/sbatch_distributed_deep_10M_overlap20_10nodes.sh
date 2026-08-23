#!/bin/bash
# DistributedBench on DEEP 10M, overlapping partition (20%), 10 MPI ranks / 10 nodes.
#
# Prerequisite:
#   base.10M.partition.k=8.GP.o=0.2          (wc -l → 10 shards)
#   base.10M.partition.k=8.GP.o=0.2.routing_index
#   QueryAttribution qa_o20 (job 2762247 or equivalent)
#
# Usage (from gp-ann repo):
#   sbatch /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_distributed_deep_10M_overlap20_10nodes.sh
#
# Log:
#   /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_10M_o20-%j.log

#SBATCH -J dist_10m_o20_10n
#SBATCH -p ncpushort
#SBATCH -N 10
#SBATCH -n 10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_10M_o20-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_10M_o20-%j.log

set -euo pipefail

log() { echo "[$(date -Is)] $*" >&2; }

log "==== DEEP 10M overlap o20 (10 nodes, 10 ranks) jobid=${SLURM_JOB_ID:-?} ===="
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
PART="${PART:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.o=0.2}"
ROUTER="${ROUTER:-/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.partition.k=8.GP.o=0.2.routing_index}"
K="${K:-10}"

test -x "$BENCH" || { log "FATAL: not executable: $BENCH"; exit 1; }
for f in "$BASE" "$QUERY" "$GT" "$PART" "$ROUTER"; do
  test -f "$f" || { log "FATAL: missing file: $f"; exit 1; }
done
test -f "${ROUTER}.routing_index_partition" || {
  log "FATAL: missing router sidecar: ${ROUTER}.routing_index_partition"
  exit 1
}

NUM_SHARDS=$(wc -l < "$PART")
if [[ "${SLURM_NTASKS}" -ne "${NUM_SHARDS}" ]]; then
  log "FATAL: SLURM_NTASKS=${SLURM_NTASKS} but partition has ${NUM_SHARDS} shards (use -N ${NUM_SHARDS} -n ${NUM_SHARDS})"
  exit 1
fi

log "BENCH=$BENCH K=$K num_shards=$NUM_SHARDS (num_probes=5 in binary)"
log "PART=$PART"
log "ROUTER=$ROUTER"
log "inputs OK; starting mpirun (long quiet period until recall)"

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
