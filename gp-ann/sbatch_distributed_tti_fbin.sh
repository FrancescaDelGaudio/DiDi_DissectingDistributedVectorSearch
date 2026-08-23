#!/bin/bash
# DistributedBench on TTI 10M (MIPS / inner product), one overlapping GP partition.
#
# Submit with -N/-n = wc -l(PARTITION_FILE) (see submit_new_experiments_distbench_tti.py).
#
# Required env (via sbatch --export):
#   PARTITION_FILE, RECALL_K, BENCH, SHARDS_PROBE
# Optional:
#   BASE_FBIN, QUERY_FBIN, GT_BIN, GP_ANN_ROOT, OUTPUT_ROOT
#
# Outputs (set by submit_new_experiments_distbench_tti.py):
#   ${OUTPUT_ROOT}/logs/<jobname>_<jobid>.log
# Default OUTPUT_ROOT:
#   /gpfs/scratch/delgaudio/vectors/TTI/new_experiments_distributedbench

#SBATCH -p ncpushort
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00

set -euo pipefail

: "${PARTITION_FILE:?PARTITION_FILE required}"
: "${RECALL_K:?RECALL_K required}"
: "${BENCH:?BENCH required}"
: "${SHARDS_PROBE:?SHARDS_PROBE required}"

GP_ANN_ROOT="${GP_ANN_ROOT:-${SLURM_SUBMIT_DIR:-/gpfs/users/delgaudio/exp_eval/gp-ann}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gpfs/scratch/delgaudio/vectors/TTI/new_experiments_distributedbench}"
mkdir -p "${OUTPUT_ROOT}/logs"
BASE_FBIN="${BASE_FBIN:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2i10M.fbin}"
QUERY_FBIN="${QUERY_FBIN:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2iquery10K.fbin}"
GT_BIN="${GT_BIN:-/gpfs/scratch/delgaudio/vectors/TTI/computedGT_tti10M_10kqueries_k100_faiss.bin}"
ROUTER_INDEX="${ROUTER_INDEX:-${PARTITION_FILE}.routing_index}"

log() { echo "[$(date -Is)] $*" >&2; }

NUM_SHARDS=$(wc -l < "$PARTITION_FILE")
NUM_SHARDS="${NUM_SHARDS//[[:space:]]/}"

if [[ -z "${SLURM_NTASKS:-}" ]] || [[ "${SLURM_NTASKS}" -ne "${NUM_SHARDS}" ]]; then
  log "FATAL: SLURM_NTASKS=${SLURM_NTASKS:-?} but partition has ${NUM_SHARDS} shards"
  exit 1
fi

log "==== TTI DistributedBench (MIPS) jobid=${SLURM_JOB_ID:-?} ===="
log "NODELIST=${SLURM_NODELIST:-?} NNODES=${SLURM_NNODES:-?} NTASKS=${SLURM_NTASKS}"
log "PARTITION_FILE=$PARTITION_FILE"
log "ROUTER_INDEX=$ROUTER_INDEX"
log "RECALL_K=$RECALL_K SHARDS_PROBE(compiled)=$SHARDS_PROBE num_shards=$NUM_SHARDS"
log "BENCH=$BENCH"

# shellcheck disable=SC1091
source "${GP_ANN_ROOT}/scripts/slurm_env_mpi_gcc11.sh"
slurm_env_mpi_gcc11
slurm_mpi_export_flags
log "mpirun=$(command -v mpirun)"

test -x "$BENCH" || { log "FATAL: not executable: $BENCH"; exit 1; }
for f in "$BASE_FBIN" "$QUERY_FBIN" "$GT_BIN" "$PARTITION_FILE" "$ROUTER_INDEX"; do
  test -f "$f" || { log "FATAL: missing: $f"; exit 1; }
done
test -f "${ROUTER_INDEX}.routing_index_partition" || {
  log "FATAL: missing router sidecar: ${ROUTER_INDEX}.routing_index_partition"
  exit 1
}

log "inputs OK; starting mpirun (quiet until recall lines)"

mpirun -n "${SLURM_NTASKS}" "${MPIRUN_EXPORT[@]}" \
  --map-by ppr:1:node \
  --bind-to none \
  "$BENCH" \
  "$BASE_FBIN" \
  "$QUERY_FBIN" \
  "$GT_BIN" \
  "$RECALL_K" \
  "$PARTITION_FILE" \
  "$ROUTER_INDEX"

log "==== done ===="
