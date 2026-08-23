#!/bin/bash
# DistributedBench (MIPS / inner product) on TTI 10M: overlapping partition 20%, 10 MPI ranks on 10 physical nodes.
#
# Prerequisites:
#   Partition (MIPS): yangt2i10M.partition.k=8.GP.o=0.2  (typically wc -l == 10)
#   QueryAttribution (MIPS): yangt2i10M.qa_o20.* + .routing_index
#
# Usage:
#   sbatch /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_distributed_tti10M_overlap20_10nodes.sh
#
# Overrides (environment):
#   BENCH BASE QUERY GT PART ROUTER K
#
# Log:
#   /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_tti10M_o20_%j.log
#
# Note: Queries are 10K × dim 200. Default K=100 ⇒ recall@100 (eval uses first K neighbors from GT).
#   num_probes (max shards queried per query) defaults to 9 in src/distributed_query_benchmark.h — rebuild DistributedBench after changing it.
#
# Queue: default is ncpushort (2h wall) for a short first run. If the job hits the time
# limit or the partition is busy, switch to ncpum and a longer -t (see comments in file).

#SBATCH -J dist_tti10m_o20_10n
#SBATCH -p ncpushort
#SBATCH -N 10
#SBATCH -n 10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_tti10M_o20-%j.log
#SBATCH -e /gpfs/scratch/delgaudio/gp_ann/slurm_distributedbench_tti10M_o20-%j.log

set -euo pipefail

log() { echo "[$(date -Is)] $*" >&2; }

log "==== TTI 10M MIPS overlap o20 (10 nodes, 10 ranks) jobid=${SLURM_JOB_ID:-?} ===="
log "NODELIST=${SLURM_NODELIST:-?} NNODES=${SLURM_NNODES:-?} NTASKS=${SLURM_NTASKS:-?}"

# Repo root for scripts/ (do not use SLURM_SUBMIT_DIR alone: sbatch may be run from a build dir).
_GP_ANN_DEFAULT="/gpfs/users/delgaudio/exp_eval/gp-ann"
if [[ -n "${GP_ANN_ROOT:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm_env_mpi_gcc11.sh" ]]; then
  GP_ANN_ROOT="$SLURM_SUBMIT_DIR"
else
  GP_ANN_ROOT="$_GP_ANN_DEFAULT"
fi
unset _GP_ANN_DEFAULT
log "GP_ANN_ROOT=${GP_ANN_ROOT}"
# shellcheck disable=SC1091
source "${GP_ANN_ROOT}/scripts/slurm_env_mpi_gcc11.sh"
log "loading MPI modules..."
slurm_env_mpi_gcc11
slurm_mpi_export_flags
log "mpirun=$(command -v mpirun) ($(mpirun --version 2>&1 | head -1))"

BENCH="${BENCH:-${GP_ANN_ROOT}/release_mips_distbench/DistributedBench}"
BASE="${BASE:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2i10M.fbin}"
QUERY="${QUERY:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2iquery10K.fbin}"
GT="${GT:-/gpfs/scratch/delgaudio/vectors/TTI/computedGT_tti10M_10kqueries_k100_faiss.bin}"
PART="${PART:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2i10M.partition.k=8.GP.o=0.2}"
ROUTER="${ROUTER:-/gpfs/scratch/delgaudio/vectors/TTI/yangt2i10M.partition.k=8.GP.o=0.2.routing_index}"
K="${K:-100}"

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

log "BENCH=$BENCH (MIPS build) recall_K=$K num_shards=${NUM_SHARDS} (num_probes=9 compiled in DistributedBench)"
log "PART=$PART"
log "ROUTER=$ROUTER"
log "inputs OK; starting mpirun (possible long quiet period during load/search)"

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
