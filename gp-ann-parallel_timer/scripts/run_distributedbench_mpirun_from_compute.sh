#!/usr/bin/env bash
# Run inside an active Slurm allocation (salloc/sbatch). mpirun starts on the
# first allocated node so ORTE daemons can connect back across the cluster network.
set -euo pipefail

: "${SLURM_JOB_ID:?Run only inside a Slurm job (missing SLURM_JOB_ID)}"
: "${SLURM_JOB_NODELIST:?missing SLURM_JOB_NODELIST}"
: "${SLURM_NNODES:?missing SLURM_NNODES}"

HOSTFILE=/gpfs/scratch/delgaudio/gp_ann/hostfile.${SLURM_JOB_ID}
scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${HOSTFILE}"
FIRST=$(head -n1 "${HOSTFILE}")

export OPAL_PREFIX=/gpfs/scratch/delgaudio/openmpi_usr_mirror
export PATH="${OPAL_PREFIX}/bin:${PATH}"

export NNODES="$SLURM_NNODES"
export JBID="$SLURM_JOB_ID"

BENCH_BIN="${BENCH_BIN:-/gpfs/scratch/delgaudio/gp_ann/bin/DistributedBench}"
BASE_FBIN="${BASE_FBIN:-/gpfs/scratch/delgaudio/vectors/DEEP/learn.350M.fbin}"
QUERY_FBIN="${QUERY_FBIN:-/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin}"
GT_BIN="${GT_BIN:-/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep350M_10kqueries_k100.bin}"
K="${K:-100}"
PART_FILE="${PART_FILE:-/gpfs/scratch/delgaudio/gp_ann/deep_learn_350M.k=8.GP}"
ROUTER_FILE="${ROUTER_FILE:-/gpfs/scratch/delgaudio/gp_ann/deep_learn_350M.k=8.GP.routing_index}"

exec srun -N 1 -n 1 --cpu-bind=no -w "${FIRST}" /bin/bash -s <<ENDRUN
set -euo pipefail
export OPAL_PREFIX=${OPAL_PREFIX}
export PATH=${OPAL_PREFIX}/bin:${PATH:-}
export LD_LIBRARY_PATH=${OPAL_PREFIX}/lib:/gpfs/scratch/delgaudio/openmpi_usr_lib_shim:/softs/gcc/11.3.0/lib64:/softs/gcc/11.3.0/lib:/lib64:\${LD_LIBRARY_PATH:-}
HOSTFILE=/gpfs/scratch/delgaudio/gp_ann/hostfile.${JBID}
mpirun -np ${NNODES} \
  --map-by ppr:1:node \
  --bind-to none \
  --hostfile "${HOSTFILE}" \
  -x OPAL_PREFIX -x PATH -x LD_LIBRARY_PATH \
  ${BENCH_BIN} \
  ${BASE_FBIN} \
  ${QUERY_FBIN} \
  ${GT_BIN} \
  ${K} \
  ${PART_FILE} \
  ${ROUTER_FILE}
ENDRUN
