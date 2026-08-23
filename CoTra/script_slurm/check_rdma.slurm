#!/bin/bash
# Probe InfiniBand / RDMA on every compute node allocated by Slurm.
#
# Usage (from login node):
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   mkdir -p script_slurm/logs
#   sbatch script_slurm/check_rdma.slurm
#
# Optional: different node count
#   sbatch -N 2 script_slurm/check_rdma.slurm
#
# After the job:
#   sacct -j <JOBID> --format=JobID,State,ExitCode
#   cat script_slurm/logs/check_rdma-<JOBID>.out

#SBATCH -J cotra_check_rdma
#SBATCH -p ncpum
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH -o /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/check_rdma-%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/check_rdma-%j.err

set -euo pipefail

mkdir -p /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

command -v module >/dev/null 2>&1 || {
  echo "FATAL: 'module' command not available in batch environment."
  exit 1
}

module purge
module load gcc/10.3.0
module load infiniband-diags 2>/dev/null || true
module load rdma-core 2>/dev/null || true

echo "==== CoTra RDMA probe start $(date -Is) jobid=${SLURM_JOB_ID:-?} ===="
echo "nodelist=${SLURM_JOB_NODELIST:-?} nnodes=${SLURM_NNODES:-?} ntasks=${SLURM_NTASKS:-?}"
echo

srun --ntasks-per-node=1 --cpus-per-task=1 bash -lc '
  log() { echo "[$(date -Is)] [$(hostname)] $*"; }
  log "=== node probe start ==="
  log "SLURM_NODEID=${SLURM_NODEID:-?} SLURM_PROCID=${SLURM_PROCID:-?}"

  IBV_DEVICES="$(command -v ibv_devices || true)"
  IBSTAT="$(command -v ibstat || true)"
  if [[ -z "$IBV_DEVICES" || -z "$IBSTAT" ]]; then
    log "ERROR: ibv_devices/ibstat not in PATH"
    log "  ibv_devices=${IBV_DEVICES:-NOT_FOUND}"
    log "  ibstat=${IBSTAT:-NOT_FOUND}"
    exit 1
  fi

  log "--- ibv_devices ---"
  "$IBV_DEVICES"
  echo
  log "--- ibstat ---"
  "$IBSTAT"
  echo

  if "$IBSTAT" 2>/dev/null | grep -q "State:[[:space:]]*Active"; then
    log "RESULT: PASS (Active IB port)"
  else
    log "RESULT: FAIL (no Active IB port)"
    exit 1
  fi
  log "=== node probe end ==="
  echo
'

echo "==== CoTra RDMA probe end $(date -Is) ===="
