#!/bin/bash
# Submit all four CoTra distributed index jobs (DEEP/TTI × 4p/6p).
#
# Each job uses #SBATCH --exclusive so Slurm will NOT place two CoTra jobs on
# the same node. To run all four in parallel you need 4+4+6+6 = 20 exclusive
# nodes on ncpum; otherwise some jobs stay PD until enough nodes are free.
#
# Usage:
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   bash script_slurm/build_cotra.sh
#   bash script_slurm/build_cotra_6p.sh
#   bash script_slurm/submit_all_index_jobs.sh
#
# After jobs start (ST=RUNNING), verify disjoint node sets:
#   bash script_slurm/check_index_job_nodes.sh
#
# Sequential mode (one job at a time, only 6 nodes peak):
#   bash script_slurm/submit_all_index_jobs.sh --sequential

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p script_slurm/logs script_slurm/runtime

SEQUENTIAL=false
if [[ "${1:-}" == "--sequential" ]]; then
  SEQUENTIAL=true
fi

BIN4="${ROOT}/build/tests/scala_index"
BIN6="${ROOT}/build-6p/tests/scala_index_6p"
test -x "$BIN4" || {
  echo "FATAL: missing $BIN4 — run: bash script_slurm/build_cotra.sh" >&2
  exit 1
}
test -x "$BIN6" || {
  echo "FATAL: missing $BIN6 — run: bash script_slurm/build_cotra_6p.sh" >&2
  exit 1
}

submit_one() {
  local script="$1"
  local dep="${2:-}"
  if [[ -n "$dep" ]]; then
    sbatch --dependency="afterok:${dep}" "$script"
  else
    sbatch "$script"
  fi
}

echo "=== CoTra index batch submit ==="
echo "Mode: $([[ "$SEQUENTIAL" == true ]] && echo sequential || echo parallel, --exclusive)"
echo "Peak nodes (parallel): 20 (4+4+6+6 on ncpum)"
echo

if [[ "$SEQUENTIAL" == true ]]; then
  J1="$(submit_one script_slurm/scala_index_deep10M.slurm)"
  J1="${J1##* }"
  J2="$(submit_one script_slurm/scala_index_tti10M.slurm "$J1")"
  J2="${J2##* }"
  J3="$(submit_one script_slurm/scala_index_deep10M_6p.slurm "$J2")"
  J3="${J3##* }"
  J4="$(submit_one script_slurm/scala_index_tti10M_6p.slurm "$J3")"
  J4="${J4##* }"
  echo "Submitted (sequential chain):"
  echo "  DEEP 4p:  $J1"
  echo "  TTI  4p:  $J2  (afterok:$J1)"
  echo "  DEEP 6p:  $J3  (afterok:$J2)"
  echo "  TTI  6p:  $J4  (afterok:$J3)"
else
  J1="$(submit_one script_slurm/scala_index_deep10M.slurm)"
  J1="${J1##* }"
  J2="$(submit_one script_slurm/scala_index_tti10M.slurm)"
  J2="${J2##* }"
  J3="$(submit_one script_slurm/scala_index_deep10M_6p.slurm)"
  J3="${J3##* }"
  J4="$(submit_one script_slurm/scala_index_tti10M_6p.slurm)"
  J4="${J4##* }"
  echo "Submitted (parallel, each --exclusive):"
  echo "  DEEP 4p:  $J1"
  echo "  TTI  4p:  $J2"
  echo "  DEEP 6p:  $J3"
  echo "  TTI  6p:  $J4"
  echo
  echo "When all are RUNNING, check for node overlap:"
  echo "  bash script_slurm/check_index_job_nodes.sh $J1 $J2 $J3 $J4"
fi

echo
echo "Monitor: squeue -u \"\$USER\" -o \"%.10i %.14j %.8T %.10M %.6D %N\""
