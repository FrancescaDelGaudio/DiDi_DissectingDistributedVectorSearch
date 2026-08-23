#!/bin/bash
# Submit four scala_anns jobs (4p only): DEEP/TTI × k=10/100.
#
# Sequential chain, k=100 before k=10 per dataset (ncpumshort, -t 00:10:00).
#
# Usage:
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   bash script_slurm/submit_anns_sweep.sh
#
# Optional index wait:
#   COTRA_INDEX_JOB_DEEP=<id> COTRA_INDEX_JOB_TTI=<id> bash script_slurm/submit_anns_sweep.sh

set -euo pipefail

COTRA_ROOT="/gpfs/users/delgaudio/exp_eval/CoTra"
cd "$COTRA_ROOT"
mkdir -p script_slurm/logs script_slurm/runtime

submit_one() {
  local job_name="$1"
  local dataset="$2"
  local res_knn="$3"
  local dep="${4:-}"

  local dep_flag=()
  if [[ -n "$dep" ]]; then
    dep_flag=(--dependency="afterok:${dep}")
  fi

  echo "Submitting ${job_name} (dataset=${dataset}, res_knn=${res_knn})"
  sbatch \
    "${dep_flag[@]}" \
    --job-name="$job_name" \
    --export=ALL,COTRA_DATASET="${dataset}",COTRA_RES_KNN="${res_knn}" \
    script_slurm/scala_anns.slurm
}

DEEP_DEP="${COTRA_INDEX_JOB_DEEP:-}"
TTI_DEP="${COTRA_INDEX_JOB_TTI:-}"

echo "==== CoTra anns sweep (4p, sequential k=100 → k=10) ===="
echo "DEEP index dependency: ${DEEP_DEP:-none}"
echo "TTI  index dependency: ${TTI_DEP:-none}"
echo

J_DEEP_100="$(submit_one cotra_deep_k100 deep 100 "$DEEP_DEP")"; J_DEEP_100="${J_DEEP_100##* }"
J_DEEP_10="$(submit_one cotra_deep_k10 deep 10 "$J_DEEP_100")"; J_DEEP_10="${J_DEEP_10##* }"
J_TTI_100="$(submit_one cotra_tti_k100 tti 100 "$J_DEEP_10")"; J_TTI_100="${J_TTI_100##* }"
J_TTI_10="$(submit_one cotra_tti_k10 tti 10 "$J_TTI_100")"; J_TTI_10="${J_TTI_10##* }"

echo
echo "  DEEP k=100: $J_DEEP_100"
echo "  DEEP k=10:  $J_DEEP_10  (afterok:$J_DEEP_100)"
echo "  TTI  k=100: $J_TTI_100  (afterok:$J_DEEP_10)"
echo "  TTI  k=10:  $J_TTI_10  (afterok:$J_TTI_100)"
echo
echo "Monitor: squeue -u \"\$USER\" -p ncpumshort"
