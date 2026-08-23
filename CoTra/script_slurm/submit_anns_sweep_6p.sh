#!/bin/bash
# Submit four scala_anns_6p jobs (DEEP L2 + TTI MIPS).
# Per dataset: k=100 first, then k=10 with --dependency=afterok.
#
# Usage:
#   bash script_slurm/submit_anns_sweep_6p.sh

set -euo pipefail

COTRA_ROOT="/gpfs/users/delgaudio/exp_eval/CoTra"
cd "$COTRA_ROOT"

submit_one() {
  local job_name="$1"
  local dataset="$2"
  local res_knn="$3"
  local dep="${4:-}"
  local dep_flag=()
  [[ -n "$dep" ]] && dep_flag=(--dependency="afterok:${dep}")
  echo "Submitting ${job_name} (6p, dataset=${dataset}, res_knn=${res_knn})"
  sbatch "${dep_flag[@]}" --job-name="$job_name" \
    --export=ALL,COTRA_DATASET="${dataset}",COTRA_RES_KNN="${res_knn}" \
    script_slurm/scala_anns_6p.slurm
}

DEEP_DEP="${COTRA_INDEX_JOB_DEEP_6P:-}"
TTI_DEP="${COTRA_INDEX_JOB_TTI_6P:-}"

echo "==== CoTra anns 6p sweep (k=100 before k=10 per dataset) ===="

J_DEEP_100="$(submit_one cotra6_deep_k100 deep 100 "$DEEP_DEP")"; J_DEEP_100="${J_DEEP_100##* }"
J_DEEP_10="$(submit_one cotra6_deep_k10 deep 10 "$J_DEEP_100")"; J_DEEP_10="${J_DEEP_10##* }"
J_TTI_100="$(submit_one cotra6_tti_k100 tti 100 "$TTI_DEP")"; J_TTI_100="${J_TTI_100##* }"
J_TTI_10="$(submit_one cotra6_tti_k10 tti 10 "$J_TTI_100")"; J_TTI_10="${J_TTI_10##* }"

echo "  DEEP 6p k=100: $J_DEEP_100"
echo "  DEEP 6p k=10:  $J_DEEP_10  (afterok:$J_DEEP_100)"
echo "  TTI  6p k=100: $J_TTI_100"
echo "  TTI  6p k=10:  $J_TTI_10  (afterok:$J_TTI_100)"
