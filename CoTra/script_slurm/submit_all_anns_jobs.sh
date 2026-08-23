#!/bin/bash
# Submit all eight CoTra query jobs (DEEP/TTI × 4p/6p × k=10/100).
#
# Default: one sequential chain on ncpumshort (--exclusive), k=100 before k=10
# per dataset/partition (same partition/time as successful 5932632/5932634 run).
#   bash script_slurm/submit_all_anns_jobs.sh
#
# Parallel (eight independent jobs — needs 32 free nodes if all start together):
#   bash script_slurm/submit_all_anns_jobs.sh --parallel
#
# Each job: #SBATCH -p ncpumshort -t 00:10:00 --mem=150G --exclusive

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p script_slurm/logs script_slurm/runtime

PARALLEL=false
if [[ "${1:-}" == "--parallel" ]]; then
  PARALLEL=true
fi

BIN4="${ROOT}/build/tests/scala_anns"
BIN6="${ROOT}/build-6p/tests/scala_anns_6p"
test -x "$BIN4" || {
  echo "FATAL: missing $BIN4 — run: bash script_slurm/build_cotra.sh" >&2
  exit 1
}
test -x "$BIN6" || {
  echo "FATAL: missing $BIN6 — run: bash script_slurm/build_cotra_6p.sh" >&2
  exit 1
}

submit_4p() {
  local job_name="$1"
  local dataset="$2"
  local res_knn="$3"
  local dep="${4:-}"
  local drain_sec="${5:-0}"
  local dep_flag=()
  [[ -n "$dep" ]] && dep_flag=(--dependency="afterany:${dep}")
  sbatch "${dep_flag[@]}" --exclusive --job-name="$job_name" \
    --export=ALL,COTRA_DATASET="${dataset}",COTRA_RES_KNN="${res_knn}",COTRA_IB_DRAIN_SEC="${drain_sec}" \
    script_slurm/scala_anns.slurm
}

submit_6p() {
  local job_name="$1"
  local dataset="$2"
  local res_knn="$3"
  local dep="${4:-}"
  local drain_sec="${5:-0}"
  local dep_flag=()
  [[ -n "$dep" ]] && dep_flag=(--dependency="afterany:${dep}")
  sbatch "${dep_flag[@]}" --exclusive --job-name="$job_name" \
    --export=ALL,COTRA_DATASET="${dataset}",COTRA_RES_KNN="${res_knn}",COTRA_IB_DRAIN_SEC="${drain_sec}" \
    script_slurm/scala_anns_6p.slurm
}

# TTI 4p k=10 only: one Slurm job per ef (COTRA_EF), no in-process sweep.
# All other configs keep the default 21-ef sweep in scala_anns.
submit_tti_4p_k10_single_ef() {
  local dep="${1:-}"
  local drain_sec="${2:-0}"
  local ef jid dep_flag
  local -a efs=(3 4 5 6 8 10 15 20 30 40 50 70 100 150 200 250 300 400 500 600 700)
  for ef in "${efs[@]}"; do
    dep_flag=()
    [[ -n "$dep" ]] && dep_flag=(--dependency="afterany:${dep}")
    jid="$(sbatch "${dep_flag[@]}" --exclusive --job-name="cotra4_tti_k10_ef${ef}" \
      --export=ALL,COTRA_DATASET=tti,COTRA_RES_KNN=10,COTRA_EF="${ef}",COTRA_IB_DRAIN_SEC="${drain_sec}" \
      script_slurm/scala_anns.slurm)"
    dep="${jid##* }"
  done
  echo "$dep"
}

if [[ "$PARALLEL" == true ]]; then
  echo "=== CoTra anns batch (parallel, k=100 before k=10 per dataset) ==="
  J1="$(submit_4p cotra_deep_k100 deep 100)"; J1="${J1##* }"
  J2="$(submit_4p cotra_deep_k10 deep 10)"; J2="${J2##* }"
  J3="$(submit_4p cotra_tti_k100 tti 100)"; J3="${J3##* }"
  J4="$(submit_tti_4p_k10_single_ef)"
  J5="$(submit_6p cotra6_deep_k100 deep 100)"; J5="${J5##* }"
  J6="$(submit_6p cotra6_deep_k10 deep 10)"; J6="${J6##* }"
  J7="$(submit_6p cotra6_tti_k100 tti 100)"; J7="${J7##* }"
  J8="$(submit_6p cotra6_tti_k10 tti 10)"; J8="${J8##* }"
  echo "Submitted (parallel, no dependencies):"
  echo "  DEEP 4p k=100:  $J1"
  echo "  DEEP 4p k=10:   $J2"
  echo "  TTI  4p k=100:  $J3"
  echo "  TTI  4p k=10:   $J4  (21 single-ef jobs, no sweep)"
  echo "  DEEP 6p k=100:  $J5"
  echo "  DEEP 6p k=10:   $J6"
  echo "  TTI  6p k=100:  $J7"
  echo "  TTI  6p k=10:   $J8"
else
  echo "=== CoTra anns full relaunch (sequential, ncpumshort, k=100 → k=10) ==="
  DRAIN="${COTRA_IB_DRAIN_SEC:-30}"
  J1="$(submit_4p cotra_deep_k100 deep 100)"; J1="${J1##* }"
  J2="$(submit_4p cotra_deep_k10 deep 10 "$J1" "$DRAIN")"; J2="${J2##* }"
  J3="$(submit_4p cotra_tti_k100 tti 100 "$J2" "$DRAIN")"; J3="${J3##* }"
  J4="$(submit_tti_4p_k10_single_ef "$J3" "$DRAIN")"
  J5="$(submit_6p cotra6_deep_k100 deep 100 "$J4" "$DRAIN")"; J5="${J5##* }"
  J6="$(submit_6p cotra6_deep_k10 deep 10 "$J5" "$DRAIN")"; J6="${J6##* }"
  J7="$(submit_6p cotra6_tti_k100 tti 100 "$J6" "$DRAIN")"; J7="${J7##* }"
  J8="$(submit_6p cotra6_tti_k10 tti 10 "$J7" "$DRAIN")"; J8="${J8##* }"
  echo "Submitted (sequential chain, k=100 before k=10 each step):"
  echo "  DEEP 4p k=100:  $J1"
  echo "  DEEP 4p k=10:   $J2  (afterok:$J1)"
  echo "  TTI  4p k=100:  $J3  (afterok:$J2)"
  echo "  TTI  4p k=10:   $J4  (21 single-ef jobs after:$J3, no sweep)"
  echo "  DEEP 6p k=100:  $J5  (afterok:$J4)"
  echo "  DEEP 6p k=10:   $J6  (afterok:$J5)"
  echo "  TTI  6p k=100:  $J7  (afterok:$J6)"
  echo "  TTI  6p k=10:   $J8  (afterok:$J7)"
fi

echo
echo "Monitor: squeue -u \"\$USER\" -p ncpumshort"
