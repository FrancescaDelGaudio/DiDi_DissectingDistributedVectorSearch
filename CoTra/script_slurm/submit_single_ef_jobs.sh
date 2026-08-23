#!/bin/bash
# Submit one Slurm job per ef value (no in-process sweep).
#
# Requires build with COTRA_EF support in exec_query.h.
#
# Examples:
#   # TTI 4p k=10, all 21 ef (sequential chain):
#   bash script_slurm/submit_single_ef_jobs.sh tti 10 4
#
#   # From ef=20 only (e.g. after sweep hung on 8th iteration = ef 20):
#   bash script_slurm/submit_single_ef_jobs.sh tti 10 4 --from-ef 20
#
#   # Explicit ef list:
#   COTRA_EF_LIST="8,10,15,20" bash script_slurm/submit_single_ef_jobs.sh tti 10 4
#
# Env:
#   COTRA_EF_LIST   comma-separated ef values (overrides default list / --from-ef)
#   COTRA_IB_DRAIN_SEC  sleep between chained jobs (default 30)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET="${1:?dataset: deep|tti}"
RES_KNN="${2:?res_knn: 10|100}"
PARTS="${3:?partitions: 4|6}"
FROM_EF=""
while [[ $# -gt 3 ]]; do
  case "$4" in
    --from-ef) FROM_EF="${5:?}"; shift 2 ;;
    *) shift ;;
  esac
done

DEFAULT_EFS=(3 4 5 6 8 10 15 20 30 40 50 70 100 150 200 250 300 400 500 600 700)

if [[ -n "${COTRA_EF_LIST:-}" ]]; then
  IFS=',' read -ra EFS <<< "$COTRA_EF_LIST"
elif [[ -n "$FROM_EF" ]]; then
  EFS=()
  skip=1
  for e in "${DEFAULT_EFS[@]}"; do
    if [[ "$skip" -eq 1 && "$e" -lt "$FROM_EF" ]]; then
      continue
    fi
    skip=0
    EFS+=("$e")
  done
else
  EFS=("${DEFAULT_EFS[@]}")
fi

if [[ "$PARTS" == "4" ]]; then
  SLURM=script_slurm/scala_anns.slurm
  test -x build/tests/scala_anns
else
  SLURM=script_slurm/scala_anns_6p.slurm
  test -x build-6p/tests/scala_anns_6p
fi

DRAIN="${COTRA_IB_DRAIN_SEC:-30}"
DEP=""
echo "=== CoTra single-ef jobs: ${DATASET} ${PARTS}p k=${RES_KNN} (${#EFS[@]} jobs, sequential) ==="
for ef in "${EFS[@]}"; do
  jname="cotra${PARTS}_${DATASET}_k${RES_KNN}_ef${ef}"
  dep_flag=()
  [[ -n "$DEP" ]] && dep_flag=(--dependency="afterany:${DEP}")
  jid="$(sbatch "${dep_flag[@]}" --exclusive --job-name="$jname" \
    --export=ALL,COTRA_DATASET="${DATASET}",COTRA_RES_KNN="${RES_KNN}",COTRA_EF="${ef}",COTRA_IB_DRAIN_SEC="${DRAIN}" \
    "$SLURM")"
  jid="${jid##* }"
  echo "  ef=${ef}  job=${jid}${DEP:+ (after:${DEP})}"
  DEP="$jid"
done
echo
echo "Monitor: squeue -u \"\$USER\" -p ncpumshort"
