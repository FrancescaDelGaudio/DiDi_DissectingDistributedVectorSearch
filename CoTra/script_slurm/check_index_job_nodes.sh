#!/bin/bash
# Verify that CoTra index jobs use disjoint node sets (no shared ncpum nodes).
#
# Usage:
#   bash script_slurm/check_index_job_nodes.sh
#   bash script_slurm/check_index_job_nodes.sh 5926596 5926598 5928330 5928331

set -euo pipefail

if [[ $# -gt 0 ]]; then
  JOB_IDS=("$@")
else
  mapfile -t JOB_IDS < <(
    squeue -u "${USER}" -h -o "%i %j" 2>/dev/null \
      | awk '$2 ~ /^cotra_idx/ {print $1}'
  )
fi

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
  echo "No cotra_idx* jobs in queue for user ${USER}."
  exit 0
fi

declare -A SEEN
OVERLAP=0

echo "=== CoTra index job node check ==="
for jid in "${JOB_IDS[@]}"; do
  line="$(scontrol show job "$jid" 2>/dev/null \
    | tr ' ' '\n' \
    | grep -E '^(JobId|JobName|JobState|NodeList)=' \
    | paste - - -)"
  [[ -n "$line" ]] || {
    echo "Job $jid: not found (completed or invalid id)"
    continue
  }
  name="$(echo "$line" | sed -n 's/.*JobName=\([^ ]*\).*/\1/p')"
  state="$(echo "$line" | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')"
  nodes="$(echo "$line" | sed -n 's/.*NodeList=\([^ ]*\).*/\1/p')"
  echo "Job $jid ($name) state=$state nodes=$nodes"

  # Expand Slurm node list: ncpum[003-004,008] -> ncpum003 ncpum004 ncpum008
  expanded="$(scontrol show hostnames "$nodes" 2>/dev/null || echo "$nodes")"
  while read -r host; do
    [[ -z "$host" ]] && continue
    if [[ -n "${SEEN[$host]:-}" ]]; then
      echo "  OVERLAP: $host already used by job ${SEEN[$host]}"
      OVERLAP=1
    else
      SEEN[$host]="$jid"
    fi
  done <<< "$expanded"
done

echo
if [[ $OVERLAP -eq 1 ]]; then
  echo "FAIL: at least one node is shared between jobs."
  echo "Cancel overlapping jobs and resubmit with --exclusive (see submit_all_index_jobs.sh)."
  exit 1
fi

echo "OK: no overlapping nodes among listed jobs."
