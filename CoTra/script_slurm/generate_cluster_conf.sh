#!/bin/bash
# Generate CoTra cluster .conf from the nodes Slurm assigned to this job.
#
# CoTra identifies machines by resolving hostnames in the conf file to local IP.
# Hostnames are taken from $SLURM_JOB_NODELIST (unknown before the job starts).
#
# Usage (inside a Slurm batch script, after #SBATCH -N <N>):
#   source script_slurm/generate_cluster_conf.sh
#   generate_cluster_conf
#   echo "CONF_FILE=$CONF_FILE"
#
# Optional env:
#   COTRA_MACHINE_NUM=4     expected node count (must match compile-time MACHINE_NUM)
#   COTRA_MEMCACHED_PORT=2378
#   COTRA_CONF_DIR=...      where to write the conf (default: script_slurm/runtime)

set -euo pipefail

generate_cluster_conf() {
  : "${SLURM_JOB_ID:?run inside a Slurm batch job}"
  : "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is empty}"

  local expected="${COTRA_MACHINE_NUM:-4}"
  local port="${COTRA_MEMCACHED_PORT:-2378}"
  local conf_dir="${COTRA_CONF_DIR:-${SLURM_SUBMIT_DIR:-.}/script_slurm/runtime}"
  local nodes=()

  mkdir -p "$conf_dir"

  while IFS= read -r node; do
    [[ -n "$node" ]] && nodes+=("$node")
  done < <(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort)

  if [[ "${#nodes[@]}" -ne "$expected" ]]; then
    echo "FATAL: Slurm allocated ${#nodes[@]} nodes, expected COTRA_MACHINE_NUM=$expected" >&2
    echo "  nodelist: ${nodes[*]}" >&2
    echo "  Submit with: sbatch -N $expected ..." >&2
    return 1
  fi

  local leader="${nodes[0]}"
  CONF_FILE="${conf_dir}/cluster-job${SLURM_JOB_ID}.conf"

  {
    echo "$leader"
    echo "$port"
    local id=0
    for node in "${nodes[@]}"; do
      echo "${node}=${id}"
      id=$((id + 1))
    done
  } > "$CONF_FILE"

  export CONF_FILE
  export COTRA_LEADER_HOST="$leader"
  export COTRA_NODE_LIST="${nodes[*]}"

  echo "==== CoTra cluster conf generated ===="
  echo "CONF_FILE=$CONF_FILE"
  echo "Leader (machine 0): $leader"
  echo "Nodes (${#nodes[@]}): ${nodes[*]}"
  echo "--- conf contents ---"
  cat "$CONF_FILE"
  echo "---------------------"
}
