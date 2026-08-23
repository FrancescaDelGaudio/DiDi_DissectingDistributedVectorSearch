#!/bin/bash
# Run on each compute node (via srun) to sanity-check cluster conf.
set -euo pipefail

: "${CONF_FILE:?CONF_FILE not set}"

echo "[$(hostname)] CONF_FILE=$CONF_FILE"

if [[ ! -f "$CONF_FILE" ]]; then
  echo "[$(hostname)] FATAL: conf missing"
  exit 1
fi

host_short="$(hostname -s)"
host_full="$(hostname)"
entry="$(grep -E "^${host_short}=|^${host_full}=" "$CONF_FILE" || true)"

if [[ -z "$entry" ]]; then
  echo "[$(hostname)] FATAL: hostname not in conf"
  exit 1
fi

echo "[$(hostname)] conf entry: $entry"
echo "[$(hostname)] OK"
