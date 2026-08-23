#!/bin/bash
# Verify CoTra .conf IP mapping the same way scala_index does at startup.
# Run via srun on each allocated node (CONF_FILE must be set).
set -euo pipefail

: "${CONF_FILE:?CONF_FILE not set}"

resolve_hostname() {
  local host="$1"
  if command -v getent &>/dev/null; then
    getent ahostsv4 "$host" | awk '/STREAM/ {print $1; exit}'
  elif command -v host &>/dev/null; then
    host "$host" | awk '/has address/ {print $4; exit}'
  else
    return 1
  fi
}

usable_ipv4() {
  local ip="$1"
  [[ -z "$ip" ]] && return 1
  [[ "$ip" == 127.* ]] && return 1
  [[ "$ip" == 169.254.* ]] && return 1
  return 0
}

# hostname -I prints all addresses on one line; split into tokens.
get_local_cluster_ips() {
  local ip
  for ip in $(hostname -I 2>/dev/null); do
    usable_ipv4 "$ip" && echo "$ip"
  done
  resolve_hostname "$(hostname -s)" 2>/dev/null || true
}

host_short="$(hostname -s)"
mapfile -t local_ips < <(get_local_cluster_ips | awk '!seen[$0]++')
if [[ "${#local_ips[@]}" -eq 0 ]]; then
  echo "[$(hostname)] FATAL: could not determine local cluster IPv4"
  exit 1
fi

map_ip=""
while IFS= read -r line; do
  [[ "$line" == *"="* ]] || continue
  name="${line%%=*}"
  id="${line#*=}"
  if [[ "$name" == "$host_short" || "$name" == "$(hostname)" ]]; then
    map_ip="$(resolve_hostname "$name" || true)"
    echo "[$(hostname)] hostname=$name id=$id dns_ip=${map_ip:-?} local_ips=${local_ips[*]}"
    for lip in "${local_ips[@]}"; do
      if [[ "$map_ip" == "$lip" ]]; then
        echo "[$(hostname)] IP OK ($lip)"
        exit 0
      fi
    done
    echo "[$(hostname)] FATAL: none of local IPs match conf map (expected $map_ip)"
    exit 1
  fi
done < "$CONF_FILE"

echo "[$(hostname)] FATAL: hostname not listed in $CONF_FILE"
exit 1
