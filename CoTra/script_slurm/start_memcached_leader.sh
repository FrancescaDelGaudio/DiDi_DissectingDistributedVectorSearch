#!/bin/bash
# Start memcached on the CoTra leader (machine 0) inside a Slurm job.
#
# Requires: CONF_FILE (from generate_cluster_conf.sh) or COTRA_LEADER_HOST + port.
# Only the node whose hostname matches the leader line starts memcached.
#
# Stops any stale CoTra memcached on the leader port before starting a fresh
# daemon (flush_all + kill). Without this, consecutive jobs on the same nodes
# reuse polluted keys (Machine-id-*, 0-1, ...) and RDMA Connect can segfault.
#
# Usage:
#   generate_cluster_conf   # sets CONF_FILE
#   bash script_slurm/start_memcached_leader.sh

set -euo pipefail

resolve_hostname() {
  local hostname=$1
  if command -v getent &>/dev/null; then
    getent ahostsv4 "$hostname" | awk '/STREAM/ {print $1; exit}'
  elif command -v host &>/dev/null; then
    host "$hostname" | awk '/has address/ {print $4; exit}'
  else
    return 1
  fi
}

if [[ -n "${CONF_FILE:-}" && -f "$CONF_FILE" ]]; then
  LEADER_HOST="$(sed -n '1p' "$CONF_FILE")"
  MEMCACHED_PORT="$(sed -n '2p' "$CONF_FILE")"
elif [[ -n "${COTRA_LEADER_HOST:-}" ]]; then
  LEADER_HOST="$COTRA_LEADER_HOST"
  MEMCACHED_PORT="${COTRA_MEMCACHED_PORT:-2378}"
else
  echo "FATAL: set CONF_FILE or COTRA_LEADER_HOST" >&2
  exit 1
fi

MY_HOST="$(hostname -s)"
MY_HOST_FQDN="$(hostname -f 2>/dev/null || hostname)"
LEADER_SHORT="$(echo "$LEADER_HOST" | cut -d. -f1)"

if [[ "$MY_HOST" != "$LEADER_SHORT" && "$MY_HOST_FQDN" != "$LEADER_HOST" && "$(hostname)" != "$LEADER_HOST" ]]; then
  echo "[$(hostname)] not leader ($LEADER_HOST); skip memcached"
  exit 0
fi

find_memcached() {
  if command -v memcached &>/dev/null; then
    command -v memcached
    return 0
  fi
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
    module load memcached 2>/dev/null || module load memcached/1.6 2>/dev/null || true
  fi
  if command -v memcached &>/dev/null; then
    command -v memcached
    return 0
  fi
  local deps="${COTRA_DEPS_PREFIX:-/gpfs/scratch/delgaudio/cotra/deps}"
  for cand in \
    "${COTRA_MEMCACHED_BIN:-}" \
    "$deps/bin/memcached" \
    /usr/bin/memcached \
    /usr/local/bin/memcached; do
    if [[ -n "$cand" && -x "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

MEMCACHED_BIN="$(find_memcached || true)"
if [[ -z "$MEMCACHED_BIN" ]]; then
  echo "FATAL: memcached not found on leader $(hostname)" >&2
  echo "  Build locally: bash script_slurm/build_cotra_deps.sh" >&2
  echo "  Then set COTRA_DEPS_PREFIX=/gpfs/scratch/delgaudio/cotra/deps" >&2
  exit 1
fi

PID_FILE="${TMPDIR:-/tmp}/memcached-cotra-${SLURM_JOB_ID:-$$}.pid"

LEADER_IP="$(resolve_hostname "$LEADER_HOST" || true)"
if [[ -z "${LEADER_IP:-}" ]]; then
  echo "FATAL: could not resolve leader IP for $LEADER_HOST" >&2
  exit 1
fi

mc_python() {
  python3 - "$LEADER_IP" "$MEMCACHED_PORT" "$@"
}

stop_stale_memcached() {
  echo "[$(hostname)] stopping stale memcached on ${LEADER_IP}:${MEMCACHED_PORT} (if any)"
  mc_python stop <<'PY' || true
import glob
import os
import signal
import socket
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])

def mc_cmd(cmd: str) -> str:
    s = socket.create_connection((host, port), timeout=2)
    s.sendall(cmd.encode())
    data = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\r\n" in data:
            break
    s.close()
    return data.decode(errors="replace")

try:
    mc_cmd("flush_all\r\n")
    mc_cmd("quit\r\n")
except OSError:
    pass

for pf in glob.glob("/tmp/memcached-cotra-*.pid"):
    try:
        with open(pf) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        pass
    try:
        os.remove(pf)
    except OSError:
        pass

time.sleep(0.5)
PY

  if command -v fuser &>/dev/null; then
    fuser -k "${MEMCACHED_PORT}/tcp" 2>/dev/null || true
  fi
  sleep 1
}

mc_init_keys() {
  mc_python init <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])

def mc_cmd(cmd: str) -> str:
    s = socket.create_connection((host, port), timeout=5)
    s.sendall(cmd.encode())
    data = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\r\n" in data:
            break
    s.close()
    return data.decode(errors="replace")

mc_cmd("flush_all\r\n")
for key in ("serverNum", "clientNum"):
    mc_cmd(f"set {key} 0 0 1\r\n0\r\n")
    resp = mc_cmd(f"get {key}\r\n")
    if "0" not in resp:
        print(f"FATAL: memcached key {key} not initialized; got: {resp!r}", file=sys.stderr)
        sys.exit(1)
print(f"memcached keys initialized on {host}:{port}")
PY
}

stop_stale_memcached

echo "[$(hostname)] starting memcached on ${LEADER_IP}:${MEMCACHED_PORT} (host ${LEADER_HOST})"
"$MEMCACHED_BIN" -u "$USER" -l "$LEADER_IP" -p "$MEMCACHED_PORT" -c 10000 -d -P "$PID_FILE"
sleep 1

mc_init_keys || {
  echo "FATAL: memcached key init failed on ${LEADER_IP}:${MEMCACHED_PORT}" >&2
  exit 1
}

echo "[$(hostname)] memcached ready (pid file: $PID_FILE)"
