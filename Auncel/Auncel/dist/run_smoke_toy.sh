#!/usr/bin/env bash
# Smoke test: single worker, toy fbin dataset (login node).
set -euo pipefail

DIST="$(cd "$(dirname "$0")" && pwd)"
WORK="${AUNCEL_WORK_DIR:-$DIST/smoke_toy}"

module load intel/20U4/mkl 2>/dev/null || true
export LD_LIBRARY_PATH="${MKLROOT:-}/lib/intel64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p "$WORK/results"
cd "$WORK"

pkill -f "$DIST/worker" 2>/dev/null || true
sleep 1

echo "==> Worker (toy defaults: random_base_100.fbin)"
"$DIST/worker" 3456 > worker.log 2>&1 &
WORKER_PID=$!
trap 'kill $WORKER_PID 2>/dev/null || true' EXIT
sleep 2
head -12 worker.log

echo "==> Train (IVF build + error profile)"
"$DIST/master" train 127.0.0.1:3456

echo "==> Search"
"$DIST/master" search "$WORK/results" 127.0.0.1:3456

ls -la "$WORK/results" || true
tail -25 worker.log

kill $WORKER_PID
trap - EXIT
echo "Done."
