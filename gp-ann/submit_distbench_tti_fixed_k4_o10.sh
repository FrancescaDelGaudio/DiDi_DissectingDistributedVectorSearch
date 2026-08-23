#!/usr/bin/env bash
# TTI fixed 4-shard, o=0.1: nprobes {2,3,4} x K {10,100} -> 6 jobs
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/submit_distbench_tti_fixed_sweep.py" --ogp-k 4 --overlap 0.1 "$@"
