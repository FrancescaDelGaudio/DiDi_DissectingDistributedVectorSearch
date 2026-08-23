#!/usr/bin/env bash
# Submit DistributedBench sweep: DEEP fixed 4-shard, overlap=0.2
#   nprobes in {2,3,4} x recall@K in {10,100}  -> 6 jobs
#
# Usage:
#   ./submit_distbench_deep_fixed_o20.sh
#   ./submit_distbench_deep_fixed_o20.sh --dry-run
#   ./submit_distbench_deep_fixed_o20.sh --shards-probe 3 --recall-k 10

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/submit_distbench_deep_fixed_sweep.py" --overlap 0.2 "$@"
