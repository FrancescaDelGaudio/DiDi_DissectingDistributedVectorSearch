#!/usr/bin/env bash
# Submit DistributedBench sweep: DEEP fixed 6-shard, overlap=0.2
#   nprobes in {2,3,4,5} x recall@K in {10,100}  -> 8 jobs (-N 6 -n 6)
#
# Usage (after QA / routing_index is ready):
#   ./submit_distbench_deep_fixed_k6_o20.sh
#   ./submit_distbench_deep_fixed_k6_o20.sh --dry-run

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/submit_distbench_deep_fixed_sweep.py" --ogp-k 6 --overlap 0.2 "$@"
