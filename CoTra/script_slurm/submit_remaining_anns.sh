#!/bin/bash
# Full relaunch of all eight CoTra query jobs (alias for submit_all_anns_jobs.sh).
#
# Sequential chain on ncpumshort, k=100 before k=10 per dataset/partition.
#
# Usage:
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   bash script_slurm/submit_remaining_anns.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/script_slurm/submit_all_anns_jobs.sh"
