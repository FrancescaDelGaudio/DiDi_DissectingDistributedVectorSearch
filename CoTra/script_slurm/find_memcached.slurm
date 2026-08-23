#!/bin/bash
# Find memcached on compute nodes (run from login node).
#SBATCH -J cotra_find_memcached
#SBATCH -p ncpum
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:05:00
#SBATCH -o /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/find_memcached-%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/CoTra/script_slurm/logs/find_memcached-%j.err

set -euo pipefail
cd /gpfs/users/delgaudio/exp_eval/CoTra

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

echo "HOST=$(hostname)"
echo "=== module avail memcached ==="
module avail memcached 2>&1 || true
echo "=== which memcached (before module) ==="
which memcached 2>&1 || true
for m in memcached memcached/1.6 memcached/1.6.9; do
  module load "$m" 2>/dev/null && echo "loaded $m" && break || true
done
echo "=== which memcached (after module) ==="
which memcached 2>&1 || true
ls -la /usr/bin/memcached /usr/local/bin/memcached 2>&1 || true
