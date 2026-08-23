#!/usr/bin/env bash
#SBATCH --job-name=auncel-2w-3n
#SBATCH --partition=ncpushort
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /gpfs/scratch/delgaudio/vectors/toy/logs/auncel_2w3n_%j.out
#SBATCH -e /gpfs/scratch/delgaudio/vectors/toy/logs/auncel_2w3n_%j.err

# Legacy online train+search on random_100k (no prebuilt index).
# Prefer: run_auncel_prepare_100k.sh → build → search_3node.sh

set -euo pipefail

DIST="${AUNCEL_DIST_ROOT:-/gpfs/users/delgaudio/exp_eval/Auncel/Auncel/dist}"
# shellcheck source=sh/auncel_100k_env.sh
source "${DIST}/sh/auncel_100k_env.sh"
LOG_DIR="${AUNCEL_LOG_DIR}"
RESULTS="${AUNCEL_DATA_DIR}/auncel_results_${SLURM_JOB_ID}"
COORD="${AUNCEL_DATA_DIR}/auncel_coord_${SLURM_JOB_ID}"

GLOBAL_RECALL="${AUNCEL_RECALL:-0.9}"
HYPERPARAM_ID="${AUNCEL_HP_ID:-1}"

mkdir -p "$LOG_DIR" "$RESULTS" "$COORD"
rm -f "${COORD}/ip0" "${COORD}/ip1"

module load intel/20U4/mkl 2>/dev/null || true
export LD_LIBRARY_PATH="${MKLROOT:-}/lib/intel64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export AUNCEL_EXIT_AFTER_SEARCH=1

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
W0_NODE="${NODES[1]}"
W1_NODE="${NODES[2]}"

echo "=== Auncel online train+search (random_100k) ==="
echo "query: ${AUNCEL_QUERY}  k: ${AUNCEL_K}  index: ${AUNCEL_INDEX_KEY}"

worker_online() {
  local shard="$1" node="$2"
  local log="${LOG_DIR}/auncel_${SLURM_JOB_ID}_w${shard}.log"
  srun --nodes=1 --nodelist="${node}" --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c "
      module load intel/20U4/mkl 2>/dev/null || true
      export LD_LIBRARY_PATH=\${MKLROOT:-}/lib/intel64:\${LD_LIBRARY_PATH:-}
      export OMP_NUM_THREADS=${OMP_NUM_THREADS}
      export AUNCEL_EXIT_AFTER_SEARCH=1
      IP=\$(hostname -i | awk '{print \$1}')
      echo \$IP > ${COORD}/ip${shard}
      cd ${DIST}
      exec ./worker 3456 \
        ${AUNCEL_SHARDS}/${shard}db.fbin ${AUNCEL_QUERY} \
        ${AUNCEL_SHARDS}/${shard}gti.ivecs ${AUNCEL_SHARDS}/${shard}gtd.fvecs \
        ${AUNCEL_INDEX_KEY} ${GLOBAL_RECALL} ${HYPERPARAM_ID} ${OMP_NUM_THREADS}
    " > "${log}" 2>&1 &
  echo $!
}

SRUN_W0=$(worker_online 0 "${W0_NODE}")
SRUN_W1=$(worker_online 1 "${W1_NODE}")

for _ in $(seq 1 300); do
  [[ -f "${COORD}/ip0" && -f "${COORD}/ip1" ]] && break
  sleep 1
done

IP0="$(cat "${COORD}/ip0")"
IP1="$(cat "${COORD}/ip1")"
sleep 2

cd "${DIST}"
./master train "${IP0}:3456" "${IP1}:3456"
./master search "${RESULTS}" --global-recall "${GLOBAL_RECALL}" --k "${AUNCEL_K}" \
  "${IP0}:3456" "${IP1}:3456"
./reduce "${RESULTS}" 2 0 "${AUNCEL_SHARD_OFFSET}"
./eval_results --gt "${AUNCEL_GT}" --result "${RESULTS}"
ls -la "${RESULTS}"

wait "${SRUN_W0}" 2>/dev/null || true
wait "${SRUN_W1}" 2>/dev/null || true
echo "Done."
