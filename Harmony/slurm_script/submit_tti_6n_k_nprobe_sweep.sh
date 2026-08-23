#!/usr/bin/bash
# TTI 10M recall/QPS sweep on 6 physical nodes (6 workers, coordinator colocated).
# Same nprobe grid as DEEP plus 512, 768, 1024 -> 20 x k in {10,100} = 40 jobs.
# Note: nprobe=1024 may OOM on ~247GB nodes; retry with fewer OMP threads if needed.
#
# Usage:
#   ./submit_tti_6n_k_nprobe_sweep.sh
#   ./submit_tti_6n_k_nprobe_sweep.sh --dry-run

set -eo pipefail

SBATCH_SCRIPT="/gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/harmony_tti10m_6n_ncpushort.sh"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

K_VALUES=(10 100)
NPROBE_VALUES=(1 2 4 6 8 10 12 16 24 32 48 64 96 128 192 256 384 512 768 1024)

for k in "${K_VALUES[@]}"; do
  for nprobe in "${NPROBE_VALUES[@]}"; do
    job_name="hmy-tti6n-k${k}-np${nprobe}"
    log_base="/gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_tti10m_6n_k${k}_np${nprobe}_%j"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "sbatch --job-name=${job_name} -o ${log_base}.out -e ${log_base}.err --export=ALL,HARMONY_K=${k},HARMONY_NPROBE=${nprobe},HARMONY_GROUP=4,HARMONY_TEAM=3 ${SBATCH_SCRIPT}"
    else
      sbatch \
        --job-name="${job_name}" \
        -o "${log_base}.out" \
        -e "${log_base}.err" \
        --export=ALL,HARMONY_K="${k}",HARMONY_NPROBE="${nprobe}",HARMONY_GROUP=4,HARMONY_TEAM=3 \
        "${SBATCH_SCRIPT}"
    fi
  done
done

echo "Done. Submitted $(( ${#K_VALUES[@]} * ${#NPROBE_VALUES[@]} )) TTI 6-node jobs (or dry-run lines above)."
