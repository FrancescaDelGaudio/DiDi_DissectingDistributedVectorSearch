#!/usr/bin/bash
# TTI 10M on 5 nodes (4 workers): k in {10,100} x nprobe in {256,384,512,768}
# (1024 OOM on ~247GB nodes; 256 fills gap below 384 after sweep up to 192)
#
# Usage:
#   ./submit_tti_5n_k_nprobe_sweep.sh
#   ./submit_tti_5n_k_nprobe_sweep.sh --dry-run

set -eo pipefail

SBATCH_SCRIPT="/gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/harmony_tti10m_5n_ncpushort.sh"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

K_VALUES=(10 100)
NPROBE_VALUES=(256 384 512 768)

for k in "${K_VALUES[@]}"; do
  for nprobe in "${NPROBE_VALUES[@]}"; do
    job_name="hmy-tti5n-k${k}-np${nprobe}"
    log_base="/gpfs/users/delgaudio/exp_eval/Harmony/slurm_script/log/harmony_tti10m_5n_k${k}_np${nprobe}_%j"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "sbatch --job-name=${job_name} -o ${log_base}.out -e ${log_base}.err --export=ALL,HARMONY_K=${k},HARMONY_NPROBE=${nprobe} ${SBATCH_SCRIPT}"
    else
      sbatch \
        --job-name="${job_name}" \
        -o "${log_base}.out" \
        -e "${log_base}.err" \
        --export=ALL,HARMONY_K="${k}",HARMONY_NPROBE="${nprobe}" \
        "${SBATCH_SCRIPT}"
    fi
  done
done

echo "Done. Submitted $(( ${#K_VALUES[@]} * ${#NPROBE_VALUES[@]} )) TTI 5-node jobs."
