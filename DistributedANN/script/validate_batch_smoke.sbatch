#!/usr/bin/bash
#SBATCH --job-name=dann-batch-val
#SBATCH -p ncpushort
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /gpfs/scratch/delgaudio/dann/logs/dann-batch-val_%j.log
#SBATCH -e /gpfs/scratch/delgaudio/dann/logs/dann-batch-val_%j.log
#
# Correctness check of run_query_batch vs run_query (read-only use of the
# current binary). Same params, --print_results 1, ids must match per query.
# Runs: batch T=1 chunk=all, T=4 chunk=all, T=4 chunk=8.

set -euo pipefail
module purge
module load gcc/10.3.0
module load openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

REF_BIN="${REF_BIN:-/gpfs/users/delgaudio/exp_eval/DistributedANN/build/bin/run_query}"
BATCH_BIN="${BATCH_BIN:-/gpfs/users/delgaudio/exp_eval_da_batch/DistributedANN/build_batch/bin/run_query_batch}"
INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_smoke_1m}"
QUERY_PATH="${QUERY_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin}"
GT_PATH="${GT_PATH:-/gpfs/scratch/delgaudio/dann/test/gt_1m_100q_k10.bin}"
NQ="${NQ:-100}"
PARAMS=(--k 10 --L 200 --BW 128 --H 5 --k_head 100 --ef_head 200 --num_queries "${NQ}" --warmup 5)

OUT="/gpfs/scratch/delgaudio/dann/logs/batch_val_${SLURM_JOB_ID}"
mkdir -p "${OUT}"
MPI=(mpirun -np "${SLURM_NTASKS}" --map-by ppr:2:node --bind-to none)

extract() { grep -E '^query [0-9]+ ->' "$1" | sort -t' ' -k2,2n > "$2"; }

echo "=== REF run_query ==="
"${MPI[@]}" "${REF_BIN}" --index_dir "${INDEX_DIR}" --query_path "${QUERY_PATH}" \
    --gt_path "${GT_PATH}" "${PARAMS[@]}" --print_results 1 > "${OUT}/ref.txt"
grep -E "SWEEP_SUMMARY|recall" "${OUT}/ref.txt" || true
extract "${OUT}/ref.txt" "${OUT}/ref.ids"

fail=0
for cfg in "1 0" "4 0" "4 8"; do
  read -r T C <<< "${cfg}"
  tag="T${T}_C${C}"
  echo "=== BATCH ${tag} ==="
  OMP_NUM_THREADS="${T}" "${MPI[@]}" "${BATCH_BIN}" --index_dir "${INDEX_DIR}" \
      --query_path "${QUERY_PATH}" --gt_path "${GT_PATH}" "${PARAMS[@]}" \
      --threads "${T}" --chunk "${C}" --print_results 1 > "${OUT}/batch_${tag}.txt"
  grep -E "SWEEP_SUMMARY|\[batch\]" "${OUT}/batch_${tag}.txt" || true
  extract "${OUT}/batch_${tag}.txt" "${OUT}/batch_${tag}.ids"
  n_ref=$(wc -l < "${OUT}/ref.ids"); n_b=$(wc -l < "${OUT}/batch_${tag}.ids")
  if diff -q "${OUT}/ref.ids" "${OUT}/batch_${tag}.ids" > /dev/null && [[ "${n_ref}" -eq "${NQ}" ]]; then
    echo "VALIDATION ${tag}: IDENTICAL (${n_b}/${NQ} queries)"
  else
    ndiff=$(diff "${OUT}/ref.ids" "${OUT}/batch_${tag}.ids" | grep -c '^<' || true)
    echo "VALIDATION ${tag}: MISMATCH (${ndiff} queries differ; ref=${n_ref} batch=${n_b})"
    fail=1
  fi
done
echo "=== outputs in ${OUT} ==="
exit "${fail}"
