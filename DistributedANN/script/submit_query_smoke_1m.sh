#!/usr/bin/bash
# Submit DEEP 1M query smoke test on 2 physical nodes (Slurm).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

sbatch --export=ALL,\
INDEX_DIR=/gpfs/scratch/delgaudio/dann/index_smoke_1m,\
QUERY_PATH=/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin,\
GT_PATH=/gpfs/scratch/delgaudio/dann/test/gt_1m_100q_k10.bin,\
K=10,L=200,BW=128,H=5,K_HEAD=100,EF_HEAD=200,NUM_QUERIES=100,WARMUP=5 \
  script/run_query.sbatch

squeue -u "$(whoami)" | head -5
