#!/bin/bash
# Smoke test: DistributedBench su 2 nodi fisici, poche query (50).
# Uso: sbatch /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_distributed_smoke_2nodes.sh
# Log: sbatch_logs/dist_smoke_2n-<jobid>.out (stdout+stderr nello stesso file)
#
# MPI su QUESTO cluster: sui nodi compute (es. ncpu00x) /usr/lib64/openmpi/bin
# puo' NON esistere. La modalita consigliata e' il default: module load
#   openmpi/gcc-10.3.0/4.1.1
# Uso: sbatch senza USE_SYSTEM_MPI, oppure unset USE_SYSTEM_MPI
#
# USE_SYSTEM_MPI=1 solo se mpirun e' in /usr (es. stesso nodo del login) — spesso
# in batch NON funziona; allora usare SEMPRE i moduli sotto /softs/Modules.
#
# OPENMPI: override con OPENMPI_MODULE=... se serve un altro modulo da module avail
#
# Partizione (linee guida cluster): ncpushort = stessi nodi ncpu, max 2h, priorita piu alta.

#SBATCH -J dist_smoke_2n
#SBATCH -p ncpushort
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
# Stesso file per .out e .err, cosi vedi tutto in dist_smoke_2n-<id>.out
#SBATCH -o /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_logs/dist_smoke_2n-%j.out
#SBATCH -e /gpfs/users/delgaudio/exp_eval/gp-ann/sbatch_logs/dist_smoke_2n-%j.out
# Log sempre visibile in .out (anche se i module falliscono subito dopo)
echo "==== dist_smoke start $(date -Is) jobid=${SLURM_JOB_ID:-?} batch_host=$(hostname) ===="
echo "Se i module falliscono: nodo compute puo usare Lmod/mirror diverso. Su login, esegui: module avail openmpi"
set -euo pipefail

# module load opzionale: con USE_SYSTEM_MPI=1 si usa solo /usr (alineato al RUNPATH del binario).
USE_SYSTEM_MPI="${USE_SYSTEM_MPI:-0}"
if [[ "$USE_SYSTEM_MPI" == "1" ]]; then
  echo "USE_SYSTEM_MPI=1: nessun 'module load' MPI. Cerco mpirun (stesso albero delle lib in /usr/lib64/openmpi/lib)."
  # I nodi compute spesso hanno un PATH piu' corto del login: stesso OpenMPI, ma sotto /usr/lib64/openmpi/bin.
  export PATH="/usr/lib64/openmpi/bin:/usr/bin:/bin:$PATH"
  if ! command -v mpirun >/dev/null 2>&1; then
    echo "FATAL: mpirun assente sotto /usr su questo nodo (normale sui compute ncpu*)."
    echo "Rilancia SENZA USE_SYSTEM_MPI=1, cosi carica openmpi da module (vedi sotto in questo file)."
    exit 1
  fi
  echo "mpirun OK: $(command -v mpirun)"
  mpirun --version 2>&1 | head -5
else
  # Elenco: openmpi/gcc-10.3.0/4.1.1, openmpi/intel-19U5/4.1.1-ucx-1.10.1, ...
  OPENMPI_MODULE="${OPENMPI_MODULE:-openmpi/gcc-10.3.0/4.1.1}"
  export OPENMPI_MODULE
  # openmpi/gcc-10.3.0 richiede gcc/10.3.0; un altro gcc gia caricato (es. da .bashrc) da conflitto Lmod.
  echo "Prima di openmpi: scarico eventuali gcc/intel/mpi in conflitto (come suggerisce Lmod)."
  module unload gcc 2>/dev/null || true
  module unload intel 2>/dev/null || true
  # Rimuove lo stack "mpi/openmpi-x86_64" (path /usr rotto) se ereditato da login; lascia solo /softs
  module unload mpi/openmpi-x86_64 2>/dev/null || true
  module unload mpi 2>/dev/null || true
  if ! module load "$OPENMPI_MODULE"; then
    echo "FATAL: module load $OPENMPI_MODULE fallito. Controlla: module avail openmpi  (sul login o salloc ncpu001)"
    module avail openmpi 2>&1 | head -80 || true
    exit 1
  fi
  module list 2>&1
  command -v mpirun
  mpirun --version 2>&1 | head -3
  # Prefisso OpenMPI del modulo (libmpi, libopen-*, ... /softs non ha libmpi_cxx)
  MPIROOT="$(dirname "$(dirname "$(command -v mpirun)")")"
  export LD_LIBRARY_PATH="${MPIROOT}/lib64:${MPIROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "MPIROOT=$MPIROOT"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<vuoto>}"
fi

# DistributedBench in release_l2: se compilato con g++ 11 (non /usr/bin/g++), serve libstdc++
# di quel GCC *prima* di /lib64, altrimenti: GLIBCXX_3.4.29 not found.
if [[ -d /softs/gcc/11.3.0/lib64 ]]; then
  export LD_LIBRARY_PATH="/softs/gcc/11.3.0/lib64:/softs/gcc/11.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "CXX runtime: prepended /softs/gcc/11.3.0/lib64 for libstdc++ (g++-11 builds)"
fi

# Inoltra LD_LIBRARY_PATH e PATH a tutti i rank (-x): altrimenti il secondo nodo non vede le lib del modulo.
# Se ldd su DistributedBench mostra ancora libmpi_cxx: la build /softs OpenMPI non include quella lib;
# serve ricompilare con message-queue linkato a MPI::MPI_C (fix in external/message-queue/CMakeLists.txt).
if [[ "$USE_SYSTEM_MPI" == "1" ]]; then
  export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
MPIRUN_EXPORT=(-x LD_LIBRARY_PATH -x PATH)

# Input directories for this smoke run
DATA_DIR="/gpfs/users/delgaudio/exp_eval/gp-ann/data/smoke_100k"
ROUTING_DIR="/gpfs/users/delgaudio/exp_eval/toy"
BENCH="/gpfs/users/delgaudio/exp_eval/gp-ann/release_l2/DistributedBench"

echo "========== allocazione slurm =========="
echo "NODELIST: $SLURM_NODELIST"
echo "NNODES=$SLURM_NNODES  NTASKS=$SLURM_NTASKS  NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "========== controllo: stesso n? ========="
if ! [[ -x "$BENCH" ]]; then
  echo "ERRORE: eseguibile non trovato o non eseguibile: $BENCH" >&2
  echo "Compila e verifica: ls -l $BENCH" >&2
  exit 1
fi

# Verifica che i due rank vadano su host diversi (nomi hostname diversi)
mpirun -n "$SLURM_NTASKS" "${MPIRUN_EXPORT[@]}" bash -lc 'echo "rank=$OMPI_COMM_WORLD_RANK host=$(hostname)"'

echo "========== DistributedBench =========="
# Args: base query ground_truth k metis router
mpirun -n "$SLURM_NTASKS" "${MPIRUN_EXPORT[@]}" \
  "$BENCH" \
  "$DATA_DIR/random_100k.fbin" \
  "$DATA_DIR/random_100k.query.50.fbin" \
  "$DATA_DIR/random_100k.query.50.ground_truth.bin" \
  10 \
  "$DATA_DIR/random_100k.partition.k=2.GP" \
  "$DATA_DIR/random_100k.partition.k=2.GP.routing_index"

echo "========== fine job =========="
