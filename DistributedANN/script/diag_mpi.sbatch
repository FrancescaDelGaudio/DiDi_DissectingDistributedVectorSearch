#!/usr/bin/bash
#SBATCH --job-name=dann-diag
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -p ncpushort
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.log

# Find the launcher that yields a single MPI_COMM_WORLD of size 2.
module purge
module load gcc/10.3.0
module load openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BIN=/gpfs/users/delgaudio/exp_eval/DistributedANN/build/bin/mpi_size_test
echo "alloc nodes: ${SLURM_JOB_NODELIST}  ntasks=${SLURM_NTASKS}"

echo; echo "### A) mpirun -n 2 (current approach)"
mpirun -n 2 "${BIN}" 2>&1 || echo "A failed rc=$?"

echo; echo "### B) mpirun (no -n, use slots)"
mpirun "${BIN}" 2>&1 || echo "B failed rc=$?"

echo; echo "### C) srun --mpi=pmi2 -n 2"
srun --mpi=pmi2 -n 2 "${BIN}" 2>&1 || echo "C failed rc=$?"

echo; echo "### D) srun (default mpi) -n 2"
srun -n 2 "${BIN}" 2>&1 || echo "D failed rc=$?"

echo; echo "### E) mpirun --map-by node -n 2"
mpirun --map-by node -n 2 "${BIN}" 2>&1 || echo "E failed rc=$?"

echo "### done"
