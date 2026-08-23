#!/usr/bin/bash
#SBATCH --job-name=dann-index
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.log

# ===========================================================================
# DistributedANN — FULL INDEXING PIPELINE (Steps 1 -> 6) in one job.
# Produces every persistent artifact in INDEX_DIR. Run once, reuse for all
# query experiments.
#
# Submit:  sbatch script/build_all.sbatch
#   override params, e.g.:
#   K=64 EPSILON=0.3 sbatch --export=ALL,K=64,EPSILON=0.3 script/build_all.sbatch
# ===========================================================================

module purge
module load gcc/10.3.0
module load openmpi/gcc-10.3.0/4.1.1
export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BASE_PATH="${BASE_PATH:-/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid/base.fvecs}"
INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_deep10m_k${K:-32}}"
BUILD_DIR="${BUILD_DIR:-/gpfs/users/delgaudio/exp_eval/DistributedANN/build}"

# Step 1 (k-means + epsilon-replication)
K="${K:-32}"; EPSILON="${EPSILON:-0.2}"; MAX_REPLICAS="${MAX_REPLICAS:-3}"
REPLICA_R="${REPLICA_R:-0}"
KMEANS_ITER="${KMEANS_ITER:-20}"; KMEANS_TOL="${KMEANS_TOL:-1e-4}"
# Step 2/3 (Vamana + stitching)
R="${R:-72}"; L="${L:-100}"; ALPHA="${ALPHA:-1.2}"; PASSES="${PASSES:-2}"
# Step 4 (PQ)
M="${M:-16}"; TRAIN_SIZE="${TRAIN_SIZE:-100000}"; ITERS="${ITERS:-25}"
# Step 5 (head index)
HEAD_FRAC="${HEAD_FRAC:-0.05}"; HNSW_M="${HNSW_M:-16}"; EFC="${EFC:-200}"; METHOD="${METHOD:-random}"
METRIC="${METRIC:-l2}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
NT="${SLURM_NTASKS}"
mkdir -p "${INDEX_DIR}"

# Robust MPI launch: explicit host list + one rank per node (matches the proven
# Distributed-HNSW pattern). Avoids the isolated-singleton wireup fallback that
# plain "mpirun -n N" can hit on this cluster. --bind-to none lets OpenMP use
# all cpus-per-task cores.
_HOSTS=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | awk '{printf "%s:1,",$1}' | sed 's/,$//')
MPI=(mpirun -np "${NT}" --host "${_HOSTS}" --map-by ppr:1:node --bind-to none)
echo "launcher: ${MPI[*]}"

echo "############################################################"
echo "# DistributedANN full indexing pipeline"
echo "#   BASE_PATH : ${BASE_PATH}"
echo "#   INDEX_DIR : ${INDEX_DIR}"
echo "#   nodes=${SLURM_JOB_NUM_NODES} ntasks=${NT} omp=${OMP_NUM_THREADS}"
echo "#   K=${K} eps=${EPSILON} replica_r=${REPLICA_R} R=${R} m(pq)=${M} head_frac=${HEAD_FRAC} metric=${METRIC}"
echo "############################################################"

set -e
run() { echo; echo ">>> $*"; date '+%F %T'; "$@"; }

# --- Preflight: verify MPI wireup before doing any real (long) work. -------
# Catches the "isolated singleton" fallback (each rank sees size=1) that would
# otherwise silently run the whole pipeline non-distributed.
echo; echo "==================== PREFLIGHT: MPI wireup ===================="
WSIZE=$("${MPI[@]}" "${BUILD_DIR}/bin/mpi_size_test" | sed -n 's/.*size=\([0-9]*\).*/\1/p' | head -1)
echo "MPI_COMM_WORLD size reported = ${WSIZE} (expected ${NT})"
if [ "${WSIZE}" != "${NT}" ]; then
    echo "FATAL: MPI wireup failed (world size ${WSIZE} != ${NT} tasks)."
    echo "       The pipeline would run as isolated singletons (non-distributed)."
    echo "       Re-submit (often a transient/node-specific OOB failure)."
    exit 2
fi

echo; echo "==================== STEP 1: clustering ===================="
run "${MPI[@]}" "${BUILD_DIR}/bin/build_index" \
    --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
    --K "${K}" --kmeans_iter "${KMEANS_ITER}" --kmeans_tol "${KMEANS_TOL}" \
    --epsilon "${EPSILON}" --max_replicas "${MAX_REPLICAS}" \
    --replica_r "${REPLICA_R}" --metric "${METRIC}"

echo; echo "==================== STEP 2: per-cluster Vamana ===================="
run "${MPI[@]}" "${BUILD_DIR}/bin/build_graph" \
    --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
    --R "${R}" --L "${L}" --alpha "${ALPHA}" --passes "${PASSES}" \
    --threads "${OMP_NUM_THREADS}"

echo; echo "==================== STEP 3: global stitching ===================="
run "${MPI[@]}" "${BUILD_DIR}/bin/build_stitch" \
    --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
    --R "${R}" --alpha "${ALPHA}"

echo; echo "==================== STEP 4: PQ ===================="
run "${MPI[@]}" "${BUILD_DIR}/bin/build_pq" \
    --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
    --m "${M}" --train_size "${TRAIN_SIZE}" --iters "${ITERS}"

echo; echo "==================== STEP 5: head index ===================="
run "${MPI[@]}" "${BUILD_DIR}/bin/build_head" \
    --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
    --head_frac "${HEAD_FRAC}" --M "${HNSW_M}" --efc "${EFC}" --method "${METHOD}"

echo; echo "==================== STEP 6: KV store ===================="
run "${MPI[@]}" "${BUILD_DIR}/bin/build_kvstore" \
    --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}"

echo; echo "############################################################"
echo "# INDEXING COMPLETE — artifacts in ${INDEX_DIR}"
ls -lh "${INDEX_DIR}"
echo "############################################################"
