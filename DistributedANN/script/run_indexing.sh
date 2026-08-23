#!/usr/bin/bash
# ===========================================================================
# DistributedANN — full indexing pipeline via local mpirun (Steps 1 -> 6).
# For interactive runs / small datasets. On the cluster prefer build_all.sbatch.
#
# Usage:
#   ./run_indexing.sh <base.fvecs> <index_dir> [nranks]
#   K=16 EPSILON=0.3 M=16 ./run_indexing.sh base.fvecs /path/idx 4
# ===========================================================================
set -e

BASE_PATH="${1:?usage: run_indexing.sh <base.fvecs> <index_dir> [nranks]}"
INDEX_DIR="${2:?usage: run_indexing.sh <base.fvecs> <index_dir> [nranks]}"
NT="${3:-4}"
BUILD_DIR="${BUILD_DIR:-$(cd "$(dirname "$0")/../build" && pwd)}"

# Step params (override via env).
K="${K:-32}"; EPSILON="${EPSILON:-0.2}"; MAX_REPLICAS="${MAX_REPLICAS:-3}"
KMEANS_ITER="${KMEANS_ITER:-20}"; KMEANS_TOL="${KMEANS_TOL:-1e-4}"
R="${R:-72}"; L="${L:-100}"; ALPHA="${ALPHA:-1.2}"; PASSES="${PASSES:-2}"
M="${M:-16}"; TRAIN_SIZE="${TRAIN_SIZE:-100000}"; ITERS="${ITERS:-25}"
HEAD_FRAC="${HEAD_FRAC:-0.05}"; HNSW_M="${HNSW_M:-16}"; EFC="${EFC:-200}"; METHOD="${METHOD:-random}"
METRIC="${METRIC:-l2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p "${INDEX_DIR}"
echo "### DistributedANN indexing | base=${BASE_PATH} idx=${INDEX_DIR} nranks=${NT}"
echo "### K=${K} eps=${EPSILON} R=${R} m(pq)=${M} head_frac=${HEAD_FRAC} metric=${METRIC}"

# Preflight: verify MPI wireup (abort if ranks fall back to isolated singletons).
if [ -x "${BUILD_DIR}/bin/mpi_size_test" ]; then
    WSIZE=$(mpirun -n "${NT}" "${BUILD_DIR}/bin/mpi_size_test" | sed -n 's/.*size=\([0-9]*\).*/\1/p' | head -1)
    echo "### preflight: MPI world size=${WSIZE} (expected ${NT})"
    if [ "${WSIZE}" != "${NT}" ]; then
        echo "### FATAL: MPI wireup failed; would run non-distributed. Aborting."; exit 2
    fi
fi

echo "== STEP 1: clustering =="
mpirun -n "${NT}" "${BUILD_DIR}/bin/build_index" --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
    --K "${K}" --kmeans_iter "${KMEANS_ITER}" --kmeans_tol "${KMEANS_TOL}" --epsilon "${EPSILON}" --max_replicas "${MAX_REPLICAS}" --metric "${METRIC}"

echo "== STEP 2: per-cluster Vamana =="
mpirun -n "${NT}" "${BUILD_DIR}/bin/build_graph" --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
    --R "${R}" --L "${L}" --alpha "${ALPHA}" --passes "${PASSES}" --threads "${OMP_NUM_THREADS}"

echo "== STEP 3: global stitching =="
mpirun -n "${NT}" "${BUILD_DIR}/bin/build_stitch" --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
    --R "${R}" --alpha "${ALPHA}"

echo "== STEP 4: PQ =="
mpirun -n "${NT}" "${BUILD_DIR}/bin/build_pq" --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
    --m "${M}" --train_size "${TRAIN_SIZE}" --iters "${ITERS}"

echo "== STEP 5: head index =="
mpirun -n "${NT}" "${BUILD_DIR}/bin/build_head" --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
    --head_frac "${HEAD_FRAC}" --M "${HNSW_M}" --efc "${EFC}" --method "${METHOD}"

echo "== STEP 6: KV store =="
mpirun -n "${NT}" "${BUILD_DIR}/bin/build_kvstore" --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}"

echo "### INDEXING COMPLETE — artifacts in ${INDEX_DIR}"
ls -lh "${INDEX_DIR}"
