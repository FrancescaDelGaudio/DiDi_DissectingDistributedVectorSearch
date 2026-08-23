#!/usr/bin/bash
# ===========================================================================
# TTI MIPS end-to-end smoke test on the login node (no Slurm).
#
# Validates: indexing steps 1-6 + distributed query with METRIC=ip.
# Data: 1000 TTI base vectors + 100 queries + brute-force IP GT.
#
# Usage:
#   ./script/run_tti_mips_smoke.sh
#   NT=2 ./script/run_tti_mips_smoke.sh
#   SKIP_INDEX=1 ./script/run_tti_mips_smoke.sh   # query only (index exists)
#   SKIP_BUILD=1 ./script/run_tti_mips_smoke.sh   # skip cmake if build_ip ready
#
# Requires: module load gcc/10.3.0 openmpi/gcc-10.3.0/4.1.1
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="${ROOT}/script"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build_ip}"

DATA_DIR="${DATA_DIR:-/gpfs/scratch/delgaudio/dann/tti_smoke_1k}"
BASE_PATH="${BASE_PATH:-${DATA_DIR}/base_1k.fvecs}"
QUERY_PATH="${QUERY_PATH:-${DATA_DIR}/query_100.fbin}"
GT_PATH="${GT_PATH:-${DATA_DIR}/gt_100q_k10_ip.bin}"
INDEX_DIR="${INDEX_DIR:-/gpfs/scratch/delgaudio/dann/index_tti_smoke_1k}"

NT="${NT:-4}"
METRIC="${METRIC:-ip}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_INDEX="${SKIP_INDEX:-0}"

# Indexing params tuned for N=1000, dim=200.
K="${K:-16}"
EPSILON="${EPSILON:-0.2}"
MAX_REPLICAS="${MAX_REPLICAS:-3}"
KMEANS_ITER="${KMEANS_ITER:-10}"
KMEANS_TOL="${KMEANS_TOL:-1e-4}"
R="${R:-32}"
L="${L:-64}"
ALPHA="${ALPHA:-1.2}"
PASSES="${PASSES:-2}"
M="${M:-16}"
TRAIN_SIZE="${TRAIN_SIZE:-1000}"
ITERS="${ITERS:-15}"
HEAD_FRAC="${HEAD_FRAC:-0.05}"
HNSW_M="${HNSW_M:-16}"
EFC="${EFC:-100}"
METHOD="${METHOD:-random}"

# Query params (small beam is enough for 1k nodes).
Q_K="${Q_K:-10}"
Q_L="${Q_L:-100}"
Q_BW="${Q_BW:-32}"
Q_H="${Q_H:-5}"
Q_K_HEAD="${Q_K_HEAD:-50}"
Q_EF_HEAD="${Q_EF_HEAD:-100}"
WARMUP="${WARMUP:-2}"
NUM_QUERIES="${NUM_QUERIES:-100}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

log() { echo ""; echo ">>> $*"; date '+%F %T'; }

mpi_run() {
    if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_JOB_NODELIST:-}" ]; then
        local _hosts
        _hosts="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | awk '{printf "%s:1,",$1}' | sed 's/,$//')"
        mpirun -np "${NT}" --host "${_hosts}" --map-by ppr:1:node --bind-to none "$@"
    else
        mpirun -n "${NT}" "$@"
    fi
}

load_modules() {
    if command -v module >/dev/null 2>&1; then
        module purge 2>/dev/null || true
        module load gcc/10.3.0
        module load openmpi/gcc-10.3.0/4.1.1
    fi
    export LD_LIBRARY_PATH="/softs/gcc/10.3.0/lib64:/softs/gcc/10.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

ensure_build() {
    if [ "${SKIP_BUILD}" = "1" ] && [ -x "${BUILD_DIR}/bin/build_index" ]; then
        echo "### SKIP_BUILD=1, using ${BUILD_DIR}/bin"
        return
    fi
    log "Configure + build (MIPS binaries in build_ip/)"
    mkdir -p "${BUILD_DIR}"
    (cd "${BUILD_DIR}" && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j "${OMP_NUM_THREADS}")
}

ensure_data() {
    if [ "${SKIP_DATA}" = "1" ] && [ -f "${BASE_PATH}" ] && [ -f "${QUERY_PATH}" ] && [ -f "${GT_PATH}" ]; then
        echo "### SKIP_DATA=1, using ${DATA_DIR}"
        return
    fi
    log "Prepare TTI smoke subsets + IP ground truth"
    python3 "${SCRIPT_DIR}/prepare_tti_smoke.py" --out-dir "${DATA_DIR}"
}

mpi_preflight() {
  log "MPI preflight (expect world size=${NT})"
  WSIZE="$(
    mpi_run "${BUILD_DIR}/bin/mpi_size_test" 2>/dev/null \
      | sed -n 's/.*size=\([0-9]*\).*/\1/p' | head -1
  )"
  echo "MPI_COMM_WORLD size = ${WSIZE:-?} (expected ${NT})"
  if [ "${WSIZE}" != "${NT}" ]; then
    echo "FATAL: MPI wireup failed."
    echo "On Slurm: check host list / ntasks. On login node: try NT=2."
    exit 2
  fi
}

run_indexing() {
    mkdir -p "${INDEX_DIR}"
    log "Indexing METRIC=${METRIC}  N~1000  W=${NT}  -> ${INDEX_DIR}"

    mpi_run "${BUILD_DIR}/bin/build_index" \
        --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
        --K "${K}" --kmeans_iter "${KMEANS_ITER}" --kmeans_tol "${KMEANS_TOL}" \
        --epsilon "${EPSILON}" --max_replicas "${MAX_REPLICAS}" --metric "${METRIC}"

    mpi_run "${BUILD_DIR}/bin/build_graph" \
        --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
        --R "${R}" --L "${L}" --alpha "${ALPHA}" --passes "${PASSES}" \
        --threads "${OMP_NUM_THREADS}"

    mpi_run "${BUILD_DIR}/bin/build_stitch" \
        --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
        --R "${R}" --alpha "${ALPHA}"

    mpi_run "${BUILD_DIR}/bin/build_pq" \
        --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
        --m "${M}" --train_size "${TRAIN_SIZE}" --iters "${ITERS}"

    mpi_run "${BUILD_DIR}/bin/build_head" \
        --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}" \
        --head_frac "${HEAD_FRAC}" --M "${HNSW_M}" --efc "${EFC}" --method "${METHOD}"

    mpi_run "${BUILD_DIR}/bin/build_kvstore" \
        --base_path "${BASE_PATH}" --index_dir "${INDEX_DIR}"

    log "Indexing artifacts"
    ls -lh "${INDEX_DIR}"
    if [ -f "${INDEX_DIR}/meta.json" ]; then
        echo "--- meta.json ---"
        cat "${INDEX_DIR}/meta.json"
    fi
}

run_query() {
    log "Query METRIC=${METRIC}  k=${Q_K} BW=${Q_BW} H=${Q_H} nq=${NUM_QUERIES}"
    mpi_run "${BUILD_DIR}/bin/run_query" \
        --index_dir "${INDEX_DIR}" \
        --query_path "${QUERY_PATH}" \
        --gt_path "${GT_PATH}" \
        --k "${Q_K}" --L "${Q_L}" --BW "${Q_BW}" --H "${Q_H}" \
        --k_head "${Q_K_HEAD}" --ef_head "${Q_EF_HEAD}" \
        --warmup "${WARMUP}" --num_queries "${NUM_QUERIES}"
}

main() {
    echo "############################################################"
    echo "# DistributedANN TTI MIPS smoke (${SLURM_JOB_ID:+Slurm ${SLURM_JOB_NUM_NODES} nodes}${SLURM_JOB_ID:-login node})"
    echo "#   DATA_DIR  : ${DATA_DIR}"
    echo "#   INDEX_DIR : ${INDEX_DIR}"
    echo "#   BUILD_DIR : ${BUILD_DIR}"
    echo "#   METRIC    : ${METRIC}  NT=${NT}  OMP=${OMP_NUM_THREADS}"
    echo "############################################################"

    load_modules
    ensure_build
    ensure_data
    mpi_preflight

    if [ "${SKIP_INDEX}" != "1" ]; then
        run_indexing
    else
        echo "### SKIP_INDEX=1, reusing ${INDEX_DIR}"
    fi

    run_query

    log "SMOKE TEST PASSED (no crash; check recall@k above)"
    echo "############################################################"
}

main "$@"
