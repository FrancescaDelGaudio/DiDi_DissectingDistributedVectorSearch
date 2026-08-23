# Shared paths for random_100k Auncel distributed runs.
# Source from other scripts:  source "$(dirname "$0")/auncel_100k_env.sh"

AUNCEL_DATA_DIR="${AUNCEL_DATA_DIR:-/gpfs/scratch/delgaudio/vectors/toy}"
AUNCEL_LOG_DIR="${AUNCEL_LOG_DIR:-${AUNCEL_DATA_DIR}/logs}"

AUNCEL_BASE="${AUNCEL_BASE:-${AUNCEL_DATA_DIR}/random_100k.fbin}"
AUNCEL_QUERY="${AUNCEL_QUERY:-${AUNCEL_DATA_DIR}/random_100k.query.50.fbin}"
AUNCEL_GT_FULL="${AUNCEL_GT_FULL:-${AUNCEL_DATA_DIR}/random_100k.ground_truth.bin}"
AUNCEL_GT="${AUNCEL_GT:-${AUNCEL_DATA_DIR}/random_100k.ground_truth.50.bin}"

AUNCEL_SHARDS="${AUNCEL_SHARDS:-${AUNCEL_DATA_DIR}/auncel_shards_100k_2w}"
AUNCEL_INDEX_DIR="${AUNCEL_INDEX_DIR:-${AUNCEL_DATA_DIR}/auncel_index_100k_2w}"

# 100k vectors / 2 workers → shard1 global id offset.
AUNCEL_SHARD_OFFSET="${AUNCEL_SHARD_OFFSET:-50000}"

# From GT header (k=10); override with env if needed.
AUNCEL_K="${AUNCEL_K:-10}"
AUNCEL_INDEX_KEY="${AUNCEL_INDEX_KEY:-IVF256,Flat}"

export AUNCEL_DATA_DIR AUNCEL_LOG_DIR AUNCEL_BASE AUNCEL_QUERY AUNCEL_GT AUNCEL_GT_FULL
export AUNCEL_SHARDS AUNCEL_INDEX_DIR AUNCEL_SHARD_OFFSET AUNCEL_K AUNCEL_INDEX_KEY
