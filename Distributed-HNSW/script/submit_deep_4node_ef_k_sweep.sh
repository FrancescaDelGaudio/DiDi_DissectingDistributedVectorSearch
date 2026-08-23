#!/usr/bin/bash
# Wrapper: DEEP10M recall@k=10 on 4 nodes. See submit_4node_ef_k_sweep.sh for all variants.
exec "$(dirname "${BASH_SOURCE[0]}")/submit_4node_ef_k_sweep.sh" deep 10 "$@"
