#!/bin/bash
# Download Cohere embed-english-v3 BEIR embeddings (corpus, queries, qrels) for selected datasets.
set -u
OUT=/gpfs/scratch/delgaudio/beir/cohere
BASE=https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/main
API=https://huggingface.co/api/datasets/CohereLabs/beir-embed-english-v3/tree/main
for p in fever/queries fever/qrels climate-fever/queries climate-fever/qrels quora/queries quora/qrels quora/corpus fever/corpus; do
  for f in $(curl -s "$API/$p" | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin)]"); do
    mkdir -p "$OUT/$(dirname $f)"
    [ -s "$OUT/$f" ] && { echo "skip $f"; continue; }
    curl -sfL --retry 5 -o "$OUT/$f.part" "$BASE/$f" && mv "$OUT/$f.part" "$OUT/$f" && echo "ok $f $(stat -c %s $OUT/$f)" || echo "FAIL $f"
  done
done
echo ALLDONE
