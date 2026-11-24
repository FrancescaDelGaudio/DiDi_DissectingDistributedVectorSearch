#!/usr/bin/env python3
"""Cohere BEIR embeddings -> numpy + exact top-10 GT (inner product, embeddings are unit-norm).
usage: beir_gt.py prep <corpus_ds>                      corpus parquet -> base.npy (float32) + ids.npy
       beir_gt.py gt <corpus_ds> <qds>:<split> [...]    queries -> q_<qds>_<split>.npy, gt_*.npy (int32 top10), qrels mapped"""
import sys, time, os
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
SRC = Path("/gpfs/scratch/delgaudio/beir/cohere"); OUT = Path("/gpfs/scratch/delgaudio/beir/np")
K, D = 10, 1024
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
def emb(t): return t.column("emb").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, D)

def prep(ds):
    o = OUT / ds; o.mkdir(parents=True, exist_ok=True)
    files = sorted((SRC / ds / "corpus").glob("*.parquet"))
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files); log(ds, "corpus rows", n)
    X = np.lib.format.open_memmap(o / "base.npy", mode="w+", dtype=np.float32, shape=(n, D)); ids = []; s = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=200_000, columns=["_id", "emb"]):
            e = b.column(1).flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, D)
            X[s:s + len(e)] = e; s += len(e); ids.extend(b.column(0).to_pylist())
        log(f.name, s)
    X.flush(); np.save(o / "ids.npy", np.array(ids, dtype=object), allow_pickle=True); log("prep done", s)

def gt(ds, specs):
    o = OUT / ds; X = np.load(o / "base.npy"); N = len(X); log("base", X.shape)
    idx = {c: i for i, c in enumerate(np.load(o / "ids.npy", allow_pickle=True))}
    for spec in specs:
        qds, split = spec.split(":"); tag = f"{qds}_{split}"
        t = pq.read_table(SRC / qds / "queries" / f"{split}.parquet"); Q = emb(t); qids = t.column("_id").to_pylist()
        np.save(o / f"q_{tag}.npy", Q); np.save(o / f"qids_{tag}.npy", np.array(qids, dtype=object), allow_pickle=True)
        qp = SRC / qds / "qrels" / f"{split}.parquet"
        if qp.exists():
            r = pq.read_table(qp).to_pydict(); qpos = {q: i for i, q in enumerate(qids)}
            pairs = [(qpos[q], idx[c]) for q, c, s in zip(r["query_id"], r["corpus_id"], r["score"]) if s > 0 and q in qpos and c in idx]
            miss = sum(1 for c in r["corpus_id"] if c not in idx)
            np.save(o / f"qrels_{tag}.npy", np.array(pairs, dtype=np.int64)); log(tag, "qrels", len(pairs), "corpus ids missing", miss)
        nq = len(Q); I = np.empty((nq, K), np.int32); S = np.empty((nq, K), np.float32); qb, cb = 2048, 1_000_000
        for a in range(0, nq, qb):
            q = Q[a:a + qb]; bi = np.empty((len(q), 0), np.int64); bs = np.empty((len(q), 0), np.float32)
            for c in range(0, N, cb):
                sc = q @ X[c:c + cb].T
                p = np.argpartition(-sc, K, axis=1)[:, :K]
                bi = np.hstack([bi, p + c]); bs = np.hstack([bs, np.take_along_axis(sc, p, 1)])
            o2 = np.argsort(-bs, axis=1)[:, :K]
            I[a:a + qb] = np.take_along_axis(bi, o2, 1); S[a:a + qb] = np.take_along_axis(bs, o2, 1)
            if (a // qb) % 10 == 0: log(tag, a + len(q), "/", nq)
        np.save(o / f"gt_{tag}.npy", I); np.save(o / f"gtscore_{tag}.npy", S); log(tag, "GT done")

if __name__ == "__main__":
    if sys.argv[1] == "prep": prep(sys.argv[2])
    else: gt(sys.argv[2], sys.argv[3:])
