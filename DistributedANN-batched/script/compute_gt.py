#!/usr/bin/env python3
"""Compute exact k-NN ground truth in gp-ann .bin format.

Output layout (little-endian):
  int32 nq, int32 k, int32 ids[nq*k], float32 dists[nq*k]

Example:
  python3 script/compute_gt.py \
    --base /gpfs/scratch/delgaudio/dann/test/base_1m.fvecs \
    --query /gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin \
    --output /gpfs/scratch/delgaudio/dann/test/gt_1m_100q_k10.bin \
    --k 10 --num_queries 100
"""

import argparse
import struct
import time
from pathlib import Path

import numpy as np


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:].astype(np.float32, copy=False)


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
        return np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)


def write_gt_bin(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    nq, k = ids.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("ii", nq, k))
        f.write(ids.astype(np.int32, copy=False).tobytes())
        f.write(dists.astype(np.float32, copy=False).tobytes())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help=".fvecs or .fbin base")
    p.add_argument("--query", type=Path, required=True, help=".fvecs or .fbin queries")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--num_queries", type=int, default=0, help="0 = all queries")
    p.add_argument("--num_base", type=int, default=0, help="0 = all base vectors")
    p.add_argument(
        "--metric",
        choices=("l2", "ip"),
        default="l2",
        help="l2: squared L2 distance; ip: inner product (vectors L2-normalised)",
    )
    args = p.parse_args()

    base = read_fbin(args.base) if args.base.suffix == ".fbin" else read_fvecs(args.base)
    queries = read_fbin(args.query) if args.query.suffix == ".fbin" else read_fvecs(args.query)
    if args.num_base > 0:
        base = base[: args.num_base]
    if args.num_queries > 0:
        queries = queries[: args.num_queries]
    nq, dim = queries.shape
    nb = base.shape[0]
    k = args.k

    if args.metric == "ip":
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
        queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)

    ids = np.empty((nq, k), dtype=np.int32)
    dists = np.empty((nq, k), dtype=np.float32)

    t0 = time.time()
    for i in range(nq):
        if args.metric == "l2":
            diff = base - queries[i]
            score = np.einsum("ij,ij->i", diff, diff, optimize=True)
            idx = np.argpartition(score, k)[:k]
            order = idx[np.argsort(score[idx])]
            ids[i] = order
            dists[i] = score[order]
        else:
            score = base @ queries[i]
            idx = np.argpartition(score, -k)[-k:]
            order = idx[np.argsort(-score[idx])]
            ids[i] = order
            dists[i] = -score[order]  # lower = better (matches run_query min-heap)

    write_gt_bin(args.output, ids, dists)
    print(
        f"Wrote {args.output}  nq={nq} nb={nb} k={k} dim={dim} metric={args.metric}"
        f"  ({time.time()-t0:.1f}s)"
    )


if __name__ == "__main__":
    main()
