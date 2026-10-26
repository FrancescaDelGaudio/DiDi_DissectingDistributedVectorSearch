#!/usr/bin/env python3
"""Extract TTI smoke subsets (1k base, 100 queries) for login-node testing.

Reads:
  yangt2i10M.fbin       — 10M base (dim=200)
  yangt2iquery10K.fbin  — 10K queries (dim=200)  [NOT yangt2iquery.bin]

Writes under --out-dir:
  base_1k.fvecs, base_1k.fbin
  query_100.fbin
  gt_100q_k10_ip.bin, gt_100q_k100_ip.bin  (brute-force MIPS GT, normalised IP)

No numpy required.
"""

import argparse
import math
import struct
import sys
from array import array
from pathlib import Path
from typing import Tuple

TTI = Path("/gpfs/scratch/delgaudio/vectors/TTI")


def read_fbin_prefix(path: Path, max_n: int) -> Tuple[int, int, array]:
    with path.open("rb") as f:
        n, d = struct.unpack("ii", f.read(8))
        take = min(n, max_n)
        buf = array("f")
        buf.frombytes(f.read(take * d * 4))
    return take, d, buf


def write_fbin(path: Path, n: int, d: int, data: array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("ii", n, d))
        f.write(data.tobytes())


def write_fvecs(path: Path, n: int, d: int, data: array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for i in range(n):
            off = i * d
            f.write(struct.pack("i", d))
            f.write(data[off : off + d].tobytes())


def vec_at(data: array, d: int, i: int) -> Tuple[float, ...]:
    off = i * d
    return tuple(data[off : off + d])


def normalize_rows(data: array, n: int, d: int) -> None:
    for i in range(n):
        off = i * d
        norm_sq = sum(data[off + j] * data[off + j] for j in range(d))
        if norm_sq <= 0.0:
            continue
        inv = 1.0 / math.sqrt(norm_sq)
        for j in range(d):
            data[off + j] *= inv


def dot(a: Tuple[float, ...], b_start: int, data: array, d: int) -> float:
    s = 0.0
    for j in range(d):
        s += a[j] * data[b_start + j]
    return s


def compute_gt_ip(base: array, nb: int, queries: array, nq: int, d: int, k: int) -> Tuple[array, array]:
    normalize_rows(base, nb, d)
    normalize_rows(queries, nq, d)

    ids = array("i", [0] * (nq * k))
    dists = array("f", [0.0] * (nq * k))

    for qi in range(nq):
        q = vec_at(queries, d, qi)
        scored = []
        for bi in range(nb):
            bo = bi * d
            scored.append((-dot(q, bo, base, d), bi))  # lower = better
        scored.sort(key=lambda x: x[0])
        for j in range(k):
            ids[qi * k + j] = scored[j][1]
            dists[qi * k + j] = scored[j][0]
    return ids, dists


def write_gt_bin(path: Path, nq: int, k: int, ids: array, dists: array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("ii", nq, k))
        f.write(ids.tobytes())
        f.write(dists.tobytes())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tti-dir", type=Path, default=TTI)
    p.add_argument("--out-dir", type=Path, default=Path("/gpfs/scratch/delgaudio/dann/tti_smoke_1k"))
    p.add_argument("--num-base", type=int, default=1000)
    p.add_argument("--num-queries", type=int, default=100)
    args = p.parse_args()

    base_src = args.tti_dir / "yangt2i10M.fbin"
    query_src = args.tti_dir / "yangt2iquery10K.fbin"
    if not base_src.exists():
        sys.exit(f"Missing base: {base_src}")
    if not query_src.exists():
        sys.exit(f"Missing queries: {query_src}")

    nb, d, base = read_fbin_prefix(base_src, args.num_base)
    nq, dq, queries = read_fbin_prefix(query_src, args.num_queries)
    if d != dq:
        sys.exit(f"Dim mismatch: base={d} query={dq}")

    out = args.out_dir
    write_fvecs(out / "base_1k.fvecs", nb, d, base)
    write_fbin(out / "base_1k.fbin", nb, d, base)
    write_fbin(out / "query_100.fbin", nq, d, queries)
    print(f"Wrote {out}/base_1k.fvecs, base_1k.fbin, query_100.fbin  (nb={nb} nq={nq} dim={d})")

    for k, tag in ((10, "k10"), (100, "k100")):
        base_copy = array("f", base)
        query_copy = array("f", queries)
        ids, dists = compute_gt_ip(base_copy, nb, query_copy, nq, d, k)
        gt_path = out / f"gt_100q_{tag}_ip.bin"
        write_gt_bin(gt_path, nq, k, ids, dists)
        print(f"Wrote {gt_path}  (k={k}, metric=ip normalised)")

    print(f"\nDone. Smoke data in {out}")


if __name__ == "__main__":
    main()
