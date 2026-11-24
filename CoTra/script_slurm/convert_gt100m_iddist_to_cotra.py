#!/usr/bin/env python3
"""
Convert GT.100M-style ground truth to CoTra id-only format.

GT.100M layout (same as gp-ann ReadGroundTruth in points_io.cpp):
  header: uint32 num_queries, uint32 k
  block 1: num_queries * k uint32 neighbor ids (row-major by query)
  block 2: num_queries * k float32 distances (same order)

CoTra expects: header + num_queries * k uint32 ids only (no distances).

Usage:
  python3 convert_gt100m_iddist_to_cotra.py \\
    /gpfs/scratch/delgaudio/vectors/DEEP/GT.100M \\
    /gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep100M_10kqueries_k100_cotra.bin \\
    --num_queries 10000
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def convert(src: Path, dst: Path, num_queries: int | None) -> None:
    with src.open("rb") as f:
        n, k = struct.unpack("II", f.read(8))
        n_out = num_queries if num_queries is not None else n
        if n_out > n:
            raise SystemExit(f"requested num_queries={n_out} > file n={n}")
        id_bytes = n * k * 4
        id_block = f.read(id_bytes)
        if len(id_block) != id_bytes:
            raise SystemExit("truncated id block")
        ids = struct.unpack(f"<{n * k}I", id_block)
        ids_out = list(ids[: n_out * k])
        rest = f.read()
        expected_dist = n * k * 4
        if len(rest) < expected_dist:
            print(
                f"warning: distance block size {len(rest)} expected {expected_dist}",
                file=sys.stderr,
            )

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("wb") as out:
        out.write(struct.pack("II", n_out, k))
        out.write(struct.pack(f"<{len(ids_out)}I", *ids_out))
    tmp.replace(dst)
    mx = max(ids_out) if ids_out else 0
    bad = sum(1 for i in ids_out if i >= 100_000_000)
    print(f"Wrote {dst}  n={n_out} k={k}  max_id={mx}  bad_id>=100M={bad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="truncate to first N queries (default: all in source header)",
    )
    args = ap.parse_args()
    convert(Path(args.src), Path(args.dst), args.num_queries)


if __name__ == "__main__":
    main()
