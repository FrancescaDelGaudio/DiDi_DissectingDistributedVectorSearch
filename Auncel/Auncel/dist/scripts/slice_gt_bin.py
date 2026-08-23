#!/usr/bin/env python3
"""Slice gp-ann ground_truth.bin to first N queries (ids + dists)."""

import argparse
import struct
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-queries", type=int, required=True)
    args = p.parse_args()

    with args.input.open("rb") as f:
        nq, k = struct.unpack("<II", f.read(8))
        if args.num_queries > nq:
            raise SystemExit(
                f"--num-queries {args.num_queries} > GT nq {nq} in {args.input}"
            )
        ids = f.read(nq * k * 4)
        dists = f.read(nq * k * 4)
        if len(dists) != nq * k * 4:
            raise SystemExit(f"unexpected GT size in {args.input}")

    n = args.num_queries
    out_ids = ids[: n * k * 4]
    out_dists = dists[: n * k * 4]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        out.write(struct.pack("<II", n, k))
        out.write(out_ids)
        out.write(out_dists)

    print(f"Wrote {args.output}  nq={n}  k={k}  (from {args.input} nq={nq})")


if __name__ == "__main__":
    main()
