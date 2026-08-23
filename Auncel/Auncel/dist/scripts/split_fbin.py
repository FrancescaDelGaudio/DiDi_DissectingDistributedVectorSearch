#!/usr/bin/env python3
"""Split a .fbin vector file into N equal shards (0db.fbin .. (N-1)db.fbin)."""

import argparse
import struct
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--chunk-vectors", type=int, default=10000,
                   help="Vectors per read/write chunk (default 10000)")
    args = p.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input.open("rb") as fin:
        n, d = struct.unpack("<II", fin.read(8))
        if n % args.num_shards != 0:
            raise SystemExit(
                f"n={n} not divisible by num_shards={args.num_shards}"
            )
        per_shard = n // args.num_shards
        row_bytes = d * 4
        chunk = args.chunk_vectors

        print(f"Split {args.input.name}: n={n} d={d} -> "
              f"{args.num_shards} shards x {per_shard}")

        for shard in range(args.num_shards):
            out_path = args.output_dir / f"{shard}db.fbin"
            with out_path.open("wb") as fout:
                fout.write(struct.pack("<II", per_shard, d))
                remaining = per_shard
                while remaining > 0:
                    take = min(chunk, remaining)
                    buf = fin.read(take * row_bytes)
                    if len(buf) != take * row_bytes:
                        raise SystemExit("unexpected EOF while reading input")
                    fout.write(buf)
                    remaining -= take
            print(f"  wrote {out_path} ({out_path.stat().st_size} bytes)")

    print("Done.")


if __name__ == "__main__":
    main()
