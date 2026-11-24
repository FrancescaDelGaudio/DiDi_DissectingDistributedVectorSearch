#!/usr/bin/env python3
"""
Prepare MSMARCO Web Search (101,070,374 x 768 float, MIPS) for CoTra 100M.

CoTra requires npts == -s * 1e6 exactly, so we keep the first 100M base vectors.

  base: base.vectors.bin -> base.100M.fbin (header 100M/768 + first 100M vectors)
  gt:   truth.txt (top-100 ids over full 101M, one query per line) ->
        CoTra id-only GT (uint32 n, uint32 k, n*k uint32 ids).
        Ids >= 100M are dropped; the remaining ids, in order, are the exact
        top-k over the 100M subset. k = min kept per query (93 on this GT).

Usage:
  python3 prepare_msmarco100M.py base
  python3 prepare_msmarco100M.py gt
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

DIR = Path("/gpfs/scratch/delgaudio/vectors/MSMARCOWebSearch")
NPTS = 100_000_000
DIM = 768


def make_base(src: Path, dst: Path) -> None:
    payload = NPTS * DIM * 4
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with src.open("rb") as f, tmp.open("wb") as out:
        n, d = struct.unpack("II", f.read(8))
        if d != DIM or n < NPTS:
            raise SystemExit(f"unexpected header n={n} d={d}")
        out.write(struct.pack("II", NPTS, DIM))
        # GPFS does not support copy_file_range: plain chunked read/write.
        copied = 0
        chunk = 1 << 28
        buf = bytearray(chunk)
        view = memoryview(buf)
        while copied < payload:
            got = f.readinto(view[:min(chunk, payload - copied)])
            if not got:
                raise SystemExit("short read from source")
            out.write(view[:got])
            copied += got
            if copied % (32 << 30) < chunk:
                print(f"{copied / payload * 100:.1f} %", flush=True)
    if tmp.stat().st_size != 8 + payload:
        raise SystemExit(f"bad output size {tmp.stat().st_size}")
    tmp.replace(dst)
    print(f"Wrote {dst}  n={NPTS} d={DIM}")


def make_gt(src: Path, dst: Path) -> None:
    rows = []
    with src.open() as f:
        for line in f:
            rows.append([i for i in map(int, line.split()) if i < NPTS])
    k = min(len(r) for r in rows)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("wb") as out:
        out.write(struct.pack("II", len(rows), k))
        for r in rows:
            out.write(struct.pack(f"<{k}I", *r[:k]))
    tmp.replace(dst)
    print(f"Wrote {dst}  n={len(rows)} k={k}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["base", "gt"])
    args = ap.parse_args()
    if args.what == "base":
        make_base(DIR / "base.vectors.bin", DIR / "base.100M.fbin")
    else:
        make_gt(DIR / "truth.txt",
                DIR / "computedGT_msmarco100M_9376queries_k93_cotra.bin")


if __name__ == "__main__":
    main()
