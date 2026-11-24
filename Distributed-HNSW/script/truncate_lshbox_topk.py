#!/usr/bin/env python3
"""Truncate Pyramid lshbox GT from top-100 to top-10 (same layout as 10M gt_k10).

Usage:
  python3 truncate_lshbox_topk.py IN.lshbox OUT.lshbox [--k 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def truncate_lshbox(src: Path, dst: Path, k_out: int = 10) -> None:
    with src.open() as fin:
        header = fin.readline().strip().split()
        if len(header) != 2:
            raise SystemExit(f"bad header in {src}: {header}")
        nq, k_in = int(header[0]), int(header[1])
        if k_in < k_out:
            raise SystemExit(f"{src}: k_in={k_in} < k_out={k_out}")
        rows = []
        for i, line in enumerate(fin):
            parts = line.strip().split()
            if not parts:
                continue
            qid = parts[0]
            pairs = parts[1:]
            if len(pairs) < 2 * k_out:
                raise SystemExit(
                    f"{src} line {i + 2}: need {2 * k_out} fields after qid, got {len(pairs)}"
                )
            rows.append((qid, pairs[: 2 * k_out]))
        if len(rows) != nq:
            raise SystemExit(f"{src}: expected {nq} rows, got {len(rows)}")
    with dst.open("w") as fout:
        fout.write(f"{nq}\t{k_out}\n")
        for qid, pairs in rows:
            fout.write(qid + "\t" + "\t".join(pairs) + "\n")
    print(f"wrote {dst} nq={nq} k={k_out} (from k_in={k_in})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    truncate_lshbox(args.src, args.dst, args.k)


if __name__ == "__main__":
    main()
