#!/usr/bin/env python3
"""
Prepare MSMARCO Web Search (768-d float, MIPS, 9376 queries) for all five systems,
on the 10M and 100M prefixes of base.vectors.bin (base.100M.fbin already exists).

  base10m       base.100M.fbin -> base.10M.fbin (first 10M vectors)
  fvecs N       base.<N>M.fbin -> base.<N>M.fvecs, and query.vectors.bin -> query.fvecs
                (DA, Pyramid, Harmony read fvecs)
  gt N          exact MIPS top-10 over base.<N>M.fbin, written in every format:
                  computedGT_msmarco<N>M_9376q_k10.bin        int32 nq,k | int32 ids | f32 ips  (GP, DA)
                  computedGT_msmarco<N>M_9376q_k10_cotra.bin  uint32 nq,k | uint32 ids            (CoTra)
                  pyramid_<N>M/gt_k10.lshbox                  text "nq k", "qid id ip ..."        (Pyramid)
                  ../MSMARCO<N>M/result/groundtruth_10.bin    int64 ids | f32 ips                 (Harmony)
                Distances are inner products, sorted in decreasing order (same as TTI).
  layout N      Pyramid and Harmony directories (symlinks to the fvecs files)

Usage: python3 -I prepare_msmarco.py {base10m | fvecs N | gt N | layout N} [--procs P]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import struct
from pathlib import Path

import numpy as np

DIR = Path("/gpfs/scratch/delgaudio/vectors/MSMARCOWebSearch")
VEC = DIR.parent
DIM = 768
K = 10
QUERY = DIR / "query.vectors.bin"
CHUNK = 1 << 28  # bytes per read/write


def fbin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        return struct.unpack("II", f.read(8))


def make_base10m() -> None:
    src, dst = DIR / "base.100M.fbin", DIR / "base.10M.fbin"
    n_out = 10_000_000
    n, d = fbin_header(src)
    assert d == DIM and n >= n_out, (n, d)
    payload = n_out * DIM * 4
    tmp = dst.with_suffix(".fbin.tmp")
    with src.open("rb") as f, tmp.open("wb") as out:
        f.seek(8)
        out.write(struct.pack("II", n_out, DIM))
        buf = bytearray(CHUNK)
        view = memoryview(buf)
        copied = 0
        while copied < payload:
            got = f.readinto(view[:min(CHUNK, payload - copied)])
            if not got:
                raise SystemExit("short read")
            out.write(view[:got])
            copied += got
    assert tmp.stat().st_size == 8 + payload
    tmp.rename(dst)
    print(f"wrote {dst}", flush=True)


def fbin_to_fvecs(src: Path, dst: Path) -> None:
    n, d = fbin_header(src)
    rows = max(1, CHUNK // (d * 4))
    tmp = dst.with_suffix(".fvecs.tmp")
    data = np.memmap(src, dtype=np.float32, mode="r", offset=8, shape=(n, d))
    with tmp.open("wb") as out:
        for start in range(0, n, rows):
            block = np.asarray(data[start:start + rows])
            rec = np.empty((block.shape[0], d + 1), dtype=np.float32)
            rec[:, 0] = np.array([d], dtype=np.int32).view(np.float32)[0]
            rec[:, 1:] = block
            out.write(rec.tobytes())
            if (start // rows) % 40 == 0:
                print(f"{dst.name}: {start / n * 100:.1f} %", flush=True)
    assert tmp.stat().st_size == n * (d + 1) * 4
    tmp.rename(dst)
    print(f"wrote {dst} n={n} d={d}", flush=True)


def _topk_range(args):
    base_path, n, lo, hi = args
    q = np.fromfile(QUERY, dtype=np.float32, offset=8).reshape(-1, DIM)
    base = np.memmap(base_path, dtype=np.float32, mode="r", offset=8, shape=(n, DIM))
    best_s = np.full((q.shape[0], K), -np.inf, dtype=np.float32)
    best_i = np.full((q.shape[0], K), -1, dtype=np.int64)
    step = 8_000
    for s in range(lo, hi, step):
        e = min(hi, s + step)
        scores = q @ np.asarray(base[s:e]).T                     # (nq, e-s) inner products
        part = np.argpartition(-scores, K - 1, axis=1)[:, :K]
        cand_s = np.concatenate([best_s, np.take_along_axis(scores, part, axis=1)], axis=1)
        cand_i = np.concatenate([best_i, part + s], axis=1)
        keep = np.argpartition(-cand_s, K - 1, axis=1)[:, :K]
        best_s = np.take_along_axis(cand_s, keep, axis=1)
        best_i = np.take_along_axis(cand_i, keep, axis=1)
    return best_s, best_i


def compute_gt(million: int, procs: int) -> None:
    base_path = DIR / f"base.{million}M.fbin"
    n, d = fbin_header(base_path)
    assert d == DIM and n == million * 1_000_000, (n, d)
    nq, dq = fbin_header(QUERY)
    assert dq == DIM
    bounds = np.linspace(0, n, procs * 4 + 1, dtype=np.int64)
    tasks = [(base_path, n, int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
    best_s = np.full((nq, K), -np.inf, dtype=np.float32)
    best_i = np.full((nq, K), -1, dtype=np.int64)
    with mp.get_context("fork").Pool(procs) as pool:
        for done, (s, i) in enumerate(pool.imap_unordered(_topk_range, tasks), 1):
            cs, ci = np.concatenate([best_s, s], 1), np.concatenate([best_i, i], 1)
            keep = np.argpartition(-cs, K - 1, axis=1)[:, :K]
            best_s, best_i = np.take_along_axis(cs, keep, 1), np.take_along_axis(ci, keep, 1)
            if done % procs == 0:
                print(f"gt {million}M: {done}/{len(tasks)} ranges", flush=True)
    order = np.argsort(-best_s, axis=1, kind="stable")
    ips = np.take_along_axis(best_s, order, 1).astype(np.float32)
    ids = np.take_along_axis(best_i, order, 1)
    assert ids.min() >= 0 and ids.max() < n

    tag = f"msmarco{million}M_9376q_k10"
    with (DIR / f"computedGT_{tag}.bin").open("wb") as f:
        f.write(struct.pack("ii", nq, K)); f.write(ids.astype(np.int32).tobytes()); f.write(ips.tobytes())
    with (DIR / f"computedGT_{tag}_cotra.bin").open("wb") as f:
        f.write(struct.pack("II", nq, K)); f.write(ids.astype(np.uint32).tobytes())
    pyr = DIR / f"pyramid_{million}M"
    pyr.mkdir(exist_ok=True)
    with (pyr / "gt_k10.lshbox").open("w") as f:
        f.write(f"{nq}\t{K}\n")
        for qi in range(nq):
            f.write(str(qi) + "\t" + "\t".join(f"{ids[qi, j]}\t{float(ips[qi, j])!r}" for j in range(K)) + "\n")
    hres = VEC / f"MSMARCO{million}M" / "result"
    hres.mkdir(parents=True, exist_ok=True)
    with (hres / "groundtruth_10.bin").open("wb") as f:
        f.write(ids.astype(np.int64).tobytes()); f.write(ips.tobytes())
    print(f"gt {million}M written; query 0 top ids {ids[0, :5]} ips {ips[0, :3]}", flush=True)


def layout(million: int) -> None:
    base = DIR / f"base.{million}M.fvecs"
    pyr = DIR / f"pyramid_{million}M"
    pyr.mkdir(exist_ok=True)
    for link, target in [(pyr / "base.fvecs", base), (pyr / "query.fvecs", DIR / "query.fvecs")]:
        if not link.exists():
            link.symlink_to(target)
    h = VEC / f"MSMARCO{million}M"
    for sub in ("origin", "result", "index"):
        (h / sub).mkdir(parents=True, exist_ok=True)
    for link, target in [(h / "origin" / f"MSMARCO{million}M_base.fvecs", base),
                         (h / "origin" / f"MSMARCO{million}M_query.fvecs", DIR / "query.fvecs")]:
        if not link.exists():
            link.symlink_to(target)
    print(f"layout {million}M ready: {pyr} {h}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["base10m", "fvecs", "gt", "layout"])
    ap.add_argument("million", nargs="?", type=int, choices=[10, 100])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.step == "base10m":
        make_base10m()
    elif a.step == "fvecs":
        fbin_to_fvecs(DIR / f"base.{a.million}M.fbin", DIR / f"base.{a.million}M.fvecs")
        if not (DIR / "query.fvecs").exists():
            fbin_to_fvecs(QUERY, DIR / "query.fvecs")
    elif a.step == "gt":
        compute_gt(a.million, a.procs)
    else:
        layout(a.million)


if __name__ == "__main__":
    main()
