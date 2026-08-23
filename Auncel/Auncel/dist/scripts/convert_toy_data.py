#!/usr/bin/env python3
"""Convert toy fbin / GT.bin to Faiss fvecs/ivecs for Auncel dist smoke tests."""

import argparse
import struct
from pathlib import Path

DEFAULT_TOY = Path("/gpfs/scratch/delgaudio/vectors/toy/generated_deep_files")


def fbin_to_fvecs(src: Path, dst: Path) -> None:
    data = src.read_bytes()
    n, d = struct.unpack_from("<ii", data, 0)
    flat = struct.unpack_from(f"<{n * d}f", data, 8)
    with dst.open("wb") as out:
        for i in range(n):
            out.write(struct.pack("<i", d))
            out.write(struct.pack(f"<{d}f", *flat[i * d : (i + 1) * d]))
    print(f"fbin -> fvecs: {src.name}  n={n} d={d}  ->  {dst}")


def gt_bin_to_faiss(src: Path, dst_i: Path, dst_d: Path) -> None:
    data = src.read_bytes()
    nq, k = struct.unpack_from("<ii", data, 0)
    off = 8
    ids = struct.unpack_from(f"<{nq * k}i", data, off)
    off += 4 * nq * k
    dis = struct.unpack_from(f"<{nq * k}f", data, off)
    with dst_i.open("wb") as fi, dst_d.open("wb") as fd:
        for q in range(nq):
            fi.write(struct.pack("<i", k))
            fi.write(struct.pack(f"<{k}i", *ids[q * k : (q + 1) * k]))
            fd.write(struct.pack("<i", k))
            fd.write(struct.pack(f"<{k}f", *dis[q * k : (q + 1) * k]))
    print(f"GT bin -> ivecs/fvecs: nq={nq} k={k}  ->  {dst_i.name}, {dst_d.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_TOY)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TOY / "pyramid",
    )
    p.add_argument("--base-fbin", default="random_base_100.fbin")
    p.add_argument("--query-fbin", default="random_queries_10_k5.fbin")
    p.add_argument("--gt-bin", default="random_GT_10_k5.bin")
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    indir = args.input_dir

    fbin_to_fvecs(indir / args.base_fbin, out / "base.fvecs")
    fbin_to_fvecs(indir / args.query_fbin, out / "query.fvecs")
    gt_bin_to_faiss(indir / args.gt_bin, out / "idx.ivecs", out / "dis.fvecs")


if __name__ == "__main__":
    main()
