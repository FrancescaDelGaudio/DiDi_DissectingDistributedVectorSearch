#!/usr/bin/env python3
"""Generate TSV config for Pyramid 100M query sweep on 8 physical nodes.

Grid (84 tasks):
  dataset in {deep, tti}
  recall_k in {10, 100}
  sender_ef in {50, 100, 200}
  wakeup_k (K) in {1, 10, 20, 50, 70, 80, 100}
"""

from __future__ import annotations

import argparse
from pathlib import Path

BUILD = Path("/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/build")
DEEP_DATA = Path("/gpfs/scratch/delgaudio/vectors/DEEP/pyramid_100M")
TTI_DATA = Path("/gpfs/scratch/delgaudio/vectors/TTI/pyramid_100M")

SENDER_EF_VALUES = (50, 100, 200)
WAKEUP_K_VALUES = (1, 10, 20, 50, 70, 80, 100)
RECALL_K_VALUES = (10, 100)
DATASETS = ("deep", "tti")
NUM_SUBHNSW = 8

HEADER = (
    "task_id\tjob_name\tdataset\tnum_subhnsw\twakeup_rate\tsender_ef\t"
    "recall_k\tdim\tdata\tout\tgt\tdhnsw"
)


def paths_for(dataset: str, recall_k: int) -> tuple[int, Path, Path, Path, Path]:
    if dataset == "deep":
        dim = 96
        data = DEEP_DATA
        out = data / "index_8shard_m10k"
        dhnsw = BUILD / f"bug_8_k{recall_k}"
    elif dataset == "tti":
        dim = 200
        data = TTI_DATA
        out = data / "index_8shard_m10k_mips"
        dhnsw = BUILD / f"bug_mips_8_k{recall_k}"
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    gt = data / ("gt_k10.lshbox" if recall_k == 10 else "gt.lshbox")
    return dim, data, out, gt, dhnsw


def rows() -> list[str]:
    lines = [HEADER]
    tid = 0
    for dataset in DATASETS:
        for recall_k in RECALL_K_VALUES:
            dim, data, out, gt, dhnsw = paths_for(dataset, recall_k)
            for sender_ef in SENDER_EF_VALUES:
                for wakeup_k in WAKEUP_K_VALUES:
                    job_name = (
                        f"dhnsw-8n-100m-{dataset}-K{wakeup_k}-ef{sender_ef}-k{recall_k}"
                    )
                    lines.append(
                        "\t".join(
                            map(
                                str,
                                [
                                    tid,
                                    job_name,
                                    dataset,
                                    NUM_SUBHNSW,
                                    wakeup_k,
                                    sender_ef,
                                    recall_k,
                                    dim,
                                    data,
                                    out,
                                    gt,
                                    dhnsw,
                                ],
                            )
                        )
                    )
                    tid += 1
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "configs",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out = args.out_dir / "pyramid_ef_k_8node_100m.tsv"
    text = "\n".join(rows()) + "\n"
    out.write_text(text)
    n_tasks = text.count("\n") - 1
    print(f"wrote {out} ({n_tasks} tasks)")


if __name__ == "__main__":
    main()
