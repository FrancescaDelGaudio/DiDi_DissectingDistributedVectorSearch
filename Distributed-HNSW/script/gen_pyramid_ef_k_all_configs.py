#!/usr/bin/env python3
"""Generate unified Pyramid query-sweep config: 4n/6n (10M) + 8n (100M).

252 tasks = 3 layouts × deep|tti × @10|@100 × ef{50,100,200} × K{1,10,20,50,70,80,100}
"""

from __future__ import annotations

import argparse
from pathlib import Path

BUILD = Path("/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/build")
DEEP_10M = Path("/gpfs/scratch/delgaudio/vectors/DEEP/generated_deep_files/pyramid")
TTI_10M = Path("/gpfs/scratch/delgaudio/vectors/TTI")
DEEP_100M = Path("/gpfs/scratch/delgaudio/vectors/DEEP/pyramid_100M")
TTI_100M = Path("/gpfs/scratch/delgaudio/vectors/TTI/pyramid_100M")

SENDER_EF_VALUES = (50, 100, 200)
WAKEUP_K_VALUES = (1, 10, 20, 50, 70, 80, 100)
RECALL_K_VALUES = (10, 100)
DATASETS = ("deep", "tti")

HEADER = (
    "task_id\tjob_name\tdataset\tnum_subhnsw\twakeup_rate\tsender_ef\t"
    "recall_k\tdim\tdata\tout\tgt\tdhnsw\tnodes\tntasks"
)


def paths_for(dataset: str, num_subhnsw: int, recall_k: int, scale: str):
    if scale == "10m":
        if dataset == "deep":
            dim, data = 96, DEEP_10M
            out = data / f"index_{num_subhnsw}shard_m10k"
            dhnsw = BUILD / f"bug_{num_subhnsw}_k{recall_k}"
        else:
            dim, data = 200, TTI_10M
            out = data / f"index_{num_subhnsw}shard_m10k_mips"
            dhnsw = BUILD / f"bug_mips_{num_subhnsw}_k{recall_k}"
    else:
        if dataset == "deep":
            dim, data = 96, DEEP_100M
            out = data / "index_8shard_m10k"
            dhnsw = BUILD / f"bug_8_k{recall_k}"
        else:
            dim, data = 200, TTI_100M
            out = data / "index_8shard_m10k_mips"
            dhnsw = BUILD / f"bug_mips_8_k{recall_k}"
    gt = data / ("gt_k10.lshbox" if recall_k == 10 else "gt.lshbox")
    return dim, data, out, gt, dhnsw


def rows() -> list[str]:
    lines = [HEADER]
    tid = 0
    # (num_subhnsw, scale, nodes, ntasks)
    layouts = (
        (4, "10m", 4, 6),
        (6, "10m", 6, 8),
        (8, "100m", 8, 10),
    )
    for num_subhnsw, scale, nodes, ntasks in layouts:
        for dataset in DATASETS:
            for recall_k in RECALL_K_VALUES:
                dim, data, out, gt, dhnsw = paths_for(
                    dataset, num_subhnsw, recall_k, scale
                )
                for sender_ef in SENDER_EF_VALUES:
                    for wakeup_k in WAKEUP_K_VALUES:
                        tag = "10m" if scale == "10m" else "100m"
                        job_name = (
                            f"dhnsw-{num_subhnsw}n-{tag}-{dataset}-"
                            f"K{wakeup_k}-ef{sender_ef}-k{recall_k}"
                        )
                        lines.append(
                            "\t".join(
                                map(
                                    str,
                                    [
                                        tid,
                                        job_name,
                                        dataset,
                                        num_subhnsw,
                                        wakeup_k,
                                        sender_ef,
                                        recall_k,
                                        dim,
                                        data,
                                        out,
                                        gt,
                                        dhnsw,
                                        nodes,
                                        ntasks,
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
    out = args.out_dir / "pyramid_ef_k_all_layouts.tsv"
    text = "\n".join(rows()) + "\n"
    out.write_text(text)
    print(f"wrote {out} ({text.count(chr(10)) - 1} tasks)")


if __name__ == "__main__":
    main()
