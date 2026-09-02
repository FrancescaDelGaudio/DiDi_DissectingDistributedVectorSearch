#!/usr/bin/env python3
"""Regenerate configs/distbench_ef_nprobe_k{4,6}.tsv for the array sweeps."""

from pathlib import Path

GP_ANN_ROOT = Path("/gpfs/users/delgaudio/exp_eval/gp-ann")
PARTITION_DIR = Path(
    "/gpfs/scratch/delgaudio/vectors/DEEP/new_experiments_gpann_fixed"
)
OVERLAP_VALUES = (0.1, 0.2, 0.3)
EF_SEARCH_VALUES = (100, 150, 250, 400)
RECALL_K_VALUES = (10, 100)
HEADER = "\t".join(
    [
        "task_id",
        "job_name",
        "nodes",
        "overlap",
        "nprobe",
        "ef_search",
        "recall_k",
        "partition_file",
        "router_index",
        "bench",
    ]
)


def write_config(ogp_k: int) -> int:
    out = GP_ANN_ROOT / "configs" / "distbench_ef_nprobe_k{}.tsv".format(ogp_k)
    rows = []
    tid = 0
    for overlap in OVERLAP_VALUES:
        o = "{:g}".format(overlap)
        part = PARTITION_DIR / "base.10M.partition.k={}.GP.o={}".format(ogp_k, o)
        router = Path(str(part) + ".routing_index")
        o_tag = str(int(round(overlap * 10)))
        for nprobe in range(1, ogp_k + 1):
            bench = (
                GP_ANN_ROOT
                / "release_l2_distbench_np{}".format(nprobe)
                / "DistributedBench"
            )
            for ef in EF_SEARCH_VALUES:
                for rk in RECALL_K_VALUES:
                    name = "distfix_k{}_o{}_np{}_ef{}_k{}".format(
                        ogp_k, o_tag, nprobe, ef, rk
                    )
                    rows.append(
                        "\t".join(
                            [
                                str(tid),
                                name,
                                str(ogp_k),
                                o,
                                str(nprobe),
                                str(ef),
                                str(rk),
                                str(part),
                                str(router),
                                str(bench),
                            ]
                        )
                    )
                    tid += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    print("Wrote {} ({} tasks, array=0-{})".format(out, tid, tid - 1))
    return tid


def main():
    for k in (4, 6):
        write_config(k)


if __name__ == "__main__":
    main()
