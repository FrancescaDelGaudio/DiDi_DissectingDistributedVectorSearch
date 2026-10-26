#!/usr/bin/env python3
"""Generate configs/distbench_tti100M_ef_nprobe_k8.tsv for TTI 100M DistBench (MIPS).

Same grid as DEEP 100M:
  overlap in {0.1, 0.2, 0.3}
  nprobe in 1..8
  ef_search in {50, 150, 250, 400}
  recall_k in {10, 100}
  => 3 * 8 * 4 * 2 = 192 tasks

Partition mapping (matches QA campaign):
  o=0.1 -> fixed-machines partition with eps=0.05
  o=0.2 / o=0.3 -> fixed-machines partitions with eps=0.35
"""

from pathlib import Path

GP_ANN_ROOT = Path("/gpfs/users/delgaudio/exp_eval/gp-ann")
TTI_ROOT = Path("/gpfs/scratch/delgaudio/vectors/TTI")
OGP_K = 8
OVERLAP_VALUES = (0.1, 0.2, 0.3)
EF_SEARCH_VALUES = (50, 150, 250, 400)
RECALL_K_VALUES = (10, 100)

PARTITIONS = {
    0.1: (
        TTI_ROOT / "base.100M.partition.k=8.GP.o=0.1",
        TTI_ROOT / "base.100M.partition.k=8.GP.o=0.1.routing_index",
    ),
    0.2: (
        TTI_ROOT / "base.100M.partition.k=8.GP.o=0.2",
        TTI_ROOT / "base.100M.partition.k=8.GP.o=0.2.routing_index",
    ),
    0.3: (
        TTI_ROOT / "base.100M.partition.k=8.GP.o=0.3",
        TTI_ROOT / "base.100M.partition.k=8.GP.o=0.3.routing_index",
    ),
}

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


def main() -> int:
    out = GP_ANN_ROOT / "configs" / "distbench_tti100M_ef_nprobe_k8.tsv"
    rows = []
    tid = 0
    missing = []

    for overlap in OVERLAP_VALUES:
        part, router = PARTITIONS[overlap]
        if not part.is_file():
            missing.append(str(part))
        if not router.is_file():
            missing.append(str(router))
        sidecar = Path(str(router) + ".routing_index_partition")
        if not sidecar.is_file():
            missing.append(str(sidecar))

        o = "{:g}".format(overlap)
        o_tag = str(int(round(overlap * 10)))
        for nprobe in range(1, OGP_K + 1):
            bench = (
                GP_ANN_ROOT
                / "release_mips_distbench_np{}".format(nprobe)
                / "DistributedBench"
            )
            if not bench.is_file():
                missing.append(str(bench))
            for ef in EF_SEARCH_VALUES:
                for rk in RECALL_K_VALUES:
                    name = "disttti100_k{}_o{}_np{}_ef{}_k{}".format(
                        OGP_K, o_tag, nprobe, ef, rk
                    )
                    rows.append(
                        "\t".join(
                            [
                                str(tid),
                                name,
                                str(OGP_K),
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

    if missing:
        uniq = sorted(set(missing))
        raise SystemExit(
            "Missing required files ({}):\n  {}".format(len(uniq), "\n  ".join(uniq))
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    print("Wrote {} ({} tasks, array=0-{})".format(out, tid, tid - 1))
    return tid


if __name__ == "__main__":
    main()
