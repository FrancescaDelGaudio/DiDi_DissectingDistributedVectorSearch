#!/usr/bin/env python3
"""
Submit DistributedBench on TTI new_experiments_gpann partitions (MIPS / inner product).

Grid (30 jobs by default) — same structure as DEEP:
  - OGP K=4 (5 MPI shards): overlap 0.1/0.2/0.3 x shardsProbe 3,4 x recall k 10,100  -> 12
  - OGP K=6, o=0.1/0.2 (7 shards): shardsProbe 3,4,6 x k 10,100  -> 12
  - OGP K=6, o=0.3 (8 shards): shardsProbe 3,4,6 x k 10,100  -> 6

Binaries: release_mips_distbench_np{3,4,6}/DistributedBench

Usage:
  python3 submit_new_experiments_distbench_tti.py --dry-run
  python3 submit_new_experiments_distbench_tti.py
  python3 submit_new_experiments_distbench_tti.py --ogp-k 4 --overlap 0.1 --shards-probe 3 --recall-k 10
"""

import argparse
import subprocess
import sys
from pathlib import Path

GP_ANN_ROOT = Path("/gpfs/users/delgaudio/exp_eval/gp-ann")
SBATCH_SCRIPT = GP_ANN_ROOT / "sbatch_distributed_tti_fbin.sh"
OUTPUT_ROOT = Path("/gpfs/scratch/delgaudio/vectors/TTI/new_experiments_distributedbench")
LOG_DIR = OUTPUT_ROOT / "logs"
PARTITION_DIR = Path("/gpfs/scratch/delgaudio/vectors/TTI/new_experiments_gpann")
PARTITION_STEM = "yangt2i10M.partition"
BASE_FBIN = Path("/gpfs/scratch/delgaudio/vectors/TTI/yangt2i10M.fbin")
QUERY_FBIN = Path("/gpfs/scratch/delgaudio/vectors/TTI/yangt2iquery10K.fbin")
GT_BIN = Path("/gpfs/scratch/delgaudio/vectors/TTI/computedGT_tti10M_10kqueries_k100_faiss.bin")
RECALL_K_VALUES = (10, 100)

PROBE_GRID = [
    (4, 0.1, (3, 4)),
    (4, 0.2, (3, 4)),
    (4, 0.3, (3, 4)),
    (6, 0.1, (3, 4, 6)),
    (6, 0.2, (3, 4, 6)),
    (6, 0.3, (3, 4, 6)),
]


def partition_path(ogp_k, overlap):
    o = "{:g}".format(overlap)
    return PARTITION_DIR / "{}.k={}.GP.o={}".format(PARTITION_STEM, ogp_k, o)


def bench_binary(shards_probe):
    return GP_ANN_ROOT / "release_mips_distbench_np{}".format(shards_probe) / "DistributedBench"


def count_shards(part_file):
    with open(part_file) as f:
        return sum(1 for _ in f)


def slurm_job_name(num_shards, overlap, shards_probe, recall_k):
    o_tag = str(int(round(overlap * 10)))
    return "dist_tti_s{}_o{}_np{}_k{}".format(num_shards, o_tag, shards_probe, recall_k)


def iter_experiments(ogp_k_filter, overlap_filter, probe_filter, recall_k_filter):
    for ogp_k, overlap, probes in PROBE_GRID:
        if ogp_k_filter is not None and ogp_k not in ogp_k_filter:
            continue
        if overlap_filter is not None and overlap not in overlap_filter:
            continue
        part = partition_path(ogp_k, overlap)
        if not part.is_file():
            yield {"skip": True, "reason": "missing partition", "part": part}
            continue
        num_shards = count_shards(part)
        for shards_probe in probes:
            if probe_filter is not None and shards_probe not in probe_filter:
                continue
            if shards_probe > num_shards:
                continue
            for recall_k in RECALL_K_VALUES:
                if recall_k_filter is not None and recall_k not in recall_k_filter:
                    continue
                yield {
                    "skip": False,
                    "ogp_k": ogp_k,
                    "overlap": overlap,
                    "num_shards": num_shards,
                    "shards_probe": shards_probe,
                    "recall_k": recall_k,
                    "part": part,
                    "router": Path(str(part) + ".routing_index"),
                    "bench": bench_binary(shards_probe),
                }


def submit_one(exp, dry_run):
    job_name = slurm_job_name(
        exp["num_shards"], exp["overlap"], exp["shards_probe"], exp["recall_k"]
    )
    n = exp["num_shards"]
    log_base = LOG_DIR / "{}_%j".format(job_name)

    export = {
        "GP_ANN_ROOT": str(GP_ANN_ROOT),
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "PARTITION_FILE": str(exp["part"]),
        "ROUTER_INDEX": str(exp["router"]),
        "RECALL_K": str(exp["recall_k"]),
        "SHARDS_PROBE": str(exp["shards_probe"]),
        "BENCH": str(exp["bench"]),
        "BASE_FBIN": str(BASE_FBIN),
        "QUERY_FBIN": str(QUERY_FBIN),
        "GT_BIN": str(GT_BIN),
    }
    export_pairs = ",".join("{}={}".format(k, v) for k, v in export.items())

    cmd = [
        "sbatch",
        "--job-name={}".format(job_name),
        "--nodes={}".format(n),
        "-n{}".format(n),
        "-o{}.log".format(log_base),
        "-e{}.log".format(log_base),
        "--export=ALL,{}".format(export_pairs),
        str(SBATCH_SCRIPT),
    ]

    print("{} {}".format("[dry-run]" if dry_run else "", " ".join(cmd)))
    print("  partition: {} ({} shards, OGP k={} o={})".format(
        exp["part"], exp["num_shards"], exp["ogp_k"], exp["overlap"]))
    print("  router:    {}".format(exp["router"]))
    print("  bench:     {} (MIPS, shardsProbe={})".format(exp["bench"], exp["shards_probe"]))
    if dry_run:
        return None

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        return result.returncode
    job_id = result.stdout.strip().split()[-1]
    print("  submitted job {}".format(job_id))
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ogp-k", type=int, action="append", dest="ogp_k_values")
    p.add_argument("--overlap", type=float, action="append", dest="overlaps")
    p.add_argument("--shards-probe", type=int, action="append", dest="shards_probes")
    p.add_argument("--recall-k", type=int, action="append", dest="recall_k_values")
    return p.parse_args()


def main():
    args = parse_args()
    if not SBATCH_SCRIPT.is_file():
        print("FATAL: missing {}".format(SBATCH_SCRIPT), file=sys.stderr)
        return 1

    for path, label in (
        (BASE_FBIN, "base"),
        (QUERY_FBIN, "query"),
        (GT_BIN, "gt"),
    ):
        if not path.is_file() and not args.dry_run:
            print("FATAL: missing {}: {}".format(label, path), file=sys.stderr)
            return 1

    if not args.dry_run:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    n_total = n_sub = n_skip = n_err = 0
    for exp in iter_experiments(
        tuple(args.ogp_k_values) if args.ogp_k_values else None,
        tuple(args.overlaps) if args.overlaps else None,
        tuple(args.shards_probes) if args.shards_probes else None,
        tuple(args.recall_k_values) if args.recall_k_values else None,
    ):
        n_total += 1
        if exp.get("skip"):
            n_skip += 1
            print("SKIP {}: {}".format(exp["part"], exp["reason"]))
            continue
        if not exp["bench"].is_file():
            n_err += 1
            print("FATAL: missing binary {}".format(exp["bench"]), file=sys.stderr)
            continue
        if not exp["router"].is_file():
            n_skip += 1
            print("SKIP missing router: {}".format(exp["router"]))
            continue
        rc = submit_one(exp, args.dry_run)
        if rc:
            n_err += 1
        else:
            n_sub += 1

    print(
        "\nDone: {} planned, submitted={}, skipped={}, errors={}.".format(
            n_total, n_sub, n_skip, n_err
        )
    )
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
