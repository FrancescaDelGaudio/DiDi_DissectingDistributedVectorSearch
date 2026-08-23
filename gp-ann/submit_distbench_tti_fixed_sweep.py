#!/usr/bin/env python3
"""
Submit DistributedBench (MIPS) on TTI fixed-machines-mode partitions.

Defaults by --ogp-k:
  k=4: shardsProbe in {2, 3, 4}     -> 6 jobs per overlap
  k=6: shardsProbe in {2, 3, 4, 5}  -> 8 jobs per overlap

Binaries: release_mips_distbench_np{N}/DistributedBench

Usage:
  python3 submit_distbench_tti_fixed_sweep.py --ogp-k 6 --overlap 0.1
  python3 submit_distbench_tti_fixed_sweep.py --ogp-k 6 --overlap 0.2 --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

GP_ANN_ROOT = Path("/gpfs/users/delgaudio/exp_eval/gp-ann")
SBATCH_SCRIPT = GP_ANN_ROOT / "sbatch_distributed_tti_fbin.sh"
PARTITION_DIR = Path(
    "/gpfs/scratch/delgaudio/vectors/TTI/new_experiments_gpann_fixed"
)
OUTPUT_ROOT = PARTITION_DIR / "distbench"
LOG_DIR = OUTPUT_ROOT / "logs"
RECALL_K_VALUES = (10, 100)

DEFAULT_PROBES = {
    4: (2, 3, 4),
    6: (2, 3, 4, 5),
}


def partition_path(ogp_k, overlap):
    o = "{:g}".format(overlap)
    return PARTITION_DIR / "yangt2i10M.partition.k={}.GP.o={}".format(ogp_k, o)


def bench_binary(shards_probe):
    return (
        GP_ANN_ROOT
        / "release_mips_distbench_np{}".format(shards_probe)
        / "DistributedBench"
    )


def slurm_job_name(ogp_k, overlap, shards_probe, recall_k):
    o_tag = str(int(round(overlap * 10)))
    return "disttti_k{}_o{}_np{}_k{}".format(ogp_k, o_tag, shards_probe, recall_k)


def iter_experiments(ogp_k, overlap, probe_values, probe_filter, recall_k_filter):
    part = partition_path(ogp_k, overlap)
    if not part.is_file():
        yield {"skip": True, "reason": "missing partition", "part": part}
        return

    num_shards = sum(1 for _ in open(part))
    router = Path(str(part) + ".routing_index")

    for shards_probe in probe_values:
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
                "router": router,
                "bench": bench_binary(shards_probe),
            }


def submit_one(exp, dry_run):
    job_name = slurm_job_name(
        exp["ogp_k"], exp["overlap"], exp["shards_probe"], exp["recall_k"]
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
    print("  bench:     {} (shardsProbe={}) recall@K={}".format(
        exp["bench"], exp["shards_probe"], exp["recall_k"]))
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
    p.add_argument("--ogp-k", type=int, default=6, choices=sorted(DEFAULT_PROBES))
    p.add_argument("--overlap", type=float, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--shards-probe", type=int, action="append", dest="shards_probes")
    p.add_argument("--recall-k", type=int, action="append", dest="recall_k_values")
    return p.parse_args()


def main():
    args = parse_args()
    if not SBATCH_SCRIPT.is_file():
        print("FATAL: missing {}".format(SBATCH_SCRIPT), file=sys.stderr)
        return 1

    if not args.dry_run:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    probe_values = DEFAULT_PROBES[args.ogp_k]
    n_total = n_sub = n_skip = n_err = 0
    for exp in iter_experiments(
        args.ogp_k,
        args.overlap,
        probe_values,
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
        "\nDone (TTI k={} o={}): {} planned, submitted={}, skipped={}, errors={}.".format(
            args.ogp_k, args.overlap, n_total, n_sub, n_skip, n_err
        )
    )
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
