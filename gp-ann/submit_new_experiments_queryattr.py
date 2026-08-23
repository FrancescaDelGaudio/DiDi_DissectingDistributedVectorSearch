#!/usr/bin/env python3
"""
Submit QueryAttribution for each partition in new_experiments_gpann.

  - L2 / DEEP: release_l2/QueryAttribution
  - MIPS / TTI: release_mips/QueryAttribution

One job per partition file (6 DEEP + 6 TTI = 12 jobs).
REQUESTED_SHARDS is set in sbatch via wc -l on PARTITION_FILE.

Outputs under:
  .../DEEP/new_experiments_queryAttribution/base.10M.k{K}_o{O}.qa
  .../TTI/new_experiments_queryAttribution/yangt2i10M.k{K}_o{O}.qa

Router index is written next to the partition file:
  .../new_experiments_gpann/*.partition.k=K.GP.o=O.routing_index

Usage:
  python3 submit_new_experiments_queryattr.py --dry-run
  python3 submit_new_experiments_queryattr.py
  python3 submit_new_experiments_queryattr.py --metric deep
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional

GP_ANN_ROOT = Path("/gpfs/users/delgaudio/exp_eval/gp-ann")
SBATCH_SCRIPT = GP_ANN_ROOT / "sbatch_queryattr_fbin.sh"
LOG_DIR = Path("/gpfs/scratch/delgaudio/gp_ann/new_experiments_queryAttribution/logs")

K_VALUES = (4, 6)
OVERLAP_VALUES = (0.1, 0.2, 0.3)
GT_K = 100


class DatasetConfig(NamedTuple):
    name: str
    job_tag: str
    queryattr_bin: Path
    base_fbin: Path
    query_fbin: Path
    gt_bin: Path
    partition_dir: Path
    partition_stem: str
    qa_dir: Path
    qa_stem: str


DATASETS = {
    "l2": DatasetConfig(
        name="l2",
        job_tag="deep",
        queryattr_bin=GP_ANN_ROOT / "release_l2" / "QueryAttribution",
        base_fbin=Path("/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.fbin"),
        query_fbin=Path("/gpfs/scratch/delgaudio/vectors/DEEP/query.public.10K.fbin"),
        gt_bin=Path(
            "/gpfs/scratch/delgaudio/vectors/DEEP/computedGT_deep10M_10kqueries_k100_faiss.bin"
        ),
        partition_dir=Path("/gpfs/scratch/delgaudio/vectors/DEEP/new_experiments_gpann"),
        partition_stem="base.10M.partition",
        qa_dir=Path("/gpfs/scratch/delgaudio/vectors/DEEP/new_experiments_queryAttribution"),
        qa_stem="base.10M",
    ),
    "mips": DatasetConfig(
        name="mips",
        job_tag="tti",
        queryattr_bin=GP_ANN_ROOT / "release_mips" / "QueryAttribution",
        base_fbin=Path("/gpfs/scratch/delgaudio/vectors/TTI/yangt2i10M.fbin"),
        query_fbin=Path("/gpfs/scratch/delgaudio/vectors/TTI/yangt2iquery10K.fbin"),
        gt_bin=Path(
            "/gpfs/scratch/delgaudio/vectors/TTI/computedGT_tti10M_10kqueries_k100_faiss.bin"
        ),
        partition_dir=Path("/gpfs/scratch/delgaudio/vectors/TTI/new_experiments_gpann"),
        partition_stem="yangt2i10M.partition",
        qa_dir=Path("/gpfs/scratch/delgaudio/vectors/TTI/new_experiments_queryAttribution"),
        qa_stem="yangt2i10M",
    ),
}

# CLI --metric deep|tti|all maps to partition keys
METRIC_ALIASES = {
    "deep": "l2",
    "tti": "mips",
    "l2": "l2",
    "mips": "mips",
    "all": "all",
}


def partition_file(cfg, k, overlap):
    o = "{:g}".format(overlap)
    return cfg.partition_dir / "{}.k={}.GP.o={}".format(cfg.partition_stem, k, o)


def qa_out_prefix(cfg, k, overlap):
    o = "{:g}".format(overlap)
    return cfg.qa_dir / "{}.k{}_o{}".format(cfg.qa_stem, k, o)


def slurm_job_name(cfg, k, overlap):
    o_tag = str(int(round(overlap * 10)))
    return "{}_o{}_s{}_qa".format(cfg.job_tag, o_tag, k)


def submit_one(cfg, k, overlap, dry_run):
    part = partition_file(cfg, k, overlap)
    out = qa_out_prefix(cfg, k, overlap)
    job_name = slurm_job_name(cfg, k, overlap)

    if not part.is_file():
        print("SKIP missing partition: {}".format(part), file=sys.stderr)
        return 2

    export = {
        "BASE_FBIN": str(cfg.base_fbin),
        "QUERY_FBIN": str(cfg.query_fbin),
        "GT_BIN": str(cfg.gt_bin),
        "PARTITION_FILE": str(part),
        "OUT_PREFIX": str(out),
        "QUERYATTR_BIN": str(cfg.queryattr_bin),
        "GT_K": str(GT_K),
    }
    export_pairs = ",".join("{}={}".format(key, value) for key, value in export.items())

    log_base = LOG_DIR / "{}_%j".format(job_name)
    cmd = [
        "sbatch",
        "--job-name={}".format(job_name),
        "-o{}.log".format(log_base),
        "-e{}.log".format(log_base),
        "--export=ALL,{}".format(export_pairs),
        str(SBATCH_SCRIPT),
    ]

    print("{} {}".format("[dry-run]" if dry_run else "", " ".join(cmd)))
    print("  partition: {}".format(part))
    print("  qa out:    {}.{{routes,searches,pareto}}".format(out))
    print("  router:    {}.routing_index".format(part))
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
    p.add_argument(
        "--metric",
        choices=("l2", "mips", "deep", "tti", "all"),
        default="all",
    )
    p.add_argument("--k", type=int, action="append", dest="k_values")
    p.add_argument("--overlap", type=float, action="append", dest="overlaps")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_metrics(metric_arg):
    key = METRIC_ALIASES[metric_arg]
    if key == "all":
        return [DATASETS["l2"], DATASETS["mips"]]
    return [DATASETS[key]]


def main():
    args = parse_args()
    k_values = tuple(args.k_values) if args.k_values else K_VALUES
    overlaps = tuple(args.overlaps) if args.overlaps else OVERLAP_VALUES
    metrics = resolve_metrics(args.metric)

    if not SBATCH_SCRIPT.is_file():
        print("FATAL: missing {}".format(SBATCH_SCRIPT), file=sys.stderr)
        return 1

    for cfg in metrics:
        if not cfg.queryattr_bin.is_file():
            print("FATAL: missing {}".format(cfg.queryattr_bin), file=sys.stderr)
            return 1
        if not args.dry_run:
            if not cfg.base_fbin.is_file():
                print("FATAL: missing {}".format(cfg.base_fbin), file=sys.stderr)
                return 1
            if not cfg.query_fbin.is_file():
                print("FATAL: missing {}".format(cfg.query_fbin), file=sys.stderr)
                return 1
            if not cfg.gt_bin.is_file():
                print("FATAL: missing {}".format(cfg.gt_bin), file=sys.stderr)
                return 1
            cfg.qa_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    n_jobs = 0
    errors = 0
    skipped = 0
    for cfg in metrics:
        for k in k_values:
            for overlap in overlaps:
                n_jobs += 1
                rc = submit_one(cfg, k, overlap, args.dry_run)
                if rc == 2:
                    skipped += 1
                elif rc:
                    errors += 1

    print(
        "\nDone: {} configured (submitted={}, skipped={}, errors={}).".format(
            n_jobs,
            0 if args.dry_run else n_jobs - skipped - errors,
            skipped,
            errors,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
