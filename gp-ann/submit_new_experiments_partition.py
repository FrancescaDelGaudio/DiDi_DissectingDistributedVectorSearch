#!/usr/bin/env python3
"""
Submit overlapping GP partition jobs for new_experiments_gpann.

  - L2 (release_l2/Partition) on DEEP base.10M.fbin
  - MIPS (release_mips/Partition) on TTI yangt2i10M.fbin

Sweeps K in {4, 6} and overlap in {0.1, 0.2, 0.3} → 6 jobs per metric, 12 total.

Outputs:
  /gpfs/scratch/delgaudio/vectors/DEEP/new_experiments_gpann/base.10M.partition.k=K.GP.o=O
  /gpfs/scratch/delgaudio/vectors/TTI/new_experiments_gpann/yangt2i10M.partition.k=K.GP.o=O

Usage:
  python3 submit_new_experiments_partition.py --dry-run
  python3 submit_new_experiments_partition.py
  python3 submit_new_experiments_partition.py --metric l2
  python3 submit_new_experiments_partition.py --metric mips --k 4 --overlap 0.1
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional

GP_ANN_ROOT = Path("/gpfs/users/delgaudio/exp_eval/gp-ann")
SBATCH_SCRIPT = GP_ANN_ROOT / "sbatch_partition_fbin.sh"
LOG_DIR = Path("/gpfs/scratch/delgaudio/gp_ann/new_experiments_gpann/logs")

K_VALUES = (4, 6)
OVERLAP_VALUES = (0.1, 0.2, 0.3)


class MetricConfig(NamedTuple):
    name: str
    partition_bin: Path
    dataset_fbin: Path
    out_prefix: Path
    job_tag: str


METRICS = {
    "l2": MetricConfig(
        name="l2",
        partition_bin=GP_ANN_ROOT / "release_l2" / "Partition",
        dataset_fbin=Path("/gpfs/scratch/delgaudio/vectors/DEEP/base.10M.fbin"),
        out_prefix=Path(
            "/gpfs/scratch/delgaudio/vectors/DEEP/new_experiments_gpann/base.10M.partition"
        ),
        job_tag="deep",
    ),
    "mips": MetricConfig(
        name="mips",
        partition_bin=GP_ANN_ROOT / "release_mips" / "Partition",
        dataset_fbin=Path("/gpfs/scratch/delgaudio/vectors/TTI/yangt2i10M.fbin"),
        out_prefix=Path(
            "/gpfs/scratch/delgaudio/vectors/TTI/new_experiments_gpann/yangt2i10M.partition"
        ),
        job_tag="tti",
    ),
}


def partition_output_path(cfg: MetricConfig, k: int, overlap: float) -> Path:
    o = f"{overlap:g}" if overlap == int(overlap) else str(overlap)
    return Path(f"{cfg.out_prefix}.k={k}.GP.o={o}")


def slurm_job_name(cfg: MetricConfig, k: int, overlap: float) -> str:
    """e.g. deep_o1_s4  (dataset tag, overlap 10% -> o1, K=4 -> s4)."""
    o_tag = str(int(round(overlap * 10)))  # 0.1 -> 1, 0.2 -> 2, 0.3 -> 3
    return f"{cfg.job_tag}_o{o_tag}_s{k}"


def submit_one(
    cfg: MetricConfig,
    k: int,
    overlap: float,
    dry_run: bool,
) -> Optional[int]:
    o_str = f"{overlap:g}"
    job_name = slurm_job_name(cfg, k, overlap)
    out_file = partition_output_path(cfg, k, overlap)

    export = {
        "DATASET_FBIN": str(cfg.dataset_fbin),
        "OUT_PREFIX": str(cfg.out_prefix),
        "K": str(k),
        "OVERLAP": o_str,
        "PARTITION_BIN": str(cfg.partition_bin),
        "SLURM_LOG_DIR": str(LOG_DIR),
    }
    export_pairs = ",".join(f"{key}={value}" for key, value in export.items())

    log_base = LOG_DIR / f"{job_name}_%j"
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"-o{log_base}.log",
        f"-e{log_base}.log",
        f"--export=ALL,{export_pairs}",
        str(SBATCH_SCRIPT),
    ]

    print(f"{'[dry-run] ' if dry_run else ''}{' '.join(cmd)}")
    print(f"  -> {out_file}")
    if dry_run:
        return None
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        return result.returncode
    job_id = result.stdout.strip().split()[-1]
    print(f"  submitted job {job_id}")
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metric",
        choices=("l2", "mips", "all"),
        default="all",
        help="Which metric/dataset to run (default: all)",
    )
    p.add_argument(
        "--k",
        type=int,
        action="append",
        dest="k_values",
        help=f"K values (default: {list(K_VALUES)})",
    )
    p.add_argument(
        "--overlap",
        type=float,
        action="append",
        dest="overlaps",
        help=f"Overlap fractions (default: {list(OVERLAP_VALUES)})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    k_values = tuple(args.k_values) if args.k_values else K_VALUES
    overlaps = tuple(args.overlaps) if args.overlaps else OVERLAP_VALUES

    if not SBATCH_SCRIPT.is_file():
        print(f"FATAL: missing sbatch script: {SBATCH_SCRIPT}", file=sys.stderr)
        return 1

    metrics = list(METRICS.values()) if args.metric == "all" else [METRICS[args.metric]]

    for cfg in metrics:
        if not cfg.partition_bin.is_file():
            print(f"FATAL: Partition binary not found: {cfg.partition_bin}", file=sys.stderr)
            return 1
        if not args.dry_run and not cfg.dataset_fbin.is_file():
            print(f"FATAL: dataset fbin not found: {cfg.dataset_fbin}", file=sys.stderr)
            return 1
        if not args.dry_run:
            cfg.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    n_jobs = 0
    errors = 0
    for cfg in metrics:
        for k in k_values:
            for overlap in overlaps:
                n_jobs += 1
                rc = submit_one(cfg, k, overlap, args.dry_run)
                if rc:
                    errors += 1

    print(
        f"\nDone: {n_jobs} job(s) "
        f"({'dry-run' if args.dry_run else 'submitted'}, errors={errors})."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
