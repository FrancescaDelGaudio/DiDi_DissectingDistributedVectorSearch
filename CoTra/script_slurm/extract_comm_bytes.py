#!/usr/bin/env python3
"""Extract comm_sum lines from CoTra leader .out logs."""
import re
import sys
from pathlib import Path

PAT = re.compile(
    r"^(\d+)\tcomm_sum\twrites=(\d+)\treads=(\d+)\t"
    r"write_bytes=(\d+)\tread_bytes=(\d+)\s*$"
)

EXPERIMENTS = [
    ("DEEP 4p k=10", "6911247", "scala_anns-cotra4_deep_k10_comm-6911247.out"),
    ("TTI 4p k=10 (skip ef=20)", "6911248", "scala_anns-cotra4_tti_k10_comm-6911248.out"),
    ("DEEP 6p k=10", "6911249", "scala_anns_6p-cotra6_deep_k10_comm-6911249.out"),
    ("TTI 6p k=10 (skip ef=20)", "6911250", "scala_anns_6p-cotra6_tti_k10_comm-6911250.out"),
]

LOG_DIR = Path(__file__).resolve().parent / "logs"


def parse_log(path: Path) -> dict[int, tuple]:
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        m = PAT.match(line.strip())
        if m:
            ef = int(m.group(1))
            rows[ef] = tuple(int(m.group(i)) for i in range(2, 6))
    return rows


def main():
    all_efs = set()
    data = {}
    for label, job, fname in EXPERIMENTS:
        rows = parse_log(LOG_DIR / fname)
        data[label] = rows
        all_efs.update(rows.keys())
    efs = sorted(all_efs)
    for label, rows in data.items():
        print(f"# {label} ({len(rows)} ef values)")
        for ef in efs:
            if ef in rows:
                w, r, wb, rb = rows[ef]
                print(f"{ef}\t{w}\t{r}\t{wb}\t{rb}")
        print()


if __name__ == "__main__":
    main()
