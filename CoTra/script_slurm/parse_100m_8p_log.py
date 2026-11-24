#!/usr/bin/env python3
"""Parse CoTra scala_anns_8p leader .out logs (100M sweep)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"

EXPERIMENTS_100M_8P = [
    ("DEEP 100M 8p k=100", "6911861", "scala_anns_8p-cotra8_deep_k100-6911861.out"),
    ("DEEP 100M 8p k=10", "6911862", "scala_anns_8p-cotra8_deep_k10-6911862.out"),
    ("TTI 100M 8p k=100 (skip ef=20)", "6911863", "scala_anns_8p-cotra8_tti_k100-6911863.out"),
    ("TTI 100M 8p k=10 (skip ef=20)", "6911864", "scala_anns_8p-cotra8_tti_k10-6911864.out"),
]

PAT_COMM = re.compile(
    r"^(\d+)\tcomm_sum\twrites=(\d+)\treads=(\d+)\t"
    r"write_bytes=(\d+)\tread_bytes=(\d+)\s*$"
)
PAT_CORE = re.compile(
    r"^(\d+)\tcore_load\t"
    r"m0=(\d+)\tm1=(\d+)\tm2=(\d+)\tm3=(\d+)\t"
    r"m4=(\d+)\tm5=(\d+)\tm6=(\d+)\tm7=(\d+)\t"
    r"sum=(\d+)\torigin_per_node=(\d+)\s*$"
)
PAT_PERF = re.compile(
    r"^(\d+)\t(\d+\.\d+)\t([\d.]+)\s+us\t([\d.]+)\s+/s\s*$"
)
PAT_META_DATASET = re.compile(r"^dataset=(\w+)\s+dist_fn=(\w+)\s+res_knn=(\d+)")
PAT_META_START = re.compile(r"^==== scala_anns_8p start (.+) job=(\d+)")
PAT_META_DONE = re.compile(r"^==== scala_anns_8p done (.+) ====")


@dataclass
class EfRow:
    recalls: list[float] = field(default_factory=list)
    lat_us: list[float] = field(default_factory=list)
    qps: list[float] = field(default_factory=list)
    writes: int | None = None
    reads: int | None = None
    write_bytes: int | None = None
    read_bytes: int | None = None
    core_sum: int | None = None
    core_m: list[int] | None = None
    origin_per_node: int | None = None

    @property
    def recall_avg(self) -> float | None:
        return sum(self.recalls) / len(self.recalls) if self.recalls else None

    @property
    def recall_max(self) -> float | None:
        return max(self.recalls) if self.recalls else None

    @property
    def lat_avg(self) -> float | None:
        return sum(self.lat_us) / len(self.lat_us) if self.lat_us else None

    @property
    def qps_avg(self) -> float | None:
        return sum(self.qps) / len(self.qps) if self.qps else None


@dataclass
class LogMeta:
    job_id: str = ""
    start: str = ""
    done: str = ""
    dataset: str = ""
    dist_fn: str = ""
    res_knn: int = 0


def parse_log(path: Path) -> tuple[LogMeta, dict[int, EfRow]]:
    meta = LogMeta()
    rows: dict[int, EfRow] = {}
    if not path.is_file():
        return meta, rows

    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        m = PAT_META_START.match(line)
        if m:
            meta.start, meta.job_id = m.group(1), m.group(2)
            continue
        m = PAT_META_DONE.match(line)
        if m:
            meta.done = m.group(1)
            continue
        m = PAT_META_DATASET.match(line)
        if m:
            meta.dataset, meta.dist_fn, meta.res_knn = m.group(1), m.group(2), int(m.group(3))
            continue

        m = PAT_COMM.match(line)
        if m:
            ef = int(m.group(1))
            r = rows.setdefault(ef, EfRow())
            r.writes = int(m.group(2))
            r.reads = int(m.group(3))
            r.write_bytes = int(m.group(4))
            r.read_bytes = int(m.group(5))
            continue

        m = PAT_CORE.match(line)
        if m:
            ef = int(m.group(1))
            r = rows.setdefault(ef, EfRow())
            r.core_m = [int(m.group(i)) for i in range(2, 10)]
            r.core_sum = int(m.group(10))
            r.origin_per_node = int(m.group(11))
            continue

        m = PAT_PERF.match(line)
        if m:
            ef = int(m.group(1))
            r = rows.setdefault(ef, EfRow())
            r.recalls.append(float(m.group(2)))
            r.lat_us.append(float(m.group(3)))
            r.qps.append(float(m.group(4)))

    return meta, rows


def all_efs(data: dict[str, dict[int, EfRow]]) -> list[int]:
    s: set[int] = set()
    for rows in data.values():
        s.update(rows.keys())
    return sorted(s)
