#!/usr/bin/env python3
"""
Parse CoTra scala_anns logs into structured arrays per config.

Four configs: deep4, deep6, tti4, tti6 (nodes = 4 or 6).
Two result sets each: k=10 and k=100.

Usage (after jobs finish):
  python3 script_slurm/parse_cotra_log_arrays.py --job-deep4-k100 6879991
  python3 script_slurm/parse_cotra_log_arrays.py --log-dir script_slurm/logs --glob 'scala_anns*cotra_deep_k100*'

Output: JSON or Python dict with ef-aligned arrays.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_EFS = [
    3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 400, 500,
    600, 700,
]

FOLLOWER_LINE = re.compile(
    r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+us\s+([\d.]+)\s+/s\s*$"
)
CORE_LINE = re.compile(
    r"\[CoTra\] ef=(\d+) m=(\d+) core_participations=(\d+) origin_queries=(\d+)"
)


def parse_follower_metrics(out_text: str) -> dict[int, dict[str, float]]:
    """Per-ef metrics from follower stdout lines (any node; same recall cluster-wide)."""
    by_ef: dict[int, list[tuple[float, float, float]]] = {}
    for line in out_text.splitlines():
        m = FOLLOWER_LINE.match(line.strip())
        if not m:
            continue
        ef, recall, lat_us, qps = (
            int(m.group(1)),
            float(m.group(2)),
            float(m.group(3)),
            float(m.group(4)),
        )
        by_ef.setdefault(ef, []).append((recall, lat_us, qps))
    result: dict[int, dict[str, float]] = {}
    for ef, rows in by_ef.items():
        recalls = [r[0] for r in rows]
        lats = [r[1] for r in rows]
        qpss = [r[2] for r in rows]
        result[ef] = {
            "recall": sum(recalls) / len(recalls),
            "latency_us": sum(lats) / len(lats),
            "qps_per_node": sum(qpss) / len(qpss),
        }
    return result


def parse_core_load(err_text: str, nodes: int) -> dict[int, list[int]]:
    """Per-ef core_participations per node from .err (authoritative)."""
    by_ef: dict[int, dict[int, int]] = {}
    for line in err_text.splitlines():
        m = CORE_LINE.search(line)
        if not m:
            continue
        ef, mid, core, _origin = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(
            m.group(4)
        )
        if mid >= nodes:
            continue
        by_ef.setdefault(ef, {})[mid] = core
    out: dict[int, list[int]] = {}
    for ef, mp in sorted(by_ef.items()):
        out[ef] = [mp.get(m, 0) for m in range(nodes)]
    return out


def build_arrays(
    nodes: int,
    metrics_by_ef: dict[int, dict[str, float]],
    core_by_ef: dict[int, list[int]],
    efs: list[int] | None = None,
) -> dict[str, Any]:
    efs = efs or DEFAULT_EFS
    recall, latency_us, qps_per_node, core_m, core_sum = [], [], [], [], []
    for ef in efs:
        met = metrics_by_ef.get(ef)
        core = core_by_ef.get(ef)
        if met is None or core is None or len(core) != nodes:
            recall.append(None)
            latency_us.append(None)
            qps_per_node.append(None)
            core_m.append([None] * nodes)
            core_sum.append(None)
            continue
        recall.append(met["recall"])
        latency_us.append(met["latency_us"])
        qps_per_node.append(met["qps_per_node"])
        core_m.append(core)
        core_sum.append(sum(core))
    return {
        "nodes": nodes,
        "origin_queries_per_node": 10000 // nodes,
        "ef": efs,
        "recall": recall,
        "latency_us": latency_us,
        "qps_per_node": qps_per_node,
        "core_participations": core_m,
        "core_participations_sum": core_sum,
    }


def parse_log_pair(out_path: Path, err_path: Path, nodes: int) -> dict[str, Any]:
    out_text = out_path.read_text(errors="replace")
    err_text = err_path.read_text(errors="replace") if err_path.exists() else ""
    metrics = parse_follower_metrics(out_text)
    core = parse_core_load(err_text, nodes)
    efs = sorted(set(metrics.keys()) | set(core.keys()))
    return build_arrays(nodes, metrics, core, efs)


def merge_single_ef_logs(log_dir: Path, prefix: str, nodes: int) -> dict[str, Any]:
    """Merge TTI 4p k=10 single-ef jobs (one log pair per ef)."""
    metrics: dict[int, dict[str, float]] = {}
    core: dict[int, list[int]] = {}
    for out in sorted(log_dir.glob(f"{prefix}*.out")):
        err = out.with_suffix(".err")
        if not err.exists():
            err = Path(str(out).replace(".out", ".err"))
        chunk = parse_log_pair(out, err, nodes)
        for i, ef in enumerate(chunk["ef"]):
            if chunk["recall"][i] is not None:
                metrics[ef] = {
                    "recall": chunk["recall"][i],
                    "latency_us": chunk["latency_us"][i],
                    "qps_per_node": chunk["qps_per_node"][i],
                }
            if chunk["core_participations"][i][0] is not None:
                core[ef] = chunk["core_participations"][i]
    return build_arrays(nodes, metrics, core, DEFAULT_EFS)


CONFIGS = {
    "deep4": {"nodes": 4, "glob_sweep": "scala_anns-cotra_deep_k*"},
    "tti4": {"nodes": 4, "glob_sweep": "scala_anns-cotra_tti_k100*"},
    "tti4_k10": {"nodes": 4, "glob_single_ef": "scala_anns-cotra4_tti_k10_ef*"},
    "deep6": {"nodes": 6, "glob_sweep": "scala_anns_6p-cotra6_deep_k*"},
    "tti6": {"nodes": 6, "glob_sweep": "scala_anns_6p-cotra6_tti_k*"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", type=Path, default=Path("script_slurm/logs"))
    ap.add_argument("--out", choices=("json", "python"), default="json")
    ap.add_argument("--job-id", type=str, help="Single job id → parse matching logs")
    args = ap.parse_args()
    log_dir = args.log_dir
    result: dict[str, Any] = {}

    def latest(gpat: str) -> Path | None:
        hits = sorted(log_dir.glob(gpat + "*.out"), key=lambda p: p.stat().st_mtime)
        return hits[-1] if hits else None

    for name, k, nodes, gpat in [
        ("deep4", "k100", 4, "scala_anns-cotra_deep_k100*"),
        ("deep4", "k10", 4, "scala_anns-cotra_deep_k10*"),
        ("tti4", "k100", 4, "scala_anns-cotra_tti_k100*"),
        ("deep6", "k100", 6, "scala_anns_6p-cotra6_deep_k100*"),
        ("deep6", "k10", 6, "scala_anns_6p-cotra6_deep_k10*"),
        ("tti6", "k100", 6, "scala_anns_6p-cotra6_tti_k100*"),
        ("tti6", "k10", 6, "scala_anns_6p-cotra6_tti_k10*"),
    ]:
        if name == "tti4" and k == "k10":
            result.setdefault("tti4", {})["nodes"] = 4
            result["tti4"][k] = merge_single_ef_logs(
                log_dir, "scala_anns-cotra4_tti_k10_ef", 4
            )
            continue
        out = latest(gpat)
        if out is None:
            continue
        err = Path(str(out).replace(".out", ".err"))
        key = name
        result.setdefault(key, {"nodes": nodes})
        result[key][k] = parse_log_pair(out, err, nodes)

    if args.out == "json":
        print(json.dumps(result, indent=2))
    else:
        print("result = ", end="")
        print(repr(result))


if __name__ == "__main__":
    main()
