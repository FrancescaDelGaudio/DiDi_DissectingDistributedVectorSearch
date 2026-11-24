#!/usr/bin/env python3
"""Recall/QPS panels for MSMARCO10M and MSMARCO100M on 4/6/8 nodes, from msmarco_curves.csv
(collect_msmarco_curves.py). Identical style to plot_curves_subfigs.py (default four-per-row size):
same Pareto fronts, colours, markers, axes; no title or legend (legend_curves.pdf is shared).

Output: subfigs/curve_msmarco10m_<n>n.pdf/png, subfigs/curve_msmarco100m_<n>n.pdf/png
Usage:
    module load python/3.12.2
    python3 plot_curves_msmarco.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_curves_subfigs import panel  # noqa: E402  same drawing code and rcParams

TARGETS = [0.80, 0.90, 0.95, 0.99]
XR = (0.69, 1.0)
SYSTEMS = {"MSMARCO10M": ["GP-ANN*", "Pyramid*", "CoTra"], "MSMARCO100M": ["CoTra"]}


def main():
    pts = defaultdict(list)
    for r in csv.DictReader(open(HERE / "msmarco_curves.csv")):
        pts[(r["setup"], r["system"])].append((float(r["recall10"]), float(r["qps"]), r["config"]))
    for ds, systems in SYSTEMS.items():
        for n in (4, 6, 8):
            panel(f"{ds.lower()}_{n}n", f"{ds}-{n}n", systems, TARGETS, XR, pts,
                  ysubs=(1.0, 2.0, 5.0) if ds == "MSMARCO100M" else (1.0, 5.0))  # one narrow curve
    print("wrote 6 MSMARCO curve panels to subfigs/")


if __name__ == "__main__":
    main()
