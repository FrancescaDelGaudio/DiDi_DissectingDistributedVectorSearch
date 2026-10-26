#!/usr/bin/env python3
"""Recall@10 vs QPS Pareto frontiers, one figure per setup, all systems together.

Input: pareto_data.csv (from collect_pareto_data.py). Original code of every system.
Output: pareto_<setup>.png/.pdf (one per setup) and pareto_grid.png/.pdf (2x2), next to this file.

Usage:
    module load anaconda3/2023.09
    python3 collect_pareto_data.py && python3 plot_pareto.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
XMIN = 0.80

# Fixed colour + marker per system (marker = second channel, colour is never alone).
SYSTEMS = [
    ("Pyramid",        "Pyramid (1 thread)",                  "#2a78d6", "o"),
    ("DistributedANN", "DistributedANN (one query at a time)", "#eb6834", "s"),
    ("GP-ANN",         "GP-ANN (1-thread local search)",       "#1baf7a", "^"),
    ("CoTra",          "CoTra (16 threads)",                   "#eda100", "D"),
    ("Harmony",        "Harmony (16 threads)",                 "#e87ba4", "v"),
]
SETUPS = [("DEEP10M-4n", "DEEP10M, 4 nodes"), ("DEEP10M-6n", "DEEP10M, 6 nodes"),
          ("TTI10M-4n", "TTI10M, 4 nodes"), ("TTI10M-6n", "TTI10M, 6 nodes")]


def frontier(points):
    """Pareto frontier: no other point has both higher recall and higher QPS."""
    best = []
    top_qps = -1.0
    for r, q in sorted(points, key=lambda p: (-p[0], -p[1])):
        if q > top_qps:
            best.append((r, q))
            top_qps = q
    return sorted(best)


def load():
    data = defaultdict(lambda: defaultdict(list))
    with open(Path(__file__).resolve().parent / "pareto_data.csv") as fh:
        for row in csv.DictReader(fh):
            data[row["setup"]][row["system"]].append((float(row["recall10"]), float(row["qps"])))
    return data


def draw(ax, setup, title, data, show_legend):
    missing = []
    for key, label, colour, marker in SYSTEMS:
        pts = data[setup].get(key, [])
        fr = frontier(pts)
        if not fr or max(r for r, _ in fr) < XMIN:
            if pts:
                missing.append(f"{key} (max {max(r for r, _ in pts):.2f})")
            continue
        xs, ys = zip(*fr)
        ax.plot(xs, ys, color=colour, linewidth=1.6, marker=marker, markersize=4.2,
                markeredgecolor="#ffffff", markeredgewidth=0.8, label=label, zorder=3)
    for x in (0.90, 0.95):
        ax.axvline(x, color=INK_2, linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xlim(XMIN, 1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second")
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color=INK, pad=4)
    ax.grid(which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if missing:
        ax.text(0.02, 0.03, "Not shown, recall < 0.80:\n" + "\n".join(missing), transform=ax.transAxes,
                fontsize=6.3, color=INK_2, va="bottom", ha="left")
    if show_legend:
        ax.legend(frameon=False, fontsize=6.8, loc="upper right")


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK_2,
                         "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2})
    data = load()
    out = Path(__file__).resolve().parent
    for setup, title in SETUPS:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        draw(ax, setup, title, data, show_legend=True)
        fig.tight_layout()
        slug = setup.lower().replace("-", "_")
        fig.savefig(out / f"pareto_{slug}.pdf")
        fig.savefig(out / f"pareto_{slug}.png", dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for ax, (setup, title) in zip(axes.flat, SETUPS):
        draw(ax, setup, title, data, show_legend=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "pareto_grid.pdf")
    fig.savefig(out / "pareto_grid.png", dpi=200)
    print("wrote pareto_<setup>.png/.pdf and pareto_grid.png/.pdf in", out)


if __name__ == "__main__":
    main()
