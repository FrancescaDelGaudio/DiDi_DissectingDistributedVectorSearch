#!/usr/bin/env python3
"""Harmony: would a hand-written RDMA layer save time? Step-by-step decomposition.

Communication-related time (% of query-phase runtime) = query distribution on the master +
MPI + waiting on the workers (from the breakdown job 6960833). It is split into:
  - raw wire time at 100 Gb/s               not removable (physical)
  - MPI software cost                        removable: Harmony's exchanges replayed with no
                                             computation (same sizes, same team pairs; job
                                             6962603), minus wire time
  - waiting, worker query setup, extra cost  not removable by RDMA (the rest)
The replay ran on 6 nodes (two nodes shared by two workers), so the MPI cost is an upper bound.

Usage:
    module load anaconda3/2023.09
    python3 plot_rdma_harmony.py   # writes rdma_harmony.png/.pdf next to this file
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#ffffff"

SEGMENTS = [
    # key,    label,                                                        colour,    hatch
    ("rest",  "Waiting, worker query setup, extra cost (not removable)",    "#c9c8c3", "////"),
    ("wire",  "Raw wire time (not removable)",                              "#7a7974", None),
    ("sw",    "MPI software cost (removable by RDMA)",                      "#e87ba4", None),
]

# % of runtime: comm-related total, wire, MPI software (fastest / average replay)
RUNS = [
    # target, total, wire, sw_lo, sw_hi
    ("0.90", 46.5, 3.3, 1.3, 1.4),
    ("0.95", 43.3, 4.2, 0.1, 1.1),
]


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK,
        "hatch.color": "#ffffff", "hatch.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    STEP = 1.7   # vertical distance between bars, leaves room for the labels
    for i, (target, total, wire, sw_lo, sw_hi) in enumerate(RUNS):
        y = i * STEP
        rest = total - wire - sw_hi
        left = 0.0
        for key, _, colour, hatch in SEGMENTS:
            w = {"rest": rest, "wire": wire, "sw": sw_hi}[key]
            ax.barh(y, w, left=left, height=0.5, color=colour, hatch=hatch, edgecolor=SURFACE, linewidth=1.0)
            if key == "rest":
                chip = dict(boxstyle="round,pad=0.15", facecolor=colour, edgecolor="none")
                ax.text(left + w / 2, y, f"{total - wire - sw_hi:.1f}–{total - wire - sw_lo:.1f}%",
                        ha="center", va="center", fontsize=7, color=INK, bbox=chip)
            if key == "wire":
                ax.text(left + w / 2, y + 0.38, f"wire {w:.1f}%", ha="center", va="top", fontsize=6.5, color=INK_2)
            if key == "sw":
                ax.annotate(f"realistic saving\n{sw_lo:.1f}–{sw_hi:.1f}%", xy=(left + w / 2, y - 0.25),
                            xytext=(left + w / 2 + 1.5, y - 0.62), ha="left", va="center", fontsize=7,
                            color=INK, fontweight="bold", arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.7))
            left += w
        ax.text(left + 0.6, y, f"total {total:.1f}%", ha="left", va="center", fontsize=7, color=INK_2)
    ax.set_yticks([i * STEP for i in range(len(RUNS))])
    ax.set_yticklabels([r[0] for r in RUNS], fontsize=8)
    ax.set_ylabel("Recall@10 target")
    ax.set_ylim((len(RUNS) - 1) * STEP + 0.65, -0.9)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Share of query-phase runtime (%)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.suptitle("Harmony: what a hand-written RDMA layer could remove", x=0.01, ha="left",
                 fontsize=9, fontweight="bold", color=INK)
    handles = [Patch(facecolor=c, hatch=h, edgecolor=SURFACE, label=l) for _, l, c, h in SEGMENTS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.5)
    fig.text(0.01, 0.015,
             "Total = query distribution (master) + MPI and waiting on the workers. MPI software cost: Harmony's exchanges replayed\n"
             "with no computation (same sizes and team pairs) minus wire time; measured on 6 nodes, so an upper bound.\n"
             "The replayed query distribution takes 3–4 ms of the ~1 s measured: the rest is worker query setup and waiting.",
             fontsize=6.3, color=INK_2, va="bottom")
    fig.tight_layout(rect=(0, 0.11, 1, 0.985))
    out = Path(__file__).resolve().parent
    fig.savefig(out / "rdma_harmony.pdf")
    fig.savefig(out / "rdma_harmony.png", dpi=200)
    print(f"wrote {out / 'rdma_harmony.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
