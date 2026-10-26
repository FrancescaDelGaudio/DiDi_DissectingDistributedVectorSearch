#!/usr/bin/env python3
"""DistributedANN: would a hand-written RDMA layer save time? Step-by-step decomposition.

For each recall target, the MPI communication time (% of query-phase runtime) is split into:
  2.  waiting for the slowest node          not removable (idle time)
  3a. raw wire time at 100 Gb/s             not removable (physical)
  3c. extra cost seen only in the real run  probably not removable (inferred: 3 - 3a - 3b)
  3b. MPI software cost                     removable: measured by replaying the same
                                            exchanges with no compute, minus wire time
Realistic saving = 3b. Ceiling = 3b + 3c.

Usage:
    module load anaconda3/2023.09
    python3 plot_rdma_dann.py   # writes rdma_distributedann.png/.pdf next to this file
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#ffffff"

# Segment colours: adjacent pairs checked for normal and colour-blind separation.
SEGMENTS = [
    # key,      label,                                           colour,    hatch
    ("wait",    "Waiting for slowest node (not removable)",      "#6250d6", None),
    ("wire",    "Raw wire time (not removable)",                 "#7a7974", None),
    ("extra",   "Extra cost only in the real run (probably not)", "#c9c8c3", "////"),
    ("sw",      "MPI software cost (removable by RDMA)",         "#e87ba4", None),
]

# % of runtime. sw_lo/sw_hi: MPI software cost from the fastest / average replay;
# the bar draws sw_hi and extra = inside - wire - sw_hi, so the bar adds up to the MPI total.
RUNS = [
    # target, mpi_total, waiting, inside, wire, sw_lo, sw_hi, ceiling (= inside - wire),
    # extra range (inside - wire - sw, from the unrounded measurements; same as breakdown_tables.tex)
    ("0.90",  16.9, 8.2, 8.7,  0.65, 0.6, 1.8, 8.1,  (6.2, 7.5)),
    ("0.93*", 22.5, 9.9, 12.6, 0.73, 0.8, 2.6, 11.9, (9.3, 11.1)),
]


def ink_on(hex_colour):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return INK if 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] > 0.28 else "#ffffff"


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK,
        "hatch.color": "#ffffff", "hatch.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    xmax = 26
    for y, (target, total, wait, inside, wire, sw_lo, sw_hi, ceiling, extra_range) in enumerate(RUNS):
        extra = inside - wire - sw_hi
        widths = {"wait": wait, "wire": wire, "extra": extra, "sw": sw_hi}
        left = 0.0
        pos = {}
        for key, _, colour, hatch in SEGMENTS:
            w = widths[key]
            ax.barh(y, w, left=left, height=0.5, color=colour, hatch=hatch, edgecolor=SURFACE, linewidth=1.0)
            pos[key] = (left, w)
            if key == "wait":
                ax.text(left + w / 2, y, f"{w:.1f}%", ha="center", va="center", fontsize=7, color=ink_on(colour))
            if key == "extra":
                chip = dict(boxstyle="round,pad=0.15", facecolor=colour, edgecolor="none")
                lo, hi = extra_range
                ax.text(left + w / 2, y, f"{lo:.1f}–{hi:.1f}%", ha="center", va="center", fontsize=7, color=INK, bbox=chip)
            left += w
        # MPI total at the end of the bar
        ax.text(left + 0.3, y, f"MPI total {total:.1f}%", ha="left", va="center", fontsize=7, color=INK_2)
        # brackets below the bar: realistic saving (3b) and ceiling (3b + 3c)
        sl, sw = pos["sw"]
        el, ew = pos["extra"]
        yb = y + 0.38
        ax.plot([el, el, sl + sw, sl + sw], [yb - 0.04, yb, yb, yb - 0.04], color=INK_2, linewidth=0.7)
        ax.text(el + ew / 2, yb + 0.05, f"ceiling {ceiling:.1f}%", ha="center", va="top", fontsize=6.5, color=INK_2)
        ax.annotate(f"realistic saving\n{sw_lo:.1f}–{sw_hi:.1f}%", xy=(sl + sw / 2, y - 0.25),
                    xytext=(sl + sw / 2 + 1.6, y - 0.62), ha="left", va="center", fontsize=7, color=INK,
                    fontweight="bold", arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.7))
    ax.set_yticks(range(len(RUNS)))
    ax.set_yticklabels([r[0] for r in RUNS], fontsize=8)
    ax.set_ylabel("Recall@10 target")
    ax.set_ylim(len(RUNS) - 0.35, -0.85)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Share of query-phase runtime (%)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.suptitle("DistributedANN: what a hand-written RDMA layer could remove", x=0.01, ha="left",
                 fontsize=9, fontweight="bold", color=INK)
    handles = [Patch(facecolor=c, hatch=h, edgecolor=SURFACE, label=l) for _, l, c, h in SEGMENTS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.5)
    fig.text(0.01, 0.015,
             "MPI software cost: the same exchanges replayed with no computation, minus wire time. "
             "Extra cost = time inside the exchanges − wire − software\n(memory setup, compute threads "
             "competing for CPU, nodes out of step); a custom RDMA layer would face it too. "
             "* Highest recall reached (0.928).",
             fontsize=6.3, color=INK_2, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.985))
    out = Path(__file__).resolve().parent
    fig.savefig(out / "rdma_distributedann.pdf")
    fig.savefig(out / "rdma_distributedann.png", dpi=200)
    print(f"wrote {out / 'rdma_distributedann.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
