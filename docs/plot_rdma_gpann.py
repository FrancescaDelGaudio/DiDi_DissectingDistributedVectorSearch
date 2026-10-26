#!/usr/bin/env python3
"""GP-ANN*: would a hand-written RDMA layer save time? Step-by-step decomposition.

"MPI + waiting" (% of query-phase runtime, average rank; breakdown jobs 6961001 (nprobe=1 ef=70) / 6960834) is split into:
  - raw wire time at 100 Gb/s               not removable (physical): bytes on the busiest rank / link rate
  - MPI software cost                        removable: the same query phase replayed with GP_COMM_ONLY=1
                                             (job 6965110: same routing, messages, buffering and termination
                                             protocol, no search), communication time of the average rank,
                                             minus wire time; fastest / average of 3 repeats
  - waiting for the slowest rank + extra     not removable by RDMA (the rest)
The replay includes the message-queue termination protocol, which a custom RDMA layer would also
need, so the MPI software cost is an upper bound on the saving.

Usage:
    python3 plot_rdma_gpann.py   # writes rdma_gpann.png/.pdf next to this file
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
    ("rest",  "Waiting for slowest rank + extra cost (not removable)",     "#c9c8c3", "////"),
    ("wire",  "Raw wire time (not removable)",                              "#7a7974", None),
    ("sw",    "MPI software cost (removable by RDMA)",                      "#e87ba4", None),
]

# % of runtime: comm-related total, wire, MPI software (fastest / average replay)
RUNS = [
    # target, total, wire, sw_lo, sw_hi
    ("0.80", 16.0, 0.02, 7.5, 9.0),
    ("0.90", 37.5, 0.05, 5.2, 5.5),
    ("0.95*", 35.8, 0.03, 4.1, 4.9),
    ("0.99", 35.2, 0.02, 1.9, 2.0),
]


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK,
        "hatch.color": "#ffffff", "hatch.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
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
                ax.text(left + w / 2, y + 0.38, "wire <0.1%", ha="center", va="top", fontsize=6.5, color=INK_2)
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
    ax.set_xlim(0, 45)
    ax.set_xlabel("Share of query-phase runtime (%)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.suptitle("GP-ANN*: what a hand-written RDMA layer could remove", x=0.01, ha="left",
                 fontsize=9, fontweight="bold", color=INK)
    handles = [Patch(facecolor=c, hatch=h, edgecolor=SURFACE, label=l) for _, l, c, h in SEGMENTS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.5)
    fig.text(0.01, 0.015,
             "Total = MPI + waiting of the average rank. MPI software cost: the same query phase replayed with no search (same routing,\n"
             "messages, buffering and termination protocol) minus wire time; it includes termination detection, so it is an upper bound.\n"
             "* Measured recall 0.949.",
             fontsize=6.3, color=INK_2, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 0.985))
    out = Path(__file__).resolve().parent
    fig.savefig(out / "rdma_gpann.pdf")
    fig.savefig(out / "rdma_gpann.png", dpi=200)
    print(f"wrote {out / 'rdma_gpann.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
