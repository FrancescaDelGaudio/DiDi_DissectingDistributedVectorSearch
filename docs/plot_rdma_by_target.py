#!/usr/bin/env python3
"""RDMA gain by recall target (DEEP100M, 8 nodes): one panel per target, systems on x.

Each bar is the communication-related share of the query-phase runtime of a system, split into
  not removable  waiting for the slowest node, worker setup and extra cost of the real run
  wire time      bytes over the busiest link / 100 Gb/s
  MPI software   communication-only replay minus wire time (removable by a native RDMA layer)
The label above the bar is the QPS gain bound 1/(1 - s) - 1, with s the MPI software share
(range: fastest / average replay). Data are those of plot_rdma_{gpann,dann,harmony}.py.
The analysis covers the MPI-based systems with a large communication phase (GP-ANN*,
DistributedANN*, Harmony); missing cells are
marked "not measured". Highest target: 0.99, 0.93 for DistributedANN*. Pyramid* (MPI transfer < 2%) and CoTra (already native RDMA) are excluded.

Usage:
    python3 plot_rdma_by_target.py   # writes rdma_by_target.png/.pdf
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_breakdown import INK, INK_2, SURFACE, ink_on  # noqa: E402

SEGMENTS = [  # key, label, colour, hatch
    ("rest", "Waiting and extra cost (not removable)", "#c9c8c3", "////"),
    ("wire", "Raw wire time (not removable)",          "#7a7974", None),
    ("sw",   "MPI software cost (removable by RDMA)",  "#e87ba4", None),
]
# system -> {target: (measured target label, comm total %, wire %, sw_lo %, sw_hi %)}
DATA = {
    "GP-ANN*": {"0.80": ("0.80", 16.0, 0.02, 7.5, 9.0), "0.90": ("0.90", 37.5, 0.05, 5.2, 5.5),
                "max": ("0.99", 35.2, 0.02, 1.9, 2.0)},
    # DistributedANN* 0.80: BW=40 (recall 0.821), breakdown job 6960999 (barrier run + empty-exchange floor)
    "Distr.ANN*": {"0.80": ("0.80", 16.3, 0.43, 0.45, 1.62), "0.90": ("0.90", 16.9, 0.65, 0.6, 1.8),
                   "max": ("0.93", 22.5, 0.73, 0.8, 2.6)},
    # Harmony 0.80 (nprobe=6, recall 0.788) and 0.99 (nprobe=96): replay job 6975581
    # total = query distribution + MPI + waiting + heap exchange ("Other" of the breakdown); the
    # replay already includes the heap exchange, so sw and wire are unchanged
    "Harmony": {"0.80": ("0.80", 56.8, 1.73, 0.27, 0.33), "0.90": ("0.90", 49.2, 3.3, 1.3, 1.4),
                "0.95": ("0.95", 47.4, 4.2, 0.1, 1.1), "max": ("0.99", 44.8, 5.62, 0.94, 0.96)},
}
TARGETS = [("0.80", "Recall@10 ≥ 0.80"), ("0.90", "Recall@10 ≥ 0.90"), ("max", "Highest target")]


def gain(s):
    return 100.0 * (1.0 / (1.0 - s / 100.0) - 1.0)


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK_2, "hatch.color": "#ffffff", "hatch.linewidth": 0.8,
    })
    systems = list(DATA)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(7.0, 3.0), sharey=True)
    ymax = 62
    for ax, (target, title) in zip(axes, TARGETS):
        ticks = []
        for x, name in enumerate(systems):
            cell = DATA[name].get(target)
            if cell is None:
                ax.text(x, 1.0, "not measured", rotation=90, ha="center", va="bottom", fontsize=6.5,
                        color=INK_2)
                ticks.append(name)
                continue
            tgt, total, wire, sw_lo, sw_hi = cell
            widths = {"rest": total - wire - sw_hi, "wire": wire, "sw": sw_hi}
            bottom = 0.0
            for key, _, colour, hatch in SEGMENTS:
                h = widths[key]
                ax.bar(x, h, bottom=bottom, width=0.6, color=colour, hatch=hatch,
                       edgecolor=SURFACE, linewidth=0.8)
                bottom += h
            g_lo, g_hi = gain(sw_lo), gain(sw_hi)
            ax.text(x, total + 1.0, f"+{g_lo:.1f}–{g_hi:.1f}%", ha="center", va="bottom", fontsize=6.5,
                    color=INK, fontweight="bold")
            ticks.append(name if target != "max" else f"{name}\n(at {tgt})")
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_xlim(-0.6, len(systems) - 0.4)
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontsize=9, pad=6, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of query-phase runtime (%)")
    handles = [Patch(facecolor=c, hatch=h, edgecolor=SURFACE, label=l) for _, l, c, h in SEGMENTS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               fontsize=7.5, handlelength=1.6, columnspacing=1.0)
    fig.text(0.5, 0.005, "Label above each bar: QPS gain bound 1/(1-s)-1 from removing the MPI software cost",
             ha="center", va="bottom", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92), w_pad=0.8)
    out = Path(__file__).resolve().parent
    fig.savefig(out / "rdma_by_target.pdf")
    fig.savefig(out / "rdma_by_target.png", dpi=250)
    print(f"wrote {out / 'rdma_by_target.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
