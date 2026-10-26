#!/usr/bin/env python3
"""Index construction time of the five systems, grouped bars per setup (log scale).

Totals from the index construction tables of the paper. GP-ANN* (overlap 0.2, the one used in
the recall/QPS curves) = Partition + QueryAttribution + per-shard HNSW construction, which runs at
the start of DistributedBench and is not in its table: estimated as the duration of a DistributedBench
task minus the same task without index construction (communication-only replay, ~125 s on 100M;
~15 s assumed on 10M, where 10x less data is loaded). Harmony: same time on 4 and 6 nodes.
Values marked approximate in the tables (~) are used as given.

Usage:
    python3 plot_build_time.py   # writes final_load_comm/build_time.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter, LogLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_final_curves import STYLE, INK, INK2, GRID  # noqa: E402  same colours/markers as the curves

SETUPS = ["DEEP\n10M\n4 nodes", "DEEP\n10M\n6 nodes", "TTI\n10M\n4 nodes", "TTI\n10M\n6 nodes",
          "DEEP\n100M\n8 nodes", "TTI\n100M\n8 nodes"]
# seconds; None = not built at that scale
BUILD = {
    "GP-ANN*":         [2209 + 186, 1640 + 124, 1909 + 202, 2001 + 136, 9641 + 1086, 12363 + 1160],
    "Pyramid*":        [9960, 13920, 420, 400, 178920, 24660],
    "Harmony":         [59.2, 59.2, 111.0, 111.0, 558.94, None],
    "CoTra":           [1560, 1140, 1980, 1380, 6840, 10800],
    "DistributedANN*": [1204, 924, 994, 685, 10442, None],
}
TICKS = [(60, "1 min"), (600, "10 min"), (3600, "1 h"), (36000, "10 h"), (86400, "1 day"), (172800, "2 days")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 17, "axes.labelsize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 15, "legend.fontsize": 15,
    "axes.edgecolor": INK2, "axes.linewidth": 1.0, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
})


def main():
    fig, ax = plt.subplots(figsize=(7, 5))  # one column: include at \columnwidth
    systems = list(BUILD)
    width = 0.7 / len(systems)
    for j, s in enumerate(systems):
        for i, v in enumerate(BUILD[s]):
            x = i - 0.35 + width * (j + 0.5)
            if v is None:
                ax.text(x, 45, "n/a", rotation=90, ha="center", va="bottom", fontsize=11, color=INK2)
                continue
            ax.bar(x, v, width=width, color=STYLE[s]["color"], edgecolor="white", linewidth=1.0, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(40, 3.2e5)
    ax.yaxis.set_major_locator(FixedLocator([t for t, _ in TICKS]))
    ax.set_yticklabels([l for _, l in TICKS])
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(SETUPS)))
    ax.set_xticklabels(SETUPS)
    ax.set_xlim(-0.5, len(SETUPS) - 0.5)
    for i in range(1, len(SETUPS)):
        ax.axvline(i - 0.5, color=GRID, lw=0.9)
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.9, ls="--")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("Build time (log scale)")
    handles = [Patch(facecolor=STYLE[s]["color"], edgecolor="white", label=s) for s in systems]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = Path(__file__).resolve().parent / "final_load_comm"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "build_time.png", dpi=200)
    print(f"wrote {out / 'build_time.png'}")


if __name__ == "__main__":
    main()
