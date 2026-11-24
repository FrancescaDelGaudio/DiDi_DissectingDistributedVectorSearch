#!/usr/bin/env python3
"""Index construction time of the five systems on 4, 6 and 8 nodes: one subfigure per node count,
datasets on x, systems as bars, log scale (same style and colours as plot_build_time.py).

Seconds. Old setups (10M on 4/6 nodes, 100M on 8 nodes): values of plot_build_time.py.
New setups: index campaign of 2026-10-23 (jobs in exp_eval/index_campaign_2026-10-23_jobs.txt):
  GP-ANN*   Partition + QueryAttribution (job elapsed) + per-shard HNSW construction, which runs at
            the start of DistributedBench: first task of the curve sweep minus the same task without
            construction (~125 s on 100M, ~15 s on 10M, as in plot_build_time.py).
            DEEP10M 8: 44 + 1528 + (112 - 15); DEEP100M 4: 558 + 9145 + (2189 - 125);
            TTI100M 4: 800 + 10842 + (2828 - 125). Pending (sweep not run yet): TTI10M 8, 100M on 6.
  Pyramid*  k-means + shard partition + per-shard construction. The new L2 setups reuse the k-means of
            an earlier run (DEEP10M: 4-shard run 4826720, 9848 s; DEEP100M: 8-shard run 6899480,
            160141 s) and redo only the partition (recover) and construction; the reused time is added.
            MIPS (TTI) setups run everything from scratch (single_machine_mips + build).
  Harmony   independent of the node count (same index): train + add; TTI100M job 6977564: 102.5 + 1278.6.
  CoTra     elapsed of the index job (exit 134 at teardown is normal); DEEP100M 6p: job 6975571.
  DA*       elapsed of the index job.

Output (subfigs/): build_time_{4,6,8}n.pdf/png, legend_build_time.pdf/png
Usage:
    module load anaconda3/2023.09
    python3 plot_build_time_scaling.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_final_curves import STYLE, INK, INK2, GRID  # noqa: E402  same colours as the curves

OUT = Path(__file__).resolve().parent / "subfigs"
TEXTWIDTH = 7.0
W, H = TEXTWIDTH / 3 - 0.05, 2.3
DATASETS = ["DEEP\n10M", "TTI\n10M", "DEEP\n100M", "TTI\n100M"]
PENDING = "pending"
# nodes -> system -> [DEEP10M, TTI10M, DEEP100M, TTI100M] (seconds)
BUILD = {
    4: {"GP-ANN*":         [2209 + 186, 1909 + 202, 558 + 9145 + 2064, 800 + 10842 + 2703],
        "Pyramid*":        [9960, 420, 160141 + 392 + 12141, 2405 + 15427],
        "Harmony":         [59.2, 111.0, 558.94, 1381.1],
        "CoTra":           [1560, 1980, 8995, 15172],
        "DistributedANN*": [1204, 994, 19177, 29778]},
    6: {"GP-ANN*":         [1640 + 124, 2001 + 136, PENDING, PENDING],
        "Pyramid*":        [13920, 400, 160141 + 387 + 8788, 2241 + 8637],
        "Harmony":         [59.2, 111.0, 558.94, 1381.1],
        "CoTra":           [1140, 1380, 7111, 11034],
        "DistributedANN*": [924, 685, 13680, 22860]},
    8: {"GP-ANN*":         [44 + 1528 + 97, PENDING, 9641 + 1086, 12363 + 1160],
        "Pyramid*":        [9848 + 36 + 393, 270 + 674, 178920, 24660],
        "Harmony":         [59.2, 111.0, 558.94, 1381.1],
        "CoTra":           [703, 953, 6840, 10800],
        "DistributedANN*": [708, 578, 10442, 18173]},
}
SYSTEMS = ["GP-ANN*", "Pyramid*", "Harmony", "CoTra", "DistributedANN*"]
TICKS = [(60, "1 min"), (600, "10 min"), (3600, "1 h"), (36000, "10 h"), (86400, "1 day"), (172800, "2 days")]
YLIM = (40, 3.2e5)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel(nodes):
    fig, ax = plt.subplots(figsize=(W, H))
    width = 0.8 / len(SYSTEMS)
    for j, s in enumerate(SYSTEMS):
        for i, v in enumerate(BUILD[nodes][s]):
            x = i - 0.4 + width * (j + 0.5)
            if v == PENDING:
                ax.text(x, YLIM[0] * 1.15, "pending", rotation=90, ha="center", va="bottom", fontsize=5.5, color=INK2)
                continue
            ax.bar(x, v, width=width, color=STYLE[s]["color"], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.yaxis.set_major_locator(FixedLocator([t for t, _ in TICKS]))
    ax.set_yticklabels([l for _, l in TICKS])
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_xlim(-0.5, len(DATASETS) - 0.5)
    for i in range(1, len(DATASETS)):
        ax.axvline(i - 0.5, color=GRID, lw=0.6)
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.6, ls="--")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=2, width=0.6, pad=1.5)
    ax.set_ylabel("Build time (log scale)", labelpad=2)
    save(fig, f"build_time_{nodes}n")


def legend():
    handles = [Patch(facecolor=STYLE[s]["color"], edgecolor="white", label=s) for s in SYSTEMS]
    fig = plt.figure(figsize=(TEXTWIDTH, 0.25))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, handlelength=1.4, columnspacing=1.4)
    save(fig, "legend_build_time")


def main():
    for nodes in BUILD:
        panel(nodes)
    legend()
    print(f"wrote build_time_{{4,6,8}}n and legend_build_time to {OUT}")


if __name__ == "__main__":
    main()
