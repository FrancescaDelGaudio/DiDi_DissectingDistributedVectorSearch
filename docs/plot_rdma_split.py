#!/usr/bin/env python3
"""Communication split of a system over MPI and over native RDMA, with the method of the comm zoom
(waiting / protocol / transfer, % of the MPI query time), single pass. Data: rdma_split.csv
(collect_rdma_split.py). One half-column panel per system, same size and style as
plot_gp_rdma_comm_replay.py: per recall target a stacked bar for MPI (left) and RDMA (right).
Usage: python3 plot_rdma_split.py "GP-ANN*" gp
Writes subfigs/<slug>_rdma_split.pdf/.png
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_gp_mpi_vs_rdma import INK, SURFACE, clean, style

HERE = Path(__file__).resolve().parent
TARGETS = ["0.80", "0.90", "0.95", "0.99"]
CATS = [("transfer", "Transfer", "#7a7974"), ("protocol", "Protocol", "#eda100"), ("waiting", "Waiting", "#6250d6")]


def main():
    system, slug = sys.argv[1], sys.argv[2]
    rows = {(r["target"], r["variant"]): r for r in csv.DictReader(open(HERE / "rdma_split.csv")) if r["system"] == system}
    style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(1.65, 2.2), constrained_layout=True)
    w, gap = 0.40, 0.04
    top, ticks = 0, []
    for j, t in enumerate(TARGETS):
        for k, v in enumerate(("MPI", "RDMA")):
            r = rows[(t, v)]
            x = j + (k - 0.5) * (w + gap)
            bottom = 0.0
            for key, _label, colour in CATS:
                h = max(0.0, float(r[f"{key}_pct_of_T_MPI"]))
                ax.bar(x, h, bottom=bottom, width=w, color=colour, edgecolor=SURFACE, linewidth=0.4)
                bottom += h
            top = max(top, bottom)
            ticks.append((x, v[0]))                     # "M" / "R" under each bar
        ax.text(j, -0.085, t[1:], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5)
    ax.set_xticks([x for x, _ in ticks], [l for _, l in ticks])
    ax.tick_params(axis="x", length=0, pad=1.5, labelsize=6)
    ax.set_xlabel("Recall@10", labelpad=12)
    ax.set_ylabel("% of MPI query time")
    ax.set_ylim(0, top * 1.45)
    ax.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in reversed(CATS)], frameon=False, loc="upper center",
              ncol=2, handlelength=0.9, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.1)
    clean(ax)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "subfigs" / f"{slug}_rdma_split.{ext}", dpi=200)


if __name__ == "__main__":
    main()
