#!/usr/bin/env python3
"""Load-imbalance and communication panels for the 3 x 5 grid of the recall/QPS curves (one column per
dataset, one row per node count): DEEP10M, DEEP100M, TTI10M, TTI100M, MSMARCO10M on 4 / 6 / 8 nodes.

Same bars / dots, colours, markers, axes and n/r marks as plot_load_comm_subfigs.py, drawn at the printed
size of a fifth of a 7 in text width, with fixed margins (every panel has the same plot area).
Data: load_comm_all.csv (load_comm_all.py). Legends: legend_load.pdf / legend_comm.pdf.

Output (subfigs/): load5_<setup>.pdf/png, comm5_<setup>.pdf/png
Usage:
    module load python/3.12.2
    python3 plot_load_comm_panels5.py
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_final_curves import GRID, INK, INK2, ORDER, STYLE  # noqa: E402

OUT = HERE / "subfigs"
W, H = 7.0 / 5 - 0.02, 1.22          # inches
L, R, B, T = 0.42, 0.04, 0.29, 0.06  # fixed margins (inches)
SYSTEMS = {"DEEP10M": ORDER, "DEEP100M": ORDER,
           "TTI10M": ["GP-ANN*", "Pyramid*", "CoTra", "DistributedANN*"],  # Harmony < 0.70
           "TTI100M": ["GP-ANN*", "Pyramid*", "CoTra"], "MSMARCO10M": ["GP-ANN*", "Pyramid*", "CoTra"]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.5, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
})


def load():
    data = {}
    for r in csv.DictReader(open(HERE / "load_comm_all.csv")):
        data[(r["setup"], r["system"], float(r["target"]))] = r
    return data


def axes(fig):
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.4, ls="--")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", which="major", length=2, width=0.5, pad=1.2)
    ax.tick_params(axis="both", which="minor", length=1.2, width=0.35)
    return ax


def targets(data, setup, systems):
    return sorted({t for (s, sy, t) in data if s == setup and sy in systems})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def load_panel(data, setup, systems, ymax):
    fig = plt.figure(figsize=(W, H))
    ax = axes(fig)
    ts = targets(data, setup, systems)
    width = 0.8 / len(systems)
    for j, s in enumerate(systems):
        for i, t in enumerate(ts):
            x = i - 0.4 + width * (j + 0.5)
            r = data.get((setup, s, t))
            if r and r["imbalance"]:
                ax.bar(x, (float(r["imbalance"]) - 1) * 100, width=width, color=STYLE[s]["color"],
                       edgecolor="white", linewidth=0.3, zorder=3)
            else:
                ax.text(x, ymax * 0.015, "n/r", rotation=90, ha="center", va="bottom", fontsize=5, color=INK2)
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([f"{t:.2f}" for t in ts])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(ts) - 0.5)
    ax.set_ylim(0, ymax * 1.08)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Extra work (%)", labelpad=1)
    ax.set_xlabel("Recall", labelpad=1)
    save(fig, f"load5_{setup.lower().replace('-', '_')}")


def comm_panel(data, setup, systems):
    fig = plt.figure(figsize=(W, H))
    ax = axes(fig)
    ts = targets(data, setup, systems)
    width = 0.6 / len(systems)
    for j, s in enumerate(systems):
        st = STYLE[s]
        xs, ys = [], []
        for i, t in enumerate(ts):
            x = i - 0.3 + width * (j + 0.5)
            r = data.get((setup, s, t))
            if r and r["bytes_per_query"]:
                xs.append(x)
                ys.append(float(r["bytes_per_query"]) / 1024)
            else:
                ax.text(x, 0.3, "n/r", rotation=90, ha="center", va="bottom", fontsize=5, color=INK2)
        ax.plot(xs, ys, color=st["color"], lw=0.6, alpha=0.6, zorder=2)
        ax.plot(xs, ys, ls="none", marker=st["marker"], ms=3.6, color=st["color"], mec="white", mew=0.3, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.25, 8000)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1024:g} MB" if v >= 1024 else f"{v:g} KB"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([f"{t:.2f}" for t in ts])
    ax.set_xlim(-0.5, len(ts) - 0.5)
    ax.set_ylabel("Data/query", labelpad=1)
    ax.set_xlabel("Recall", labelpad=1)
    save(fig, f"comm5_{setup.lower().replace('-', '_')}")


def main():
    data = load()
    ymax = max((float(r["imbalance"]) - 1) * 100 for r in data.values() if r["imbalance"])
    for ds, systems in SYSTEMS.items():
        for n in (4, 6, 8):
            load_panel(data, f"{ds}-{n}n", systems, ymax)
            comm_panel(data, f"{ds}-{n}n", systems)
    print(f"wrote {2 * 3 * len(SYSTEMS)} panels load5_* / comm5_* to {OUT}")


if __name__ == "__main__":
    main()
