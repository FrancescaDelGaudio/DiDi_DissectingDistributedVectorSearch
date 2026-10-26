#!/usr/bin/env python3
"""Load-balancing and communication figures from final_load_comm.csv (load_comm_tables.py).

Same 8 panels as the recall/QPS figures (plot_final_curves.py) and the same colour/marker
per system. Every value is taken at the operating point of each recall target.
  load_imbalance.png : grouped bars, extra work on the busiest node = (L_max/L_mean - 1) x 100
  comm_volume.png    : dot plot, KB per query on a log axis (values span ~4 orders of magnitude)
"n/r" marks a target the system does not reach.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_final_curves import STYLE, INK, INK2, GRID  # noqa: E402  same colours/markers as the curves

HERE = Path(__file__).resolve().parent
OUT = HERE / "final_load_comm"

HIGH = ["GP-ANN*", "Pyramid*", "CoTra"]
ALL = ["GP-ANN*", "Pyramid*", "Harmony", "CoTra", "DistributedANN*"]
LOW = ["Harmony", "DistributedANN*"]
PANELS = [  # setup, title, systems
    ("DEEP10M-4n", "DEEP10M · 4 nodes", ALL),
    ("DEEP10M-6n", "DEEP10M · 6 nodes", ALL),
    ("TTI10M-4n", "TTI10M · 4 nodes", HIGH),
    ("TTI10M-6n", "TTI10M · 6 nodes", HIGH),
    ("TTI10M-4n", "TTI10M · 4 nodes\n(low-recall systems)", LOW),
    ("TTI10M-6n", "TTI10M · 6 nodes\n(low-recall systems)", LOW),
    ("DEEP100M-8n", "DEEP100M · 8 nodes", ALL),
    ("TTI100M-8n", "TTI100M · 8 nodes", HIGH),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 20, "axes.titlesize": 21, "axes.labelsize": 21,
    "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 22,
    "axes.edgecolor": INK2, "axes.linewidth": 1.0, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
})


def load():
    data = {}
    for r in csv.DictReader(open(HERE / "final_load_comm.csv")):
        data[(r["setup"], r["system"], float(r["target"]))] = r
    return data


def targets_of(data, setup, systems):
    return sorted({t for (s, sy, t) in data if s == setup and sy in systems})


def style_axis(ax, title):
    ax.set_title(title, pad=10)
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.9, ls="--")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend(fig, kind):
    if kind == "bar":
        handles = [Patch(facecolor=STYLE[s]["color"], edgecolor="white", label=s) for s in ALL]
    else:
        handles = [Line2D([], [], color=STYLE[s]["color"], marker=STYLE[s]["marker"], ms=13, lw=2.0,
                          mec="white", mew=0.8, label=s) for s in ALL]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
               handlelength=1.6, columnspacing=1.2, handletextpad=0.5)


def plot_load(data):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), sharey=True)
    ymax = max((float(r["imbalance"]) - 1) * 100 for r in data.values() if r["imbalance"])
    for ax, (setup, title, systems) in zip(axes.flat, PANELS):
        ts = targets_of(data, setup, systems)
        width = 0.8 / len(systems)
        for j, s in enumerate(systems):
            for i, t in enumerate(ts):
                x = i - 0.4 + width * (j + 0.5)
                r = data.get((setup, s, t))
                if r and r["imbalance"]:
                    v = (float(r["imbalance"]) - 1) * 100
                    ax.bar(x, v, width=width, color=STYLE[s]["color"], edgecolor="white", linewidth=1.0, zorder=3)
                else:
                    ax.text(x, ymax * 0.015, "n/r", rotation=90, ha="center", va="bottom", fontsize=16,
                            color=INK2)
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([f"{t:.2f}" for t in ts])
        ax.set_ylim(0, ymax * 1.08)
        style_axis(ax, title)
    for ax in axes[:, 0]:
        ax.set_ylabel("Extra work (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("Recall@10 target (≥)")
    legend(fig, "bar")
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.6, w_pad=0.8)
    fig.savefig(OUT / "load_imbalance.png", dpi=200)
    plt.close(fig)


def plot_comm(data):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), sharey=True)
    for ax, (setup, title, systems) in zip(axes.flat, PANELS):
        ts = targets_of(data, setup, systems)
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
                    ax.text(x, 0.3, "n/r", rotation=90, ha="center", va="bottom", fontsize=16, color=INK2)
            ax.plot(xs, ys, color=st["color"], lw=1.4, alpha=0.6, zorder=2)
            ax.plot(xs, ys, ls="none", marker=st["marker"], ms=12, color=st["color"], mec="white", mew=0.8,
                    zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(0.25, 8000)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{v / 1024:g} MB" if v >= 1024 else f"{v:g} KB"))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([f"{t:.2f}" for t in ts])
        ax.set_xlim(-0.5, len(ts) - 0.5)
        style_axis(ax, title)
    for ax in axes[:, 0]:
        ax.set_ylabel("Data per query")
    for ax in axes[1, :]:
        ax.set_xlabel("Recall@10 target (≥)")
    legend(fig, "dot")
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.6, w_pad=0.8)
    fig.savefig(OUT / "comm_volume.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data = load()
    plot_load(data)
    plot_comm(data)
    print(f"wrote {OUT}/load_imbalance.png and comm_volume.png")


if __name__ == "__main__":
    main()
