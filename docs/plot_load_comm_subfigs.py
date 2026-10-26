#!/usr/bin/env python3
"""Load-imbalance and communication-volume panels as separate subfigures, no titles.

Same data, fonts, colours and axes as plot_load_comm.py (load_imbalance.png / comm_volume.png):
each panel is drawn at the size it has in that 2x4 figure, so it looks the same when four panels are
placed in a row of a two-column page. The setup goes in the LaTeX subcaption; one legend file per figure.

Output (subfigs/):
  load_<panel>.pdf/png, comm_<panel>.pdf/png, legend_load.pdf/png, legend_comm.pdf/png
  panel = deep10m_4n, deep10m_6n, deep100m_8n, tti10m_4n, tti10m_6n, tti100m_8n
  (every panel from target 0.80; on TTI10M DistributedANN* is n/r at all targets, max 0.764 / 0.774)

Usage:
    python3 plot_load_comm_subfigs.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_load_comm import ALL, STYLE, INK2, load, targets_of, style_axis  # noqa: E402

OUT = Path(__file__).resolve().parent / "subfigs"
SIZE = (4.4, 2.8)  # fonts of the original 16 x 9.5 in figure; three panels per row of the page
HIGH = ["GP-ANN*", "Pyramid*", "CoTra"]
TTI10 = ["GP-ANN*", "Pyramid*", "CoTra", "DistributedANN*"]  # Harmony does not reach 0.70 on TTI10M
SUB = [  # name, setup, systems, minimum target shown
    ("deep10m_4n", "DEEP10M-4n", ALL, 0.0), ("deep10m_6n", "DEEP10M-6n", ALL, 0.0),
    ("deep100m_8n", "DEEP100M-8n", ALL, 0.0),
    ("tti10m_4n", "TTI10M-4n", TTI10, 0.80), ("tti10m_6n", "TTI10M-6n", TTI10, 0.80),
    ("tti100m_8n", "TTI100M-8n", HIGH, 0.0),
]
XLABEL = "Recall"


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def load_panel(data, setup, systems, tmin, ymax, name):
    fig, ax = plt.subplots(figsize=SIZE)
    ts = [t for t in targets_of(data, setup, systems) if t >= tmin]
    width = 0.8 / len(systems)
    for j, s in enumerate(systems):
        for i, t in enumerate(ts):
            x = i - 0.4 + width * (j + 0.5)
            r = data.get((setup, s, t))
            if r and r["imbalance"]:
                ax.bar(x, (float(r["imbalance"]) - 1) * 100, width=width, color=STYLE[s]["color"],
                       edgecolor="white", linewidth=1.0, zorder=3)
            else:
                ax.text(x, ymax * 0.015, "n/r", rotation=90, ha="center", va="bottom", fontsize=16, color=INK2)
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([f"{t:.2f}" for t in ts])
    ax.set_ylim(0, ymax * 1.08)
    style_axis(ax, "")
    ax.set_ylabel("Extra work (%)")
    ax.set_xlabel(XLABEL)
    save(fig, f"load_{name}")


def comm_panel(data, setup, systems, tmin, name):
    fig, ax = plt.subplots(figsize=SIZE)
    ts = [t for t in targets_of(data, setup, systems) if t >= tmin]
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
        ax.plot(xs, ys, ls="none", marker=st["marker"], ms=12, color=st["color"], mec="white", mew=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.25, 8000)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1024:g} MB" if v >= 1024 else f"{v:g} KB"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([f"{t:.2f}" for t in ts])
    ax.set_xlim(-0.5, len(ts) - 0.5)
    style_axis(ax, "")
    ax.set_ylabel("Data/query")
    ax.set_xlabel(XLABEL)
    save(fig, f"comm_{name}")


def legends():
    for name, handles in [
        ("legend_load", [Patch(facecolor=STYLE[s]["color"], edgecolor="white", label=s) for s in ALL]),
        ("legend_comm", [Line2D([], [], color=STYLE[s]["color"], marker=STYLE[s]["marker"], ms=13, lw=2.0,
                                mec="white", mew=0.8, label=s) for s in ALL]),
    ]:
        fig = plt.figure(figsize=(16, 0.6))
        fig.legend(handles=handles, loc="center", ncol=5, frameon=False, handlelength=1.6,
                   columnspacing=1.2, handletextpad=0.5)
        save(fig, name)


def main():
    data = load()
    ymax = max((float(r["imbalance"]) - 1) * 100 for r in data.values() if r["imbalance"])
    for name, setup, systems, tmin in SUB:
        load_panel(data, setup, systems, tmin, ymax, name)
        comm_panel(data, setup, systems, tmin, name)
    legends()
    print(f"wrote {2 * len(SUB)} panels + 2 legends to {OUT}")


if __name__ == "__main__":
    main()
