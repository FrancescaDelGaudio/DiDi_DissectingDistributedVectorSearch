#!/usr/bin/env python3
"""Recall/QPS curves as small subfigures, four per row of a two-column page (DEEP/TTI setups).

Same data, Pareto fronts, colours and markers as plot_final_curves.py, but every panel is drawn at
its printed size (~1.7 in wide, fonts in pt are the final ones) without title or legend: the setup goes
in the LaTeX subcaption and a single legend file (legend_curves.pdf) is placed above the figure.

Output: subfigs/curve_<setup>.pdf/png and subfigs/legend_curves.pdf/png

Usage:
    python3 plot_curves_subfigs.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_final_curves import FIGS, GRID, INK, ORDER, STYLE, load, pareto  # noqa: E402

OUT = Path(__file__).resolve().parent / "subfigs"
W, H = 1.72, 1.45  # inches: a quarter of a 7 in text width
# "--three": three panels per row (DEEP row, TTI row), slightly larger, files curve3_*.pdf
THREE = "--three" in sys.argv
PREFIX = "curve3" if THREE else "curve"
if THREE:
    W, H = 2.25, 1.7
# TTI10M panels: x axis from 0.7 so that the high end of DistributedANN* (max 0.764 / 0.774) is visible;
# Harmony (max 0.404 / 0.513) stays out of range and is mentioned in the caption.
OVERRIDE = {"tti10m_4n": (["GP-ANN*", "Pyramid*", "CoTra", "DistributedANN*"], [0.70, 0.80, 0.90, 0.95, 0.99], (0.69, 1.0)),
            "tti10m_6n": (["GP-ANN*", "Pyramid*", "CoTra", "DistributedANN*"], [0.70, 0.80, 0.90, 0.95, 0.99], (0.69, 1.0))}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 8 if THREE else 7.5,
    "xtick.labelsize": 7.5 if THREE else 6.5, "ytick.labelsize": 7.5 if THREE else 6.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42,
})


def panel(name, setup, systems, targets, xr, pts):
    fig, ax = plt.subplots(figsize=(W, H))
    for t in targets:
        ax.axvline(t, color="#7a7974", lw=0.7, ls=(0, (3, 2)), zorder=1)
    for s in systems:
        data = pts.get((setup, s), [])
        if not data:
            continue
        front = pareto(data)
        st = STYLE[s]
        ax.plot([p[0] for p in front], [p[1] for p in front], color=st["color"], lw=1.1,
                marker=st["marker"], ms=3.4, mec="white", mew=0.4, zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    step = 0.1 if xr[1] - xr[0] <= 0.35 else 0.2
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_minor_locator(MultipleLocator(step / 2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlim(*xr)
    ax.grid(True, which="major", color=GRID, lw=0.4, ls="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.6, pad=1.5)
    ax.tick_params(axis="both", which="minor", length=1.5, width=0.4)
    ax.set_xlabel("Recall@10", labelpad=1)
    ax.set_ylabel("QPS (log scale)", labelpad=1)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{PREFIX}_{name}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(OUT / f"{PREFIX}_{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def legend():
    handles = [Line2D([], [], color=STYLE[s]["color"], marker=STYLE[s]["marker"], lw=1.1, ms=4,
                      mec="white", mew=0.4, label=s) for s in ORDER]
    fig = plt.figure(figsize=(7.0, 0.22))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, fontsize=8,
               handlelength=2.2, columnspacing=1.6)
    fig.savefig(OUT / "legend_curves.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / "legend_curves.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main():
    pts = load()
    for name, setup, _title, systems, targets, xr in FIGS:
        if THREE and name.endswith("_low"):
            continue
        systems, targets, xr = OVERRIDE.get(name, (systems, targets, xr))
        if not name.endswith("_low"):
            xr = (0.69, xr[1])  # every main panel starts at 0.7, as the TTI10M ones
        panel(name, setup, systems, targets, xr, pts)
    legend()
    print(f"wrote {len(FIGS)} curve panels + legend_curves to {OUT}")


if __name__ == "__main__":
    main()
