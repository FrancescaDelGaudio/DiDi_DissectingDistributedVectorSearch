#!/usr/bin/env python3
"""Index construction time vs number of nodes: one panel per dataset (6 panels), one line per system
with a point at 4, 6 and 8 nodes. Same colours / markers as the recall-QPS curves.

Data: BUILD of plot_build_time_panels.py (sources in its docstring), with the Pyramid* values replaced
by the corrected ones (phase B rebuilt 2026-11-18/19 with all 128 cores, jobs in
exp_eval/pyramid_fix/rebuild_b_jobs_2026-11-18.txt; phase A unchanged):
  total = phase A + phase B (s). DEEP10M/TTI10M 4/6 nodes: June builds, already valid
  (A 4826720/5298968/5430684/5430685, B 5459673/5459675/5474744/5474745; the bar plot used 9960/13920/420/400).
OOM points (MSMARCO100M: GP-ANN*, Pyramid*, Harmony) are not drawn (said in the caption).

Output (subfigs/): buildline_<dataset>.pdf/png (6 panels, one row = 7 in; DEEP10M has the y title), legend_buildline.pdf/png, buildline_all.pdf/png (preview)
Usage:
    module load python/3.12.2
    python3 plot_build_time_lines.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_final_curves import STYLE, INK, INK2, GRID  # noqa: E402
from plot_build_time_panels import BUILD, OOM, SYSTEMS  # noqa: E402

OUT = HERE / "subfigs"
TEXTWIDTH = 7.0
NODES = [4, 6, 8]
# Pyramid* corrected totals (phase A + phase B), seconds
PYR_FIXED = {
    "DEEP10M": {4: 9848 + 123, 6: 13834 + 93, 8: 9848 + 36 + 92},
    "DEEP100M": {4: 160141 + 392 + 1003, 6: 160141 + 387 + 786, 8: 160141 + 530},
    "TTI10M": {4: 297 + 129, 6: 310 + 92, 8: 270 + 96},
    "TTI100M": {4: 2405 + 1050, 6: 2241 + 882, 8: 2292 + 741},
    "MSMARCO10M": {4: 748 + 649, 6: 706 + 486, 8: 725 + 437},
}
DATASETS = ["DEEP10M", "DEEP100M", "TTI10M", "TTI100M", "MSMARCO10M", "MSMARCO100M"]
TICKS = [(60, "1 min"), (600, "10 min"), (3600, "1 h"), (36000, "10 h"), (172800, "2 d")]
YLIM = (40, 3.2e5)
YLABEL_EXTRA = 0.16  # inches added on the left of the first panel by the y-axis title
W, H = (TEXTWIDTH - YLABEL_EXTRA) / 6, 1.35  # six panels + one y title = TEXTWIDTH
L, R, B, T = 0.40, 0.05, 0.27, 0.05

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
})


def value(dataset, system, n):
    if system == "Pyramid*" and dataset in PYR_FIXED:
        return PYR_FIXED[dataset][n]
    return BUILD[dataset][n][SYSTEMS.index(system)]


def draw(ax, dataset, ylabel):
    oom = []
    for s in SYSTEMS:
        ys = [value(dataset, s, n) for n in NODES]
        if any(isinstance(y, str) for y in ys):
            oom.append(s)
            continue
        st = STYLE[s]
        ax.plot(NODES, ys, color=st["color"], marker=st["marker"], ms=3.2, lw=1.1,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.yaxis.set_major_locator(FixedLocator([t for t, _ in TICKS]))
    ax.set_yticklabels([lab for _, lab in TICKS])
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(3.4, 8.6)
    ax.set_xticks(NODES)
    ax.set_xlabel("Nodes", labelpad=1)
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.5, ls="--")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, width=0.6, pad=1.5)
    if ylabel:
        ax.set_ylabel("Build time", labelpad=1.5)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def handles():
    return [Line2D([], [], color=STYLE[s]["color"], marker=STYLE[s]["marker"], ms=4, lw=1.1,
                   markeredgecolor="white", markeredgewidth=0.4, label=s) for s in SYSTEMS]


def main():
    # separate panels, same plot area; the first one (DEEP10M) carries the y-axis title
    for i, d in enumerate(DATASETS):
        extra = YLABEL_EXTRA if i == 0 else 0.0
        fw = W + extra
        fig = plt.figure(figsize=(fw, H))
        ax = fig.add_axes([(L + extra) / fw, B / H, (W - L - R) / fw, (H - B - T) / H])
        draw(ax, d, ylabel=(i == 0))
        save(fig, f"buildline_{d.lower()}")
    fig = plt.figure(figsize=(TEXTWIDTH, 0.25))
    fig.legend(handles=handles(), loc="center", ncol=len(SYSTEMS), frameon=False, handlelength=2.0,
               columnspacing=1.4)
    save(fig, "legend_buildline")
    # preview: the 6 panels in one row with titles and legend
    fig, axs = plt.subplots(1, 6, figsize=(TEXTWIDTH + 0.3, 1.9), sharey=True)
    for i, (ax, d) in enumerate(zip(axs, DATASETS)):
        draw(ax, d, ylabel=(i == 0))
        ax.set_title(d, fontsize=7, pad=3)
    fig.legend(handles=handles(), loc="upper center", ncol=len(SYSTEMS), frameon=False, fontsize=6.5,
               handlelength=2.0, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.78, wspace=0.12)
    save(fig, "buildline_all")
    for d in DATASETS:
        print(d, {s: [value(d, s, n) for n in NODES] for s in SYSTEMS})
    print(f"wrote buildline_* to {OUT}")


if __name__ == "__main__":
    main()
