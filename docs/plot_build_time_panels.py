#!/usr/bin/env python3
"""Index construction time, one panel per setup (dataset x node count), one bar per system.
Same style and colours as plot_build_time_scaling.py. Panels carry no dataset / node-count text
(they go into LaTeX subfigures): six panels fit one row of a two-column paper (7 in).

Seconds of wall-clock time (sources: plot_build_time_scaling.py docstring, plus):
  GP-ANN*   Partition + QueryAttribution (job elapsed) + per-shard HNSW construction at the start of
            DistributedBench = first sweep task minus 15 s (10M) / 125 s (100M) of load and queries.
            TTI10M 8: 47 + 2101 + (256 - 15); DEEP100M 6: 649 + 9889 + (1589 - 125);
            TTI100M 6: 716 + 11606 + (1747 - 125); MSMARCO10M k4: 124 + 12390 + (1047 - 15),
            k6: 134 + 12032 + (781 - 15), k8: 137 + 12844 + (733 - 15) (first tasks of 6995980 / 6995981).
  Pyramid*  MSMARCO10M: step A (symmetrised-graph binary, 6987698/700/702) + phase B (6987699/701/703).
  Harmony   MSMARCO10M (job 6982197): train 46.78 + add 170.44.
  CoTra     MSMARCO: index job elapsed (10M 6987710/11/12; 100M 6987713, 6975572, 6973801).
  DA*       MSMARCO10M: 6987714/15/16. MSMARCO100M: steps 1-4 of 6987717/18/19 (job start to the
            start of step 5, where the int overflow crashed) + resume of steps 5-6 (6994585/86/87).
  OOM       the index does not fit in our largest nodes (MSMARCO100M: GP-ANN*, Pyramid*, Harmony).

Output (subfigs/): build_time_<dataset>_<n>n.pdf/png, the same with _ylabel (y-axis title, for the
first panel of a row), legend_build_time_panels.pdf/png.
Usage:
    module load python/3.12.2
    python3 plot_build_time_panels.py
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
W, H = TEXTWIDTH / 6, 1.35          # inches, panel without y-axis title
# --short: lower panels for the column layout (3 node counts stacked per dataset), files *_s.pdf
SHORT = "--short" in sys.argv
if SHORT:
    H = 0.85
SUF = "_s" if SHORT else ""
YLABEL_EXTRA = 0.16                 # inches added on the left by the y-axis title
L, R, B, T = 0.40, 0.03, 0.05, 0.05  # fixed margins (inches): every panel has the same plot area
OOM, PENDING = "OOM", "pending"
SYSTEMS = ["GP-ANN*", "Pyramid*", "Harmony", "CoTra", "DistributedANN*"]
HMY = {"DEEP10M": 59.2, "TTI10M": 111.0, "DEEP100M": 558.94, "TTI100M": 1381.1, "MSMARCO10M": 46.78 + 170.44}
# dataset -> nodes -> [GP-ANN*, Pyramid*, Harmony, CoTra, DistributedANN*] (seconds)
BUILD = {
    "DEEP10M": {
        4: [2209 + 186, 9960, HMY["DEEP10M"], 1560, 1204],
        6: [1640 + 124, 13920, HMY["DEEP10M"], 1140, 924],
        8: [44 + 1528 + 97, 9848 + 36 + 393, HMY["DEEP10M"], 703, 708]},
    "DEEP100M": {
        4: [558 + 9145 + 2064, 160141 + 392 + 12141, HMY["DEEP100M"], 8995, 19177],
        6: [649 + 9889 + 1464, 160141 + 387 + 8788, HMY["DEEP100M"], 7111, 13680],
        8: [9641 + 1086, 178920, HMY["DEEP100M"], 6840, 10442]},
    "TTI10M": {
        4: [1909 + 202, 420, HMY["TTI10M"], 1980, 994],
        6: [2001 + 136, 400, HMY["TTI10M"], 1380, 685],
        8: [47 + 2101 + 241, 270 + 674, HMY["TTI10M"], 953, 578]},
    "TTI100M": {
        4: [800 + 10842 + 2703, 2405 + 15427, HMY["TTI100M"], 15172, 29778],
        6: [716 + 11606 + 1622, 2241 + 8637, HMY["TTI100M"], 11034, 22860],
        8: [12363 + 1160, 24660, HMY["TTI100M"], 10800, 18173]},
    "MSMARCO10M": {
        4: [124 + 12390 + 1032, 748 + 5670, HMY["MSMARCO10M"], 10742, 3911],
        6: [134 + 12032 + 766, 706 + 3617, HMY["MSMARCO10M"], 8064, 3126],
        8: [137 + 12844 + 718, 725 + 2849, HMY["MSMARCO10M"], 6269, 2810]},
    "MSMARCO100M": {
        4: [OOM, OOM, OOM, 69652, 117700 + 2142],
        6: [OOM, OOM, OOM, 62612, 80207 + 2063],
        8: [OOM, OOM, OOM, 51204, 59722 + 1931]},
}
TICKS = [(60, "1 min"), (600, "10 min"), (3600, "1 h"), (36000, "10 h"), (172800, "2 days")]
YLIM = (40, 3.2e5)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def panel(dataset, nodes, ylabel):
    extra = YLABEL_EXTRA if ylabel else 0.0
    fw = W + extra
    fig = plt.figure(figsize=(fw, H))
    ax = fig.add_axes([(L + extra) / fw, B / H, (W - L - R) / fw, (H - B - T) / H])
    for i, (s, v) in enumerate(zip(SYSTEMS, BUILD[dataset][nodes])):
        if isinstance(v, str):
            ax.text(i, YLIM[0] * 1.25, v, rotation=90, ha="center", va="bottom", fontsize=6.5,
                    color=INK2, fontweight="bold" if v == OOM else "normal")
            continue
        ax.bar(i, v, width=0.82, color=STYLE[s]["color"], edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.yaxis.set_major_locator(FixedLocator([t for t, _ in TICKS]))
    ax.set_yticklabels([lab for _, lab in TICKS])
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(-0.6, len(SYSTEMS) - 0.4)
    ax.set_xticks([])
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.5, ls="--")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=2, width=0.6, pad=1.5)
    if ylabel:
        ax.set_ylabel("Build time", labelpad=1.5)
    save(fig, f"build_time_{dataset}_{nodes}n" + ("_ylabel" if ylabel else "") + SUF)


def legend():
    handles = [Patch(facecolor=STYLE[s]["color"], edgecolor="white", label=s) for s in SYSTEMS]
    fig = plt.figure(figsize=(TEXTWIDTH, 0.25))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, handlelength=1.4, columnspacing=1.4)
    save(fig, "legend_build_time_panels")


def main():
    for dataset, by_nodes in BUILD.items():
        for nodes in by_nodes:
            for ylabel in (False, True):
                panel(dataset, nodes, ylabel)
    legend()
    print(f"wrote {sum(len(v) for v in BUILD.values())} panels (+ _ylabel) and the legend to {OUT}")


if __name__ == "__main__":
    main()
