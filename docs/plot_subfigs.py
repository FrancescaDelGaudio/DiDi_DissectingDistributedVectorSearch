#!/usr/bin/env python3
"""Breakdown and RDMA-gain panels as separate subfigures (DEEP100M, 8 nodes).

Every panel is drawn at its printed size (fonts in pt are the final ones), so it stays sharp when
included at that width; PDFs are vector, PNGs are 600 dpi. Legends are separate files, to be placed
once above each row of subfigures.

Output (subfigs/):
  breakdown_target_{080,090,max}.pdf/png        3 per row, systems on x
  breakdown_system_{gpann,pyramid,harmony,cotra,dann}.pdf/png   5 per row, recall targets on x
  rdma_target_{080,090,max}.pdf/png              3 per row, systems on x
  rdma_system_{gpann,dann,harmony,pyramid}.pdf/png   recall targets on x
  legend_breakdown.pdf/png, legend_rdma.pdf/png

Data and phase mapping: plot_breakdown.py, plot_breakdown_vertical.py, plot_rdma_by_target.py.

Usage:
    python3 plot_subfigs.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_breakdown import SYSTEMS, INK, INK_2, SURFACE, ink_on  # noqa: E402
from plot_breakdown_vertical import CATS, PANELS, collapse  # noqa: E402
from plot_rdma_by_target import SEGMENTS as RDMA_SEG, DATA as RDMA, gain  # noqa: E402

OUT = Path(__file__).resolve().parent / "subfigs"
TEXTWIDTH = 7.0          # inches, two-column page
H = 2.3                  # height of every panel, inches
SHORT = {"GP-ANN": "GP-ANN*", "Pyramid": "Pyramid*", "Harmony": "Harmony", "CoTra": "CoTra",
         "DistributedANN": "Distr.ANN*"}
SLUG = {"GP-ANN": "gpann", "Pyramid": "pyramid", "Harmony": "harmony", "CoTra": "cotra",
        "DistributedANN": "dann", "GP-ANN*": "gpann", "Distr.ANN*": "dann", "Pyramid*": "pyramid"}
# Pyramid* (job 6977296), same cell layout as plot_rdma_by_target.DATA: (target, communication share,
# wire, sw_lo, sw_hi), % of T. sw = C_replay - wire, with C_replay = the sender's MPI_Send time in the
# PYR_COMM_ONLY runs (fastest / median of 3); communication share as in plot_subfigs_comm.py.
# Only in rdma_system_pyramid (the rdma_target_* panels keep the three systems of DATA).
PYRAMID_RDMA = {"0.80": ("0.80", 90.69, 0.0632, 44.54, 45.80), "0.90": ("0.90", 72.83, 0.0409, 34.93, 35.03),
                "max": ("0.99", 33.18, 0.0086, 3.62, 3.97)}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK_2, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "hatch.color": "#ffffff", "hatch.linewidth": 0.6, "pdf.fonttype": 42,
})


def style(ax, ymax, title):
    ax.set_ylim(0, ymax)
    if title:  # per-system panels carry no title: the system name goes in the LaTeX subcaption
        ax.set_title(title, pad=11, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=2, width=0.6, pad=1.5)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("% of query-phase runtime", labelpad=2)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def qps_label(qps):
    return f"{qps/1000:.0f}k" if qps >= 10000 else f"{qps/1000:.1f}k" if qps >= 1000 else f"{qps:.0f}"


def stacked_breakdown(ax, x, key, run, width):
    _, recall, _, qps, shares = run
    cats = collapse(key, shares)
    bottom = 0.0
    for c in CATS:
        h = cats.get(c, 0.0)
        if h <= 0:
            continue
        _, colour, hatch = CATS[c]
        ax.bar(x, h, bottom=bottom, width=width, color=colour, hatch=hatch, edgecolor=SURFACE, linewidth=0.6)
        if h >= 10:
            chip = dict(boxstyle="round,pad=0.1", facecolor=colour, edgecolor="none") if hatch else None
            ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=6.5,
                    color=ink_on(colour), bbox=chip)
        bottom += h
    ax.text(x, 101.5, qps_label(qps), ha="center", va="bottom", fontsize=6, color=INK_2)


def pick(runs, target):
    if target == "max":
        return runs[-1]
    return next((r for r in runs if r[0].rstrip("*†") == target), None)


def breakdown_by_target():
    data = dict(SYSTEMS)
    for target, title, slug in [("0.80", "Recall@10 ≥ 0.80", "080"), ("0.90", "Recall@10 ≥ 0.90", "090"),
                                ("max", "Highest target", "max")]:
        fig, ax = plt.subplots(figsize=(TEXTWIDTH / 3 - 0.08, H))
        ticks = []
        for x, (key, _) in enumerate(PANELS):
            run = pick(data[key], target)
            stacked_breakdown(ax, x, key, run, 0.72)
            ticks.append(f"{SHORT[key]}\n({run[1]:.3f})")
        ax.set_xticks(range(len(PANELS)))
        ax.set_xticklabels(ticks, rotation=90, fontsize=7)
        ax.set_xlim(-0.6, len(PANELS) - 0.4)
        style(ax, 100, None)
        save(fig, f"breakdown_target_{slug}")


def breakdown_by_system():
    data = dict(SYSTEMS)
    for key, title in PANELS:
        runs = data[key]
        fig, ax = plt.subplots(figsize=(TEXTWIDTH / 5 - 0.06, H))
        for x, run in enumerate(runs):
            stacked_breakdown(ax, x, key, run, 0.72)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([r[0] for r in runs], fontsize=6.5)
        ax.set_xlim(-0.6, len(runs) - 0.4)
        style(ax, 100, None)
        save(fig, f"breakdown_system_{SLUG[key]}")


def rdma_bar(ax, x, cell, width):
    tgt, total, wire, sw_lo, sw_hi = cell
    widths = {"rest": total - wire - sw_hi, "wire": wire, "sw": sw_hi}
    bottom = 0.0
    for key, _, colour, hatch in RDMA_SEG:
        ax.bar(x, widths[key], bottom=bottom, width=width, color=colour, hatch=hatch,
               edgecolor=SURFACE, linewidth=0.6)
        bottom += widths[key]
    lo, hi = f"{gain(sw_lo):.1f}", f"{gain(sw_hi):.1f}"
    ax.text(x, total + 1.0, f"+{lo}%" if lo == hi else f"+{lo}–\n{hi}%", ha="center", va="bottom",
            fontsize=6.5, color=INK, fontweight="bold", linespacing=0.95)


RDMA_SIZE = (1.85, 1.9)  # narrower and lower than the breakdown panels: bars were very wide


def rdma_by_target():
    systems = list(RDMA)
    for target, title, slug in [("0.80", "Recall@10 ≥ 0.80", "080"), ("0.90", "Recall@10 ≥ 0.90", "090"),
                                ("max", "Highest target", "max")]:
        fig, ax = plt.subplots(figsize=RDMA_SIZE)
        ticks = []
        for x, name in enumerate(systems):
            cell = RDMA[name][target]
            rdma_bar(ax, x, cell, 0.6)
            short = {"GP-ANN*": "GP-\nANN*", "Distr.ANN*": "Distr.\nANN*", "Harmony": "Harmony"}[name]
            ticks.append(short if target != "max" else f"{short}\n({cell[0]})")
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels(ticks, fontsize=6.5)
        ax.set_xlim(-0.6, len(systems) - 0.4)
        style(ax, 60, None)
        save(fig, f"rdma_target_{slug}")


def rdma_by_system():
    for name, cells in {**RDMA, "Pyramid*": PYRAMID_RDMA}.items():
        runs = [cells[t] for t in ("0.80", "0.90", "max")]
        fig, ax = plt.subplots(figsize=RDMA_SIZE)
        for x, cell in enumerate(runs):
            rdma_bar(ax, x, cell, 0.6)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([c[0] for c in runs], fontsize=7.5)
        ax.set_xlim(-0.6, len(runs) - 0.4)
        style(ax, 60 if name != "Pyramid*" else 100, None)  # Pyramid*: 91% communication at 0.80
        save(fig, f"rdma_system_{SLUG[name]}")


def legends():
    used = [c for c in CATS]
    for name, handles, ncol in [
        ("legend_breakdown", [Patch(facecolor=CATS[c][1], hatch=CATS[c][2], edgecolor=SURFACE, label=CATS[c][0])
                              for c in used], len(used)),
        ("legend_rdma", [Patch(facecolor=c, hatch=h, edgecolor=SURFACE, label=l) for _, l, c, h in RDMA_SEG], 3),
    ]:
        fig = plt.figure(figsize=(TEXTWIDTH, 0.25))
        fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False, fontsize=8,
                   handlelength=1.8, columnspacing=1.4)
        save(fig, name)


def main():
    breakdown_by_target()
    breakdown_by_system()
    rdma_by_target()
    rdma_by_system()
    legends()
    print(f"wrote {len(list(OUT.glob('*.pdf')))} PDFs (+ PNGs at 600 dpi) to {OUT}")


if __name__ == "__main__":
    main()
