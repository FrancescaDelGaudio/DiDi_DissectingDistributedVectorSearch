#!/usr/bin/env python3
"""Per-system breakdown subfigures with the communication split into transfer, protocol and
waiting (DEEP100M, 8 nodes). Same size, fonts and bar width as plot_subfigs.py's
breakdown_system_*, so they can sit in the same row as the other systems' panels.

Output (subfigs/):
  breakdown6_system_{gpann,harmony,dann}.pdf/png   recall targets on x, no title (LaTeX subcaption)
  legend_breakdown6.pdf/png                         one legend for the row

Data and categories: plot_breakdown6.py.

Usage:
    module load anaconda3/2023.09
    python3 plot_subfigs6.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_breakdown6 import CATS, SYSTEMS, INK, INK_2, SURFACE, ink_on  # noqa: E402

OUT = Path(__file__).resolve().parent / "subfigs"
TEXTWIDTH = 7.0          # inches, two-column page (as plot_subfigs.py)
H = 2.3                  # panel height, inches (as plot_subfigs.py)
LABEL_MIN = 6.5          # label a segment when it takes at least this share
SLUG = {"GP-ANN*": "gpann", "Harmony": "harmony", "DistributedANN*": "dann"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK_2, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5, "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def qps_label(qps):
    return f"{qps/1000:.0f}k" if qps >= 10000 else f"{qps/1000:.1f}k" if qps >= 1000 else f"{qps:.0f}"


def panel(title, runs):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH / 5 - 0.06, H))
    for x, (target, recall, qps, shares) in enumerate(runs):
        bottom = 0.0
        for c, (_, colour) in CATS.items():
            h = shares.get(c, 0.0)
            if h <= 0:
                continue
            ax.bar(x, h, bottom=bottom, width=0.72, color=colour, edgecolor=SURFACE, linewidth=0.6)
            if h >= LABEL_MIN:
                ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=6.5, color=ink_on(colour))
            bottom += h
        ax.text(x, 101.5, qps_label(qps), ha="center", va="bottom", fontsize=6, color=INK_2)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([r[0] for r in runs], fontsize=6.5)
    ax.set_xlim(-0.6, len(runs) - 0.4)
    ax.set_ylim(0, 100)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=2, width=0.6, pad=1.5)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("% of query-phase runtime", labelpad=2)
    save(fig, f"breakdown6_system_{SLUG[title]}")


def legend():
    handles = [Patch(facecolor=colour, edgecolor=SURFACE, label=label) for label, colour in CATS.values()]
    fig = plt.figure(figsize=(TEXTWIDTH, 0.25))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, fontsize=8,
               handlelength=1.8, columnspacing=1.4)
    save(fig, "legend_breakdown6")


def main():
    for title, runs in SYSTEMS:
        panel(title, runs)
    legend()
    print(f"wrote breakdown6_system_*.pdf/png and legend_breakdown6.pdf/png to {OUT}")


if __name__ == "__main__":
    main()
