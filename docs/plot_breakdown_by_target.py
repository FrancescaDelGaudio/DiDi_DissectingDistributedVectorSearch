#!/usr/bin/env python3
"""Runtime breakdown by recall target (DEEP100M, 8 nodes): one panel per target, systems on x.

Panels: 0.80, 0.90 and the highest target of each system (0.99, 0.93 for DistributedANN*).
Categories, phase mapping and data are those of plot_breakdown_vertical.py / plot_breakdown.py.
Under each system name the measured recall of the bar is printed, so bars measured slightly
below the target (CoTra, Harmony and DistributedANN* at 0.80) are visible without footnotes.

Usage:
    python3 plot_breakdown_by_target.py   # writes breakdown_by_target.png/.pdf
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

TARGETS = [("0.80", "Recall@10 ≥ 0.80"), ("0.90", "Recall@10 ≥ 0.90"), ("max", "Highest target")]
SHORT = {"GP-ANN": "GP-ANN*", "Pyramid": "Pyramid*", "Harmony": "Harmony", "CoTra": "CoTra",
         "DistributedANN": "Distr.ANN*"}


def pick(runs, target):
    """Run of a system for a panel; 'max' = its last (highest) target."""
    if target == "max":
        return runs[-1]
    for run in runs:
        if run[0].rstrip("*†") == target:
            return run
    return None


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK_2, "hatch.color": "#ffffff", "hatch.linewidth": 0.8,
    })
    data = dict(SYSTEMS)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(7.0, 3.0), sharey=True)
    used = set()
    for ax, (target, title) in zip(axes, TARGETS):
        ticks = []
        for x, (key, _) in enumerate(PANELS):
            run = pick(data[key], target)
            if run is None:
                ticks.append(f"{SHORT[key]}\nn/a")
                continue
            tgt, recall, params, qps, shares = run
            cats = collapse(key, shares)
            bottom = 0.0
            for c in CATS:
                h = cats.get(c, 0.0)
                if h <= 0:
                    continue
                used.add(c)
                _, colour, hatch = CATS[c]
                ax.bar(x, h, bottom=bottom, width=0.7, color=colour, hatch=hatch,
                       edgecolor=SURFACE, linewidth=0.8)
                if h >= 8:
                    chip = dict(boxstyle="round,pad=0.12", facecolor=colour, edgecolor="none") if hatch else None
                    ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=7,
                            color=ink_on(colour), bbox=chip)
                bottom += h
            label = (f"{qps/1000:.0f}k" if qps >= 10000 else f"{qps/1000:.1f}k" if qps >= 1000
                     else f"{qps:.0f}")
            ax.text(x, 101.5, label, ha="center", va="bottom", fontsize=6.5, color=INK_2)
            ticks.append(f"{SHORT[key]}\n({recall:.3f})")
        ax.set_xticks(range(len(PANELS)))
        ax.set_xticklabels(ticks, fontsize=6.5, rotation=90)
        ax.set_xlim(-0.6, len(PANELS) - 0.4)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=9, pad=13, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of query-phase runtime (%)")
    handles = [Patch(facecolor=CATS[c][1], hatch=CATS[c][2], edgecolor=SURFACE, label=CATS[c][0])
               for c in CATS if c in used]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=0.8)
    out = Path(__file__).resolve().parent
    fig.savefig(out / "breakdown_by_target.pdf")
    fig.savefig(out / "breakdown_by_target.png", dpi=250)
    print(f"wrote {out / 'breakdown_by_target.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
