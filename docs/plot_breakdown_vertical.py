#!/usr/bin/env python3
"""Vertical runtime breakdown, one panel per system (recall targets on x), DEEP100M, 8 nodes.

Every system is mapped onto the four categories of the GP-ANN* breakdown (Figure 8):
  Routing        GP: routing index | DA: head search + per-hop routing | Harmony: IVF centroid search
                 Pyramid: routing / pipeline fill | CoTra: query intake
  Local search   GP/DA/Harmony/Pyramid: local search | CoTra: local subqueries + work stealing + remote tasks
  Merge          GP/DA/CoTra: merge | Harmony: none (final sort < 0.1%, counted with the heap exchange)
  MPI + waiting  GP: MPI + waiting | DA: MPI transfer + waiting | Harmony: query distribution + MPI + waiting
                 + heap exchange
                 Pyramid: MPI transfer + shard idle + waiting | CoTra: RDMA polling + idle
DistributedANN*'s unattributed remainder stays as "Other" rather than being folded into a category.
Data and markers (*, dagger) come from plot_breakdown.py.

Usage:
    python3 plot_breakdown_vertical.py   # writes breakdown_vertical_by_system.png/.pdf
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_breakdown import SYSTEMS, INK, INK_2, SURFACE, ink_on  # noqa: E402

CATS = {  # category -> (label, colour, hatch), colours of Figure 8
    "routing": ("Routing",       "#2a78d6", None),
    "search":  ("Local search",  "#1baf7a", None),
    "merge":   ("Merge",         "#008300", None),
    "mpiwait": ("MPI + waiting", "#e87ba4", "////"),
    "other":   ("Other",         "#e2e1dd", None),
}
MAP = {
    "GP-ANN":         {"routing": "routing", "search": "search", "merge": "merge", "mpiwait": "mpiwait"},
    "DistributedANN": {"routing": "routing", "perhop": "routing", "search": "search", "merge": "merge",
                       "mpi": "mpiwait", "wait": "mpiwait", "other": "other"},
    # Harmony "other" = heap exchange over MPI + residual master overhead (final sort < 0.1%)
    "Harmony":        {"routing": "routing", "search": "search", "other": "mpiwait",
                       "qdist": "mpiwait", "mpiwait": "mpiwait"},
    "Pyramid":        {"fill": "routing", "search": "search", "mpi": "mpiwait", "idle": "mpiwait",
                       "wait": "mpiwait"},
    "CoTra":          {"routing": "routing", "search": "search", "steal": "search", "remote": "search",
                       "merge": "merge", "rdma": "mpiwait", "cwait": "mpiwait"},
}
PANELS = [("GP-ANN", "GP-ANN*"), ("Pyramid", "Pyramid*"), ("Harmony", "Harmony"),
          ("CoTra", "CoTra"), ("DistributedANN", "DistributedANN*")]


def collapse(name, shares):
    out = {}
    for ph, v in shares.items():
        c = MAP[name][ph]
        out[c] = out.get(c, 0.0) + v
    return out


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK_2, "hatch.color": "#ffffff", "hatch.linewidth": 0.8,
    })
    data = dict(SYSTEMS)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 2.9), sharey=True,
                             gridspec_kw={"width_ratios": [len(data[k]) for k, _ in PANELS]})
    used = set()
    for ax, (key, title) in zip(axes, PANELS):
        runs = data[key]
        for x, (target, recall, params, qps, shares) in enumerate(runs):
            cats = collapse(key, shares)
            bottom = 0.0
            for c in CATS:
                h = cats.get(c, 0.0)
                if h <= 0:
                    continue
                used.add(c)
                _, colour, hatch = CATS[c]
                ax.bar(x, h, bottom=bottom, width=0.68, color=colour, hatch=hatch,
                       edgecolor=SURFACE, linewidth=0.8)
                if h >= 8:
                    chip = dict(boxstyle="round,pad=0.12", facecolor=colour, edgecolor="none") if hatch else None
                    ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=7,
                            color=ink_on(colour), bbox=chip)
                bottom += h
            label = (f"{qps/1000:.0f}k" if qps >= 10000 else f"{qps/1000:.1f}k" if qps >= 1000
                     else f"{qps:.0f}")
            ax.text(x, 101.5, label, ha="center", va="bottom", fontsize=6.5, color=INK_2)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([t for t, *_ in runs], fontsize=7.5)
        ax.set_xlim(-0.6, len(runs) - 0.4)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=9, pad=13, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Share of query-phase runtime (%)")
    fig.supxlabel("Recall@10 target (queries/s above each bar)", fontsize=8.5, color=INK, y=0.035)
    handles = [Patch(facecolor=CATS[c][1], hatch=CATS[c][2], edgecolor=SURFACE, label=CATS[c][0])
               for c in CATS if c in used]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92), w_pad=0.8)
    out = Path(__file__).resolve().parent
    fig.savefig(out / "breakdown_vertical_by_system.pdf")
    fig.savefig(out / "breakdown_vertical_by_system.png", dpi=250)
    print(f"wrote {out / 'breakdown_vertical_by_system.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
