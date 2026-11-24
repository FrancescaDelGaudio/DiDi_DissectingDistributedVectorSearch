#!/usr/bin/env python3
"""Paper panels for the skewed-workload experiment (DEEP100M, 8 nodes): one PDF per system, sized for
five panels per row of a two-column paper (7 in), same style as docs/plot_curves_panels6.py.

Comparison at the SAME CONFIGURATION: every configuration of the uniform curve is placed at its recall
on the uniform workload; dashed = uniform QPS (Pareto front), solid = skewed QPS of the same configs
(Pareto front of its own values, so neither curve is favoured).
  r1c   : skewed = workload r1c_p100 (highest mean imbalance over GP-ANN*, Pyramid*, CoTra)
  median: skewed = median over the 8 workloads (band = min-max); r2 excluded by the difficulty criterion
  pfam  : one region (R1a), growing hot fraction p: uniform (p=0, dashed) + one solid line per p, darker = larger p
Y axis linear by default (advisor's request); --log for the old log-scale panels.

Data: docs/skew_curves.csv (python3 collect_skew.py first)
Output: figs/panels/skew_{r1c,median}_<system>.pdf/png + figs/panels/legend_skew_{r1c,median}.pdf/png
Usage: /gpfs/scratch/delgaudio/skew/venv/bin/python plot_skew_panels.py
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, MultipleLocator, NullFormatter

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
sys.path.insert(0, str(DOCS))
from plot_final_curves import GRID, INK, ORDER, STYLE, pareto  # noqa: E402

OUT = HERE / "figs" / "panels"
TEXTWIDTH = 7.0
W, H = TEXTWIDTH / 5 - 0.02, 1.22          # inches, five panels per row (as curve5_*)
L, R, B, T = 0.40, 0.08, 0.29, 0.04        # fixed margins (inches): identical plot areas
FS = 7
XR = (0.79, 1.0)
TARGETS = [0.80, 0.90, 0.95, 0.99]
WL_MAIN = "r1c_p100"
WLS = ["r1a_p50", "r1a_p90", "r1a_p100", "r1b_p100", "r1c_p100", "r4_p50", "r4_p90", "r4_p100"]
LOG = "--log" in sys.argv
PFAM = [("r1a_p25", 0.25), ("r1a_p50", 0.5), ("r1a_p75", 0.75), ("r1a_p90", 0.9), ("r1a_p100", 1.0)]


def shade(color, p):
    """system colour mixed with white: light for small p, full colour at p=1."""
    # p=0.25 -> 55% white mix (light), p=1 -> 45% black mix (dark): wide contrast between the five lines
    t = -0.55 + (p - 0.25) / 0.75
    rgb = to_rgb(color)
    return tuple(c + (1 - c) * -t for c in rgb) if t < 0 else tuple(c * (1 - t) for c in rgb)


SLUG = {"GP-ANN*": "gpann", "Pyramid*": "pyramid", "Harmony": "harmony", "CoTra": "cotra",
        "DistributedANN*": "distributedann"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42,
})

cfg = {}
for r in csv.DictReader(open(DOCS / "skew_curves.csv")):
    w = "uniform" if r["workload"] == "uniform_rerun" and r["system"] == "GP-ANN*" else r["workload"]
    cfg.setdefault((r["system"], w, r["config"]), (float(r["recall10"]), float(r["qps"])))


UNI = ["uniform", "uniform_rep1", "uniform_rep2"]  # uniform baseline = per-config median of the 3 measurements


def uni_cfg(s):
    """[(recall, qps, config)] of the uniform workload: median over the available uniform measurements."""
    out = []
    for c in sorted({c for (s2, w, c) in cfg if s2 == s and w == "uniform"}):
        v = [cfg[(s, w, c)] for w in UNI if (s, w, c) in cfg]
        out.append((statistics.median(x[0] for x in v), statistics.median(x[1] for x in v), c))
    return out


# "--mono": for single-knob systems (Harmony: nprobe, CoTra: ef) draw EVERY configuration instead of the
# Pareto front, dropping only noisy points: configs in increasing cost order (= increasing uniform recall),
# a point is dropped when its QPS is higher than that of a cheaper configuration already kept (more work
# cannot be faster). Same rule for the uniform curve and every workload.
MONO = "--mono" in sys.argv
ONE_KNOB = {"Harmony", "CoTra"}


def mono_curve(s, w):
    """[(rec_uniform, qps)] of workload w (or the uniform median if w is None), monotone filter."""
    out, dropped = [], []
    for r, qu, c in sorted(uni_cfg(s)):
        q = qu if w is None else cfg.get((s, w, c), (None, None))[1]
        if q is None:
            continue
        if out and q > out[-1][1]:
            dropped.append(f"{c}={q:.0f}")
            continue
        out.append((r, q))
    if dropped:
        print(f"  {s} {w or 'uniform'}: dropped {', '.join(dropped)}")
    return out


def curves(s, wls):
    """uniform front [(rec,qps)], skewed front [(rec_uniform, median, min, max)] at the same configs."""
    ucfg = uni_cfg(s)
    uni = pareto([(a, b) for a, b, _ in ucfg])
    rows = []
    for a, _, c in ucfg:
        v = [cfg[(s, w, c)][1] for w in wls if (s, w, c) in cfg]
        if wls and len(v) == len(wls):
            rows.append((a, statistics.median(v), min(v), max(v)))
    keep = set(pareto([(a, m) for a, m, _, _ in rows]))
    return uni, sorted(r4 for r4 in rows if (r4[0], r4[1]) in keep)


def panel(s, kind):
    wls = [WL_MAIN] if kind == "r1c" else WLS
    st = STYLE[s]
    if kind == "pfam":
        have = [(w, p) for w, p in PFAM if any(k[0] == s and k[1] == w for k in cfg)]
        if MONO and s in ONE_KNOB:
            fam = [(w, p, mono_curve(s, w)) for w, p in have]
            uni = mono_curve(s, None)
        else:
            fam = [(w, p, curves(s, [w])[1]) for w, p in have]
            uni = curves(s, [])[0]
        sk = [q for _, _, f in fam for q in f]
    else:
        uni, sk = curves(s, wls)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
    for t in TARGETS:
        ax.axvline(t, color="#7a7974", lw=0.6, ls=(0, (3, 2)), zorder=1)
    uni = [p for p in uni if p[0] >= XR[0] - 0.03]
    sk = [p for p in sk if p[0] >= XR[0] - 0.03]
    if kind == "median":
        ax.fill_between([p[0] for p in sk], [p[2] for p in sk], [p[3] for p in sk], color=st["color"],
                        alpha=0.20, lw=0, zorder=2)
    ax.plot([p[0] for p in uni], [p[1] for p in uni], color=st["color"], lw=0.9, ls=(0, (3, 1.5)),
            marker=st["marker"], ms=2.4, mfc="white", mec=st["color"], mew=0.6, zorder=3)
    if kind == "pfam":
        for w, p, f in fam:
            f = [q for q in f if q[0] >= XR[0] - 0.03]
            ax.plot([q[0] for q in f], [q[1] for q in f], color=shade(st["color"], p), lw=1.0,
                    marker=st["marker"], ms=2.2, mec="white", mew=0.25, zorder=4 + p)
    else:
        ax.plot([p[0] for p in sk], [p[1] for p in sk], color=st["color"], lw=1.1,
                marker=st["marker"], ms=2.6, mec="white", mew=0.3, zorder=4)
    qs = [p[1] for p in uni] + [p[1] for p in sk] + ([p[2] for p in sk] if kind == "median" else [])
    qs = [q for q in qs if q > 0]
    kfmt = FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
    if LOG:
        ax.set_yscale("log")
        narrow = qs and max(qs) / min(qs) < 30
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if narrow else (1.0, 5.0)))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(0, max(qs) * 1.05)
        ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(kfmt)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlim(*XR)
    ax.grid(True, which="major", color=GRID, lw=0.35, ls="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=2, width=0.5, pad=1.2)
    ax.tick_params(axis="both", which="minor", length=1.2, width=0.35)
    ax.set_xlabel("Recall@10", labelpad=0.5)
    ax.set_ylabel("QPS (log scale)" if LOG else "QPS", labelpad=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"skew_{kind}{'mono' if MONO and kind == 'pfam' and s in ONE_KNOB else ''}_{SLUG[s]}"
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def legend(kind):
    if kind == "pfam":
        g = "#555555"
        h = [Line2D([], [], color=g, lw=0.9, ls=(0, (3, 1.5)), marker="o", ms=2.6, mfc="white", mec=g,
                    label="uniform ($p$=0)")]
        h += [Line2D([], [], color=shade(g, p), lw=1.0, marker="o", ms=2.6, mec="white", mew=0.25,
                     label=f"$p$={p:g}") for _, p in PFAM]
        fig = plt.figure(figsize=(TEXTWIDTH, 0.20))
        fig.legend(handles=h, loc="center", ncol=len(h), frameon=False, fontsize=FS, handlelength=2.2,
                   columnspacing=1.2, handletextpad=0.4, title="hot fraction, region R1a:", title_fontsize=FS)
        fig.savefig(OUT / "legend_skew_pfam.pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(OUT / "legend_skew_pfam.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        return
    h = [Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 1.5)), marker="o", ms=2.6, mfc="white", mec=INK,
                label="uniform workload"),
         Line2D([], [], color=INK, lw=1.1, marker="o", ms=2.8, mec="white", mew=0.3,
                label="skewed workload R1c ($p$=1), same configurations" if kind == "r1c" else "skewed workloads, same configurations (median)")]
    if kind == "median":
        h.append(Patch(facecolor=INK, alpha=0.2, lw=0, label="min–max over 8 workloads"))
    fig = plt.figure(figsize=(TEXTWIDTH, 0.20))
    fig.legend(handles=h, loc="center", ncol=len(h), frameon=False, fontsize=FS, handlelength=2.2,
               columnspacing=1.2, handletextpad=0.4)
    fig.savefig(OUT / f"legend_skew_{kind}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"legend_skew_{kind}.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    # "--sys NAME[,NAME]": only these systems; "--kind K[,K]": only these panel kinds
    SYS = sys.argv[sys.argv.index("--sys") + 1].split(",") if "--sys" in sys.argv else ORDER
    KINDS = sys.argv[sys.argv.index("--kind") + 1].split(",") if "--kind" in sys.argv else ("r1c", "median", "pfam")
    for s in SYS:
        print(s, "uniform measurements per config:",
              sorted({sum((s, w, c) in cfg for w in UNI) for (_, _, c) in uni_cfg(s)}))
    for kind in KINDS:
        for s in SYS:
            panel(s, kind)
        legend(kind)
    print(f"wrote {3 * len(ORDER)} panels + 3 legends to {OUT} ({'log' if LOG else 'linear'} y)")
