#!/usr/bin/env python3
"""Uniform vs skewed QPS/recall curves, DEEP100M 8 nodes, one panel per system (recall 0.80-0.99).

Default (summary over all skewed workloads):
  dashed  = uniform curve (Pareto front of docs/final_curves.csv)
  solid   = median over the skewed workloads of the iso-recall QPS
  band    = min-max over the skewed workloads (only recalls covered by every workload counted)
--wl W : instead of the summary, overlay the Pareto front of workload W (solid, with markers).

Data: docs/skew_curves.csv (python3 collect_skew.py first).
Output: figs/skew_curves[_<W>].pdf/png
Usage: /gpfs/scratch/delgaudio/skew/venv/bin/python plot_skew_curves.py [--wl r1a_p100]
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
sys.path.insert(0, str(DOCS))
sys.path.insert(0, str(HERE))
from plot_final_curves import GRID, INK, ORDER, STYLE, pareto  # noqa: E402
from collect_skew import iso  # noqa: E402

WL = sys.argv[sys.argv.index("--wl") + 1] if "--wl" in sys.argv else None
LINES = "--lines" in sys.argv  # one thin line per workload, colour = region family, width/alpha = p
FAM = {"r1": "#0072B2", "r2": "#D55E00", "r4": "#CC79A7"}  # Okabe-Ito (CVD-safe)
PSTY = {"p50": (0.7, 0.45), "p90": (1.0, 0.7), "p100": (1.4, 1.0)}  # (linewidth, alpha)
XR = (0.795, 0.995)
TARGETS = [0.80, 0.90, 0.95, 0.99]
GRIDR = np.round(np.arange(0.80, 0.9901, 0.0025), 4)
SKIP = {"uniform", "uniform_rerun"}
# Difficulty criterion (fixed before results): drop regions whose hot-query mean NN distance d1 differs
# >5% from the uniform query set (uniform 0.510; r1a .533 r1b .528 r1c .532 r4 .494 -> kept; r2 .377 -> dropped)
# ISOCFG: compared at the same configuration, not at the same recall (DA* uniform curve saturates at
# recall 0.93 and skewed workloads shift that ceiling, so iso-recall ratios are meaningless). Each uniform
# Pareto config is drawn at its *uniform* recall; skewed = median/min-max QPS of that config over the wl.
ISOCFG = set(ORDER) if "--samecfg" in sys.argv else {"DistributedANN*"}  # --samecfg: every system
PARTIAL = set()
EXCL = ("r2_",) if "--all" not in sys.argv else ()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})

pts = defaultdict(list)
cfg = {}
for r in csv.DictReader(open(DOCS / "skew_curves.csv")):
    pts[(r["system"], r["workload"])].append((float(r["recall10"]), float(r["qps"])))
    cfg[(r["system"], r["workload"], r["config"])] = (float(r["recall10"]), float(r["qps"]))
wls = sorted({w for _, w in pts if w not in SKIP and not w.startswith(EXCL)})

fig, axes = plt.subplots(1, len(ORDER), figsize=(7.0, 1.8 if LINES else 1.6), sharey=False)
for ax, s in zip(axes, ORDER):
    c = STYLE[s]["color"]
    for t in TARGETS:
        ax.axvline(t, color=GRID, lw=0.5, zorder=0)
    u = pareto(pts.get((s, "uniform"), []))
    u = [p for p in u if XR[0] - 0.02 <= p[0] <= 1]
    if u:
        ax.plot([p[0] for p in u], [p[1] for p in u], ls="--", color=INK if LINES else c, lw=1.1, label="uniform", zorder=5)
    if LINES:
        n = 0
        for w in wls:
            f = [p for p in pareto(pts.get((s, w), [])) if p[0] >= XR[0] - 0.02]
            if not f:
                continue
            fam, pp = w[:2], w.split("_")[1]
            lw, a = PSTY[pp]
            ax.plot([p[0] for p in f], [p[1] for p in f], color=FAM[fam], lw=lw, alpha=a)
            n += 1
    elif s in ISOCFG:
        ucfg = sorted((v[0], v[1], c3) for (s2, w2, c3), v in cfg.items() if s2 == s and w2 == "uniform")
        # every config -> (uniform recall, median/min/max skewed QPS); then the Pareto front of the skewed
        # medians, so both curves pick their own best configs (symmetric; no selection on uniform noise)
        rows = []
        for r_, q_, c3 in ucfg:
            v = [cfg[(s, w, c3)][1] for w in ([WL] if WL else wls) if (s, w, c3) in cfg]
            if v:
                rows.append((r_, statistics.median(v), min(v), max(v)))
        keep = set(pareto([(a, b) for a, b, _, _ in rows]))
        rows = sorted(r4 for r4 in rows if (r4[0], r4[1]) in keep)
        xs, med, lo, hi = [r4[0] for r4 in rows], [r4[1] for r4 in rows], [r4[2] for r4 in rows], [r4[3] for r4 in rows]
        if not WL:
            ax.fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(xs, med, color=c, lw=1.3, label=(WL or "skewed") + ",\nsame config")
        ax.legend(fontsize=5, frameon=False, loc="lower left", handlelength=1.8)
        ax.set_xlabel("recall@10 of config (uniform)", fontsize=6)
        n = len(wls)
    elif WL:
        f = [p for p in pareto(pts.get((s, WL), [])) if p[0] >= XR[0] - 0.02]
        if f:
            ax.plot([p[0] for p in f], [p[1] for p in f], color=c, lw=1.2, marker=STYLE[s]["marker"], ms=2.5,
                    label=WL)
        n = 1 if f else 0
    else:
        have = [w for w in wls if (s, w) in pts]
        med, lo, hi, xs = [], [], [], []
        pm, pl, ph, px, pn = [], [], [], [], []  # PARTIAL: median over the workloads that reach recall x
        for x in GRIDR:
            v = [iso(pts[(s, w)], x) for w in have]
            ok = [q for q in v if q is not None]
            if have and len(ok) == len(v):
                xs.append(x); med.append(statistics.median(v)); lo.append(min(v)); hi.append(max(v))
            if s in PARTIAL and ok:
                px.append(x); pm.append(statistics.median(ok)); pl.append(min(ok)); ph.append(max(ok)); pn.append(len(ok))
        if px:
            ax.fill_between(px, pl, ph, color=c, alpha=0.10, lw=0)
            ax.plot(px, pm, color=c, lw=1.0, ls=":", label="median of the wl\nreaching that recall")
            ax.legend(fontsize=5, frameon=False, loc="lower left", handlelength=1.8)
        if xs:
            ax.fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(xs, med, color=c, lw=1.3, label=None if s in PARTIAL else "skewed (median)")
        n = len(have)
    ax.set_yscale("log")
    ax.set_xlim(*XR)
    ax.set_title(f"{s}" + ("" if WL else (f"  (≤{n}/{len(wls)})" if s in PARTIAL else f"  ({n}/{len(wls)})")), fontsize=7, color=INK)
    if s not in ISOCFG or LINES:
        ax.set_xlabel("recall@10", fontsize=7)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", labelsize=6)
    ax.grid(axis="y", color=GRID, lw=0.4)
axes[0].set_ylabel("QPS", fontsize=7)
if LINES:
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, ls="--", lw=1.1, label="uniform")]
    h += [Line2D([], [], color=c, lw=1.4, label=f"{f} regions") for f, c in FAM.items()]
    h += [Line2D([], [], color="#777", lw=PSTY[k][0], alpha=PSTY[k][1], label=k.replace("p", "p=0.").replace("0.100", "1"))
          for k in PSTY]
    fig.legend(handles=h, loc="upper center", ncol=7, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
else:
    axes[0].legend(fontsize=5.5, frameon=False, loc="lower left")
    fig.tight_layout(pad=0.3)
out = HERE / "figs"
out.mkdir(exist_ok=True)
suffix = ("_all" if "--all" in sys.argv else "") + ("_samecfg" if "--samecfg" in sys.argv else "")
name = "skew_curves_lines" if LINES else (f"skew_curves_{WL}" if WL else "skew_curves") + suffix
for ext in ("pdf", "png"):
    fig.savefig(out / f"{name}.{ext}", dpi=200)
print(out / f"{name}.png")
