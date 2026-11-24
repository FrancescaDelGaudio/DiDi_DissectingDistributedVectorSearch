#!/usr/bin/env python3
"""Recall/QPS curves for every setup (DEEP / TTI / MSMARCO, 10M / 100M, 4 / 6 / 8 nodes), drawn at the
printed size of the build-time layout: six panels per row of a two-column paper (7 in), each dataset a
half-row subfigure with its 4 / 6 / 8-node panels. Same colours, markers, Pareto fronts, target lines and
axes as plot_curves_subfigs.py; fixed margins, so every panel has the same plot area.

Data: final_curves.csv (DEEP10M 4/6n, TTI10M 4/6n, DEEP100M 8n, TTI100M 8n), scaling_curves.csv (the
other DEEP / TTI setups; "DA*" = DistributedANN*), msmarco_curves.csv (MSMARCO).
Per dataset (as in the existing figure): TTI10M without Harmony (max recall < 0.70) and with target 0.70;
TTI100M without Harmony and DistributedANN* (not evaluated there, max recall < 0.60); TTI100M-8n keeps
its targets 0.80 / 0.90 / 0.92 (GP-ANN* max 0.954); MSMARCO only GP-ANN*, Pyramid*, CoTra (10M) and CoTra
(100M).

Output: subfigs/curve6_<setup>.pdf/png (--five: curve5_<setup>.pdf/png) (legend: legend_curves.pdf of plot_curves_subfigs.py)
Usage:
    module load python/3.12.2
    python3 plot_curves_panels6.py [--five]
"""
import csv
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_final_curves import GRID, INK, ORDER, STYLE, pareto  # noqa: E402

OUT = HERE / "subfigs"
TEXTWIDTH = 7.0
# "--five": five panels per row (one column per dataset, one row per node count, no MSMARCO100M),
# slightly larger panels and fonts, files curve5_*.pdf
FIVE = "--five" in sys.argv
PREFIX = "curve5" if FIVE else "curve6"
W, H = (TEXTWIDTH / 5 - 0.02, 1.22) if FIVE else (TEXTWIDTH / 6, 1.12)  # inches
L, R, B, T = (0.40, 0.08, 0.29, 0.04) if FIVE else (0.37, 0.08, 0.27, 0.04)  # fixed margins (inches)
FS = 7 if FIVE else 6.5
STD = [0.80, 0.90, 0.95, 0.99]
XR = (0.69, 1.0)
TTI = ["GP-ANN*", "Pyramid*", "CoTra"]
SPEC = {  # dataset -> (systems, targets)
    "DEEP10M": (ORDER, STD),
    "DEEP100M": (ORDER, STD),
    "TTI10M": (TTI + ["DistributedANN*"], [0.70] + STD),
    "TTI100M": (TTI, STD),
    "MSMARCO10M": (TTI, STD),
    "MSMARCO100M": (["CoTra"], STD),
}
if FIVE:
    del SPEC["MSMARCO100M"]  # CoTra only
EXCEPT = {"TTI100M-8n": ([0.80, 0.90, 0.92], (0.69, 0.98))}  # setup -> (targets, x-range)
# "--x08 SETUP[,SETUP...]": only these setups, x axis from recall 0.80 (targets below 0.80 dropped),
# y range fitted to the visible part; files <prefix>x08_<setup>.pdf (the default panels are untouched)
X08 = sys.argv[sys.argv.index("--x08") + 1].split(",") if "--x08" in sys.argv else None
X08_XR = (0.795, 1.0)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42,
})


FIXED = "/gpfs/scratch/delgaudio/dhnsw/logs/sweep_v2_fixed"

# DEEP10M-8n Pyramid*: K=10/20 ef=50 last 0.12-0.14 s and come out faster than K=1 ef=50 (noise); dropped so
# the front covers recall 0.80-0.99
DROP = {("DEEP10M-8n", "Pyramid*"): {"K=10 ef=50", "K=20 ef=50"}}


def load():
    pts = defaultdict(list)
    for name in ("final_curves.csv", "scaling_curves.csv", "msmarco_curves.csv"):
        seen = set(pts)  # setups already given by an earlier file keep that file's points
        for r in csv.DictReader(open(HERE / name)):
            key = (r["setup"], r["system"].replace("DA*", "DistributedANN*"))
            if key not in seen:
                pts[key].append((float(r["recall10"]), float(r["qps"]), r["config"]))
    # Pyramid*: corrected sweeps (rankfile slot=0-127, 16 real cores) replace the old points once the job is done
    for f in sorted(glob.glob(f"{FIXED}/pyramid_v2_sweep_*n_[0-9]*[0-9].tsv")):
        tag, job = re.search(r"sweep_(\w+?_\d+n)_(\d+)\.tsv$", f).groups()
        log = glob.glob(f"{FIXED}/pyr-v2-{tag}_{job}.log")
        if not log or "==== done ====" not in open(log[0], errors="replace").read():
            continue
        ds, n = tag.rsplit("_", 1)
        key = (f"{ds.upper()}-{n}", "Pyramid*")
        pts[key] = [(float(r["recall_at_10"]), float(r["qps"]), f"K={r['K']} ef={r['ef']}")
                    for r in csv.DictReader(open(f), delimiter="\t") if r["rc"] == "0"]
        print(f"Pyramid* {key[0]}: fixed sweep {Path(f).name}")
    for key, cfgs in DROP.items():  # short noisy runs that would hide the 0.80-0.90 part of the front
        pts[key] = [p for p in pts.get(key, []) if p[2] not in cfgs]
    return pts


def panel(setup, systems, targets, xr, pts):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
    for t in targets:
        ax.axvline(t, color="#7a7974", lw=0.6, ls=(0, (3, 2)), zorder=1)
    qs = []
    for s in systems:
        data = pts.get((setup, s), [])
        if not data:
            continue
        front = pareto(data)
        if X08:  # keep the visible points plus the last one before the axis start, so the line enters from the edge
            vis = [i for i, p in enumerate(front) if p[0] >= xr[0]]
            front = front[max(vis[0] - 1, 0):] if vis else []
            if not front:
                continue
        qs += [p[1] for p in front if xr[0] <= p[0] <= xr[1]]
        st = STYLE[s]
        ax.plot([p[0] for p in front], [p[1] for p in front], color=st["color"], lw=0.9,
                marker=st["marker"], ms=2.6, mec="white", mew=0.3, zorder=3)
    ax.set_yscale("log")
    narrow = qs and max(qs) / min(qs) < 30  # e.g. one curve only: add 2x ticks
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if narrow else (1.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlim(*xr)
    if X08 and qs:
        ax.set_ylim(min(qs) / 1.35, max(qs) * 1.35)
    ax.grid(True, which="major", color=GRID, lw=0.35, ls="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=2, width=0.5, pad=1.2)
    ax.tick_params(axis="both", which="minor", length=1.2, width=0.35)
    ax.set_xlabel("Recall@10", labelpad=0.5)
    ax.set_ylabel("QPS (log scale)", labelpad=0.5)
    name = setup.lower().replace("-", "_")
    OUT.mkdir(exist_ok=True)
    pref = PREFIX + ("x08" if X08 else "")
    fig.savefig(OUT / f"{pref}_{name}.pdf")
    fig.savefig(OUT / f"{pref}_{name}.png", dpi=600)
    plt.close(fig)


def main():
    pts = load()
    for ds, (systems, targets) in SPEC.items():
        for n in (4, 6, 8):
            setup = f"{ds}-{n}n"
            if X08 and setup not in X08:
                continue
            tg, xr = EXCEPT.get(setup, (targets, XR))
            if X08:
                tg, xr = [t for t in tg if t >= 0.8], X08_XR
            panel(setup, systems, tg, xr, pts)
    print(f"wrote panels {PREFIX}{'x08' if X08 else ''}_* to {OUT}")


if __name__ == "__main__":
    main()
