#!/usr/bin/env python3
"""Heatmap QPS(skewed)/QPS(uniform) at iso-recall: rows = systems, columns = the 11 skewed workloads.
One panel per recall target (default 0.90 and 0.99). Colour: diverging on log2(ratio), grey at 1.
Data: docs/skew_isorecall.csv (python3 collect_skew.py first). Output: figs/skew_heatmap.pdf/png
"""
import csv, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
sys.path.insert(0, str(DOCS))
from plot_final_curves import INK, ORDER  # noqa: E402

TARGETS = [0.90, 0.99]
WLS = [w for f in ("r1a", "r1b", "r1c", "r2", "r4") for w in
       ([f"{f}_p50", f"{f}_p90", f"{f}_p100"] if f in ("r1a", "r2", "r4") else [f"{f}_p100"])]
LABEL = {"GP-ANN*": "GP-ANN*", "Pyramid*": "Pyramid*", "Harmony": "Harmony", "CoTra": "CoTra",
         "DistributedANN*": "DistributedANN*"}
# diverging: red (slower) - neutral grey - blue (faster)
CMAP = LinearSegmentedColormap.from_list("div", ["#b2182b", "#ef8a62", "#e8e6e1", "#67a9cf", "#2166ac"])
NORM = TwoSlopeNorm(vmin=math.log2(1 / 6), vcenter=0, vmax=math.log2(2))

ratio = {}
for r in csv.DictReader(open(DOCS / "skew_isorecall.csv")):
    if r["ratio"]:
        ratio[(r["system"], r["workload"], float(r["target"]))] = float(r["ratio"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, len(TARGETS), figsize=(7.0, 1.75), sharey=True)
for ax, t in zip(axes, TARGETS):
    M = np.full((len(ORDER), len(WLS)), np.nan)
    for i, s in enumerate(ORDER):
        for j, w in enumerate(WLS):
            v = ratio.get((s, w, t))
            if v:
                M[i, j] = math.log2(v)
    ax.imshow(np.ma.masked_invalid(M), cmap=CMAP, norm=NORM, aspect="auto")
    ax.set_facecolor("#f4f3f0")
    for i in range(len(ORDER)):
        for j in range(len(WLS)):
            v = M[i, j]
            txt = "–" if np.isnan(v) else f"{2 ** v:.2f}"
            dark = not np.isnan(v) and abs(v) > 1.3
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.5, color="white" if dark else INK)
    ax.set_xticks(range(len(WLS)), [w.replace("_", " ") for w in WLS], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(ORDER)), [LABEL[s] for s in ORDER], fontsize=6.5)
    ax.set_xticks(np.arange(-0.5, len(WLS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ORDER)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"recall@10 = {t:.2f}", fontsize=7, color=INK)
fig.text(0.5, 0.01, "cell = QPS(skewed) / QPS(uniform) at the same recall   ·   <1 slower (red), >1 faster (blue), – not reached",
         ha="center", fontsize=6, color=INK)
fig.tight_layout(pad=0.3, rect=(0, 0.05, 1, 1))
out = HERE / "figs"; out.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"skew_heatmap.{ext}", dpi=200)
print(out / "skew_heatmap.png")
