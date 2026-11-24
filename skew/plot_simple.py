#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(sys.argv[1]); O = Path(sys.argv[2])
BLUE, ORANGE, GRAY, INK2 = "#2a78d6", "#eb6834", "#9a9893", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False})
def bars(ax, labels, vals, colors, fmt):
    b = ax.bar(labels, vals, color=colors, width=0.6)
    for r, v in zip(b, vals): ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9, color="#0b0b0b")
def endlabel(ax, x, y, txt, c):
    ax.text(x[-1], y[-1], " " + txt, color=INK2, fontsize=8.5, va="center")

# ---- ORCAS
z = np.load(R / "orcas.npz"); c = z["curve"]; m = c[:, 0] <= 0.2
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
x = c[m, 0] * 100
ax[0].plot(x, c[m, 1] * 100, color=BLUE, lw=2, label="clicks covered")
ax[0].plot(x, c[m, 3] * 100, color=ORANGE, lw=2, label="queries fully answered by T1")
ax[0].plot([0, 20], [0, 20], "--", color=GRAY, lw=1.5, label="uniform (no skew)")
ax[0].set_xlim(0, 20); ax[0].set_ylim(0, 100); ax[0].set_xlabel("T1 size (% of corpus)"); ax[0].set_ylabel("% of held-out test queries")
ax[0].set_title("Works: 10% of docs → 77% of clicks, 69% of queries"); ax[0].legend(fontsize=8.5, loc="lower right")
j = json.load(open(R / "orcas_skew.json"))
vals = [j[f"pair_share_top{p}pct_corpus"] * 100 for p in ("1", "5", "10", "20")]
bars(ax[1], ["top 1%", "top 5%", "top 10%", "top 20%"], vals, BLUE, "{:.0f}%")
ax[1].plot([-0.3, 0.3], [1, 1], color=GRAY, lw=2); ax[1].plot([0.7, 1.3], [5, 5], color=GRAY, lw=2)
ax[1].plot([1.7, 2.3], [10, 10], color=GRAY, lw=2); ax[1].plot([2.7, 3.3], [20, 20], color=GRAY, lw=2, label="uniform")
ax[1].set_ylim(0, 100); ax[1].set_ylabel("% of all clicks"); ax[1].set_xlabel("most-clicked documents (% of corpus)")
ax[1].set_title("Works: few docs get most clicks"); ax[1].legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(O / "s_orcas.png", dpi=160)

# ---- benchmark GTs
j = json.load(open(R / "nn_frequency_all.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
names = ["DEEP10M", "DEEP100M", "TTI10M", "TTI100M", "MSMARCO10M", "MSMARCO100M"]
vals = [j[f"{n}_k10"]["heldout_test_slots_in_train_nn"] * 100 for n in names]
bars(ax[0], [n.replace("MSMARCO", "MSM-") for n in names], vals, BLUE, "{:.1f}%")
ax[0].set_ylim(0, 100); ax[0].set_ylabel("% of test 10-NN that are\nalso train 10-NN"); ax[0].tick_params(axis="x", labelsize=8.5)
ax[0].set_title("Does not work: benchmark NN barely repeat (<3%)")
a, b = j["TTI100M_k100"], j["TTI100M-100kq_k100"]
lab = [f"10k queries\n(T1 = {a['heldout_train_nn_BoverN']*100:.1f}% of D)", f"100k queries\n(T1 = {b['heldout_train_nn_BoverN']*100:.1f}% of D)"]
bars(ax[1], lab, [a["heldout_test_slots_in_train_nn"] * 100, b["heldout_test_slots_in_train_nn"] * 100], BLUE, "{:.1f}%")
ax[1].set_ylim(0, 100); ax[1].set_ylabel("% of test 100-NN covered by T1")
ax[1].set_title("Promising: more queries → more reuse (TTI100M)")
fig.tight_layout(); fig.savefig(O / "s_benchmarks.png", dpi=160)

# ---- MSMARCO spatial
s = json.load(open(R / "msmarco_skew.json"))["spatial"]["1024"]; z = np.load(R / "msmarco_spatial_C1024.npz"); c = z["curve"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
bars(ax[0], ["true 10-NN of a query", "10 random docs"], [s["mean_distinct_clusters_per_query"], s["mean_distinct_clusters_random10"]], [BLUE, GRAY], "{:.1f}")
ax[0].set_ylim(0, 10.9); ax[0].set_ylabel("# k-means cells (of 1024)")
ax[0].set_title("Works: a query's NN are spatially grouped")
m = c[:, 0] <= 0.3; x = c[m, 0] * 100
ax[1].plot(x, c[m, 1] * 100, color=BLUE, lw=2, label="test NN covered")
ax[1].plot(x, c[m, 2] * 100, color=ORANGE, lw=2, label="queries fully answered by T1")
ax[1].plot([0, 30], [0, 30], "--", color=GRAY, lw=1.5, label="uniform (no skew)")
ax[1].set_xlim(0, 30); ax[1].set_ylim(0, 100); ax[1].set_xlabel("T1 size (% of D, chosen by cluster)"); ax[1].set_ylabel("% (held-out queries)")
ax[1].set_title("Not enough alone: 10% of D → 25% NN, 5% of queries"); ax[1].legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(O / "s_msmarco_spatial.png", dpi=160)
print("ok")
