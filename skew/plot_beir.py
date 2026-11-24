#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(sys.argv[1]); O = Path(sys.argv[2]); O.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GRAY, INK2 = "#2a78d6", "#eb6834", "#9a9893", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": .6, "axes.axisbelow": True, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False})
z = np.load(R / "fever_e1_curves.npz")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for key, c, lab in (("climate", ORANGE, "Climate-FEVER (topical)"), ("fever", BLUE, "FEVER (generic)")):
    v = z[key]; m = v[:, 0] <= .2
    ax[0].plot(v[m, 0] * 100, v[m, 3] * 100, color=c, lw=2, label=lab)
ax[0].plot([0, 20], [0, 0], color=GRAY, lw=0)
ax[0].set_xlim(0, 20); ax[0].set_ylim(0, 100); ax[0].set_xlabel("T1 size (% of 5.4M Wikipedia docs)"); ax[0].set_ylabel("% test queries with recall@10 ≥ 0.9 on T1")
ax[0].set_title("Topical traffic: 5% of D serves 82% of queries"); ax[0].legend(fontsize=8.5, loc="lower right")
d = np.load(R / "fever_e1_drift.npz"); tr = d["trace"]; sw = int(d["switch"])
ax[1].plot(tr[:, 0], tr[:, 1] * 100, color=GRAY, lw=2, label="static T1 (built from FEVER)")
ax[1].plot(tr[:, 0], tr[:, 2] * 100, color=BLUE, lw=2, label="re-layered T1")
ax[1].axvline(sw - .5, color=INK2, lw=1, ls=":"); ax[1].text(sw - .3, 5, "topic switch\nFEVER → Climate", fontsize=8, color=INK2)
ax[1].set_ylim(0, 105); ax[1].set_xlabel("time window (50 queries each)"); ax[1].set_ylabel("ρ: % queries descending below T1")
ax[1].set_title("Drift: ρ jumps to 100%, re-layering brings it to 47%"); ax[1].legend(fontsize=8.5, loc="lower left")
fig.tight_layout(); fig.savefig(O / "s_beir_e1.png", dpi=160); fig.savefig(O / "s_beir_e1.pdf"); print("ok")

# ---- Quora cache vs tier + FEVER volume
q = json.load(open(R / "quora_paraphrase.json")); f = json.load(open(R / "beir_fever.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
c09 = q["cache_rows"]["0.90"]
labs = ["semantic cache\n(best threshold)", "semantic cache\n(threshold 0.90)", "T1 = same docs,\nsearched"]
good = [q["cache_best_good"] * 100, c09["good"] * 100, q["T1_vector"]["q_ge09"] * 100]
bad = [0, c09["false_hit"] * 100, 0]
b = ax[0].bar(labs, good, color=[GRAY, GRAY, BLUE], width=.6)
ax[0].bar(labs, bad, bottom=good, color="#e34948", width=.6, label="wrong answers (hit, recall < 0.9)")
for r, v in zip(b, good): ax[0].text(r.get_x() + r.get_width() / 2, v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
ax[0].set_ylim(0, 100); ax[0].set_ylabel("% queries served (recall ≥ 0.9)"); ax[0].tick_params(axis="x", labelsize=8.5)
ax[0].set_title("Quora paraphrases: tier beats cache (same memory)"); ax[0].legend(fontsize=8, loc="upper left")
e3 = f["E3"]; ns = sorted(int(k) for k in e3)
ax[1].plot(ns, [e3[str(n)]["nn_cov"] * 100 for n in ns], "o-", color=BLUE, lw=2, ms=6, label="test NN covered")
ax[1].plot(ns, [e3[str(n)]["q_ge09"] * 100 for n in ns], "o-", color=ORANGE, lw=2, ms=6, label="queries with recall ≥ 0.9")
for n in ns: ax[1].text(n, e3[str(n)]["nn_cov"] * 100 + 3, f"T1={e3[str(n)]['BoverN']*100:.1f}%", ha="center", fontsize=7.5, color=INK2)
ax[1].set_xscale("log"); ax[1].set_ylim(0, 60); ax[1].set_xlabel("# train queries used to build T1 (FEVER)"); ax[1].set_ylabel("% (test queries)")
ax[1].set_title("More traffic → more reuse (not saturated)"); ax[1].legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(O / "s_beir_e2e3.png", dpi=160); fig.savefig(O / "s_beir_e2e3.pdf"); print("ok2")
