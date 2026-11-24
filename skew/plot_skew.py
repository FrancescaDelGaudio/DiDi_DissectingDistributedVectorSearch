#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = Path(sys.argv[1]); plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
B1, B2, B3, G = "#1f5fa8", "#d1495b", "#2a9d8f", "#888888"
# ORCAS
if (OUT / "orcas.npz").exists():
    z = np.load(OUT / "orcas.npz"); c = z["curve"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    dc = z["doc_counts"]; ax[0].loglog(np.arange(1, dc.size + 1), dc, color=B1)
    ax[0].set_xlabel("document rank"); ax[0].set_ylabel("# distinct queries clicking it"); ax[0].set_title("ORCAS: document popularity")
    ax[1].plot(c[:, 0] * 100, c[:, 1], color=B1, label="clicks covered (held-out)")
    ax[1].plot(c[:, 0] * 100, c[:, 2], color=B3, label="queries with ≥1 clicked doc in T1")
    ax[1].plot(c[:, 0] * 100, c[:, 3], color=B2, label="queries with all clicked docs in T1")
    ax[1].plot(c[:, 0] * 100, c[:, 0], "--", color=G, label="uniform (no skew)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("|T1| / |D|  (% of 3.2M docs, log)"); ax[1].set_ylabel("fraction"); ax[1].set_ylim(0, 1)
    ax[1].set_title("ORCAS: T1 from train queries, eval on test"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_orcas.png", dpi=150); fig.savefig(OUT / "fig_orcas.pdf")
# MSMARCO spatial
files = sorted(OUT.glob("msmarco_spatial_C*.npz"), key=lambda p: int(p.stem.split("C")[1]))
if files:
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for f, ls in zip(files, ["-", ":"]):
        C = f.stem.split("C")[1]; z = np.load(f); c, r = z["curve"], z["curve_rand"]
        ax[0].plot(c[:, 0] * 100, c[:, 1], ls, color=B1, label=f"NN covered, C={C}")
        ax[0].plot(c[:, 0] * 100, c[:, 2], ls, color=B2, label=f"queries recall@10=1, C={C}")
        ax[0].plot(c[:, 0] * 100, c[:, 3], ls, color=B3, label=f"queries recall@10≥0.9, C={C}")
        if ls == "-": ax[0].plot(r[:, 0] * 100, r[:, 1], "--", color=G, label="random clusters (uniform)")
        if ls == "-":
            ax[1].hist([z["ndist"], z["ndist_rnd"]], bins=np.arange(0.5, 11.5), color=[B1, G], label=["true 10-NN", "10 random docs"])
            ax[1].set_title(f"# k-means cells holding the 10-NN (C={C})")
            s = np.sort(z["w_train"])[::-1]; sz = z["size"][np.argsort(-z["w_train"])]
            ax[2].plot(np.cumsum(sz) / sz.sum() * 100, np.cumsum(s) / s.sum(), color=B1, label=f"cells by NN mass, C={C}")
            ax[2].plot([0, 100], [0, 1], "--", color=G, label="uniform")
    ax[0].set_xlabel("|T1| / |D|  (% of 10M)"); ax[0].set_ylabel("fraction (held-out queries)"); ax[0].set_xlim(0, 50); ax[0].set_ylim(0, 1)
    ax[0].set_title("MSMARCO-WS 10M: cluster-level T1"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("# distinct cells"); ax[1].set_ylabel("# queries"); ax[1].legend(fontsize=8)
    ax[2].set_xlabel("% of D (cells sorted by train NN mass)"); ax[2].set_ylabel("share of train NN"); ax[2].set_title("Lorenz curve of NN mass over cells"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_msmarco_spatial.png", dpi=150); fig.savefig(OUT / "fig_msmarco_spatial.pdf")
print("ok")
