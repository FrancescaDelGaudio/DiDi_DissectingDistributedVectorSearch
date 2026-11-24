#!/usr/bin/env python3
"""Neighbor-frequency (hubness) on all available exact GTs. Header uint32 nq,k then nq*k int32 ids."""
import json, struct, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
V = Path("/gpfs/scratch/delgaudio/vectors"); OUT = Path(sys.argv[1])
GTS = [("DEEP10M", "DEEP/computedGT_deep10M_10kqueries_k100.bin", 10e6),
       ("DEEP100M", "DEEP/computedGT_deep100M_10kqueries_k100_cotra.bin", 100e6),
       ("TTI10M", "TTI/computedGT_tti10M_10kqueries_k100_cotra.bin", 10e6),
       ("TTI100M", "TTI/computedGT_tti100M_10kqueries_k100.bin", 100e6),
       ("TTI100M-100kq", "TTI/computedGT_tti100M_100kqueries_k100_cotra.bin", 100e6),
       ("MSMARCO10M", "MSMARCOWebSearch/computedGT_msmarco10M_9376q_k10.bin", 10e6),
       ("MSMARCO100M", "MSMARCOWebSearch/computedGT_msmarco100M_9376queries_k93_cotra.bin", 100e6)]
rng = np.random.default_rng(0); res = {}; fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, f, N in GTS:
    with open(V / f, "rb") as fh:
        nq, k = struct.unpack("II", fh.read(8)); ids = np.frombuffer(fh.read(nq * k * 4), dtype=np.int32).reshape(nq, k)
    for kk in (10, 100):
        if kk > k: continue
        g = ids[:, :kk]; flat = g.ravel(); u, c = np.unique(flat, return_counts=True); cs = np.sort(c)[::-1]
        perm = rng.permutation(nq); tr, te = perm[: nq // 2], perm[nq // 2:]
        trn = np.unique(g[tr]); hit = np.isin(g[te], trn)
        # neighbors ranked by train count: share of test slots covered by top 1% (of N) train-ranked docs
        ut, ct = np.unique(g[tr], return_counts=True); order = ut[np.argsort(-ct, kind="stable")]
        cum = np.cumsum(cs) / cs.sum()
        r = dict(nq=int(nq), k=kk, slots=int(flat.size), distinct_frac=u.size / flat.size,
                 expected_distinct_frac_uniform=N * (1 - (1 - 1 / N) ** flat.size) / flat.size,
                 hubness_ratio=(flat.size / u.size), max_count=int(cs[0]),
                 docs_ge2=int((c >= 2).sum()), docs_ge10=int((c >= 10).sum()),
                 slot_share_top1pct_of_distinct=float(cum[max(1, u.size // 100) - 1]),
                 slot_share_top10pct_of_distinct=float(cum[max(1, u.size // 10) - 1]),
                 heldout_test_slots_in_train_nn=float(hit.mean()),
                 heldout_train_nn_BoverN=trn.size / N,
                 heldout_test_queries_ge_half_in_train_nn=float((hit.sum(1) >= kk / 2).mean()))
        res[f"{name}_k{kk}"] = r; print(name, kk, json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}), flush=True)
        a = ax[0] if kk == 10 else ax[1]
        a.loglog(np.arange(1, cs.size + 1) / cs.size, cs, label=name)
for a, kk in zip(ax, (10, 100)):
    a.set_xlabel("neighbor rank (fraction of distinct neighbors)"); a.set_ylabel("# queries having it in top-k")
    a.set_title(f"How often a vector is a true {kk}-NN"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_nn_frequency_all.png", dpi=150); fig.savefig(OUT / "fig_nn_frequency_all.pdf")
json.dump(res, open(OUT / "nn_frequency_all.json", "w"), indent=1)
