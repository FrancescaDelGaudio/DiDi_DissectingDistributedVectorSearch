#!/usr/bin/env python3
"""Quora: same intent, different wording. For each test query take one duplicate question from the corpus
(qrels) as the 'earlier query'. Do the two formulations share their top-10 NN? Cache vs T1 when the earlier
formulation was seen."""
import json, sys
from pathlib import Path
import numpy as np
NP = Path("/gpfs/scratch/delgaudio/beir/np/quora"); OUT = Path(sys.argv[1]); K = 10; rng = np.random.default_rng(0)
X = np.load(NP / "base.npy"); N = len(X)
Q = np.load(NP / "q_quora_test.npy"); G = np.load(NP / "gt_quora_test.npy"); qr = np.load(NP / "qrels_quora_test.npy")
first = {}
for q, d in qr: first.setdefault(int(q), int(d))
qs = np.array(sorted(first)); dup = np.array([first[q] for q in qs])
P = X[dup]                                              # earlier formulation (a duplicate question)
sc = P @ X.T; sc[np.arange(len(dup)), dup] = -np.inf     # exclude the duplicate itself
Gp = np.argpartition(-sc, K, axis=1)[:, :K]
sim = np.einsum("ij,ij->i", Q[qs], P)
jac = np.array([len(set(a) & set(b)) / K for a, b in zip(G[qs], Gp)])   # = recall of cached results
h = np.isin(G[qs], np.unique(Gp)).sum(1)               # T1 = union of NN of earlier formulations (vector-level)
# also: T1 recall if each query's own earlier formulation NN are in T1 (per-query view)
own = np.array([np.isin(a, b).sum() for a, b in zip(G[qs], Gp)])
ths = np.round(np.arange(0.70, 1.001, 0.01), 2)
rows = np.array([(s, (sim >= s).mean(), ((sim >= s) & (jac >= .9)).mean(), ((sim >= s) & (jac < .9)).mean()) for s in ths])
r = dict(pairs=int(len(qs)), sim_quantiles=np.quantile(sim, [.1, .25, .5, .75, .9]).round(3).tolist(),
         overlap_top10_mean=float(jac.mean()), overlap_quantiles=np.quantile(jac, [.1, .25, .5, .75, .9]).tolist(),
         frac_pairs_overlap_ge_0_5=float((jac >= .5).mean()), frac_pairs_overlap_ge_0_9=float((jac >= .9).mean()),
         T1_vector=dict(BoverN=np.unique(Gp).size / N, nn_cov=float(np.isin(G[qs], np.unique(Gp)).mean()),
                        q_ge09=float((h >= 9).mean()), q_full=float((h == K).mean())),
         cache_best_good=float(rows[:, 2].max()), cache_rows={f"{x[0]:.2f}": dict(hit=x[1], good=x[2], false_hit=x[3]) for x in rows[::5]})
np.savez(OUT / "quora_paraphrase.npz", sim=sim, jac=jac, rows=rows, t1_hits=h)
json.dump(r, open(OUT / "quora_paraphrase.json", "w"), indent=1, default=float); print(json.dumps(r, indent=1, default=float))
