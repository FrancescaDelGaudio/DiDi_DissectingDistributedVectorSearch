#!/usr/bin/env python3
"""ORCAS (18.8M Bing click pairs query->MS MARCO doc). Clicked doc = relevant 'neighbor'.
Doc popularity skew + held-out coverage of a fast tier built from train queries."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
src, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
N_CORPUS = 3_213_835  # MS MARCO document corpus v1
df = pd.read_csv(src, sep="\t", header=None, names=["qid", "q", "did", "url"], usecols=[0, 1, 2, 3],
                 dtype={"qid": np.int64, "q": str, "did": str, "url": str}, quoting=3, na_filter=False)
d = df["did"].str[1:].astype(np.int64).to_numpy(); q = df["qid"].to_numpy()
r = dict(pairs=len(df), queries=int(np.unique(q).size), clicked_docs=int(np.unique(d).size), corpus=N_CORPUS)
r["distinct_query_strings"] = int(df["q"].nunique())
_, cpd = np.unique(d, return_counts=True); cs = np.sort(cpd)[::-1]
cum = np.cumsum(cs) / cs.sum()
for p in (0.001, 0.01, 0.05, 0.10, 0.20):
    r[f"pair_share_top{p*100:g}pct_corpus"] = float(cum[min(int(p * N_CORPUS), cs.size) - 1])
r["docs_with_1_query"] = int((cs == 1).sum()); r["max_queries_per_doc"] = int(cs[0])
r["top_docs"] = df.loc[df["did"].isin(pd.Series(np.unique(d)[np.argsort(-cpd)[:10]]).map(lambda x: f"D{x}")), "url"].drop_duplicates().head(10).tolist()
_, ppq = np.unique(q, return_counts=True)
r["clicks_per_query_mean"] = float(ppq.mean()); r["clicks_per_query_median"] = float(np.median(ppq))
gini = lambda c: float(1 - 2 * np.sum(np.cumsum(np.sort(c)) / c.sum()) / c.size + 1 / c.size)
r["gini_doc_popularity_clicked_docs"] = gini(cpd)
r["gini_doc_popularity_full_corpus"] = gini(np.concatenate([cpd, np.zeros(N_CORPUS - cpd.size)]))
# held-out: split queries
rng = np.random.default_rng(0); uq = np.unique(q); tr = set()
trq = rng.random(uq.size) < 0.5; trmask = np.isin(q, uq[trq])
ut, ct = np.unique(d[trmask], return_counts=True); order = ut[np.argsort(-ct, kind="stable")]
rank = pd.Series(np.arange(order.size), index=order)
dt, qt = d[~trmask], q[~trmask]
rk = rank.reindex(dt).fillna(np.inf).to_numpy()
tq = pd.DataFrame({"q": qt, "rk": rk}).groupby("q")["rk"].agg(["min", "max"])
fr = np.unique(np.concatenate([np.logspace(-5, np.log10(order.size / N_CORPUS), 60), [0.001, 0.01, 0.05, 0.1]]))
curve = []
for f in fr:
    B = int(f * N_CORPUS)
    curve.append((B / N_CORPUS, float((rk < B).mean()), float((tq["min"] < B).mean()), float((tq["max"] < B).mean())))
curve = np.array(curve)  # B/N, test pair coverage, frac test queries with >=1 clicked doc in T1, all clicked docs in T1
np.savez(OUT / "orcas.npz", curve=curve, doc_counts=cs, clicks_per_query=np.sort(ppq)[::-1])
r["heldout"] = {}
for f in (0.001, 0.01, 0.05, 0.1):
    i = np.argmin(abs(curve[:, 0] - f)); r["heldout"][f"{f}"] = dict(BoverN=curve[i,0], pair_cov=curve[i, 1], q_any=curve[i, 2], q_all=curve[i, 3])
r["heldout_all_train_docs"] = dict(BoverN=order.size / N_CORPUS, pair_cov=float(np.isfinite(rk).mean()))
json.dump(r, open(OUT / "orcas_skew.json", "w"), indent=1, default=float); print(json.dumps(r, indent=1, default=float))
