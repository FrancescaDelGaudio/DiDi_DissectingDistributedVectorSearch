#!/usr/bin/env python3
"""BEIR experiments (Cohere embed-english-v3, exact top-10 GT from beir_gt.py).
E4 encoder check | E1 topical locality + drift (FEVER corpus, FEVER vs Climate-FEVER) |
E2 semantic cache vs tier (Quora) | E3 reuse vs number of train queries (FEVER train).
usage: beir_analysis.py {quora|fever} OUTDIR"""
import json, sys, time
from pathlib import Path
import numpy as np
NP = Path("/gpfs/scratch/delgaudio/beir/np"); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
K = 10; rng = np.random.default_rng(0); t0 = time.time(); RES = {}
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
def L(ds, name): return np.load(NP / ds / name, allow_pickle=True)

def encoder_check(ds, tag):
    G = L(ds, f"gt_{tag}.npy"); qr = L(ds, f"qrels_{tag}.npy")
    hit = np.array([d in set(G[q]) for q, d in qr]) if len(qr) else np.array([])
    per_q = {}
    for (q, d), h in zip(qr, hit): per_q.setdefault(q, []).append(h)
    r = dict(qrels=int(len(qr)), frac_relevant_in_top10=float(hit.mean()),
             frac_queries_with_a_relevant_in_top10=float(np.mean([any(v) for v in per_q.values()])))
    log("E4", tag, r); return r

def nearest(A, C, bs=100_000):
    out = np.empty(len(A), np.int32)
    for s in range(0, len(A), bs): out[s:s + bs] = np.argmax(A[s:s + bs] @ C.T, axis=1)   # unit-norm: max IP
    return out
def kmeans(S, C, it=12):
    cent = S[rng.choice(len(S), C, replace=False)].copy()
    for _ in range(it):
        a = nearest(S, cent); o = np.argsort(a, kind="stable"); st = np.searchsorted(a[o], np.arange(C))
        cnt = np.bincount(a, minlength=C); sums = np.add.reduceat(S[o], np.minimum(st, len(S) - 1), axis=0)
        ok = cnt > 0; cent[ok] = sums[ok] / cnt[ok, None]; cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        e = np.where(~ok)[0]
        if e.size: cent[e] = S[rng.choice(len(S), e.size, replace=False)]
    return cent

def cluster_tier(assign, size, Gtr, Gte, N):
    """rank cells by train-NN mass; returns rows (B/N, NN covered, q recall=1, q recall>=0.9) per prefix"""
    C = len(size); w = np.bincount(assign[Gtr].ravel(), minlength=C).astype(float)
    order = np.argsort(-w, kind="stable"); rank = np.empty(C, np.int64); rank[order] = np.arange(C)
    tr = rank[assign[Gte]]; cumN = np.cumsum(size[order]) / N
    return np.array([(cumN[m - 1], (tr < m).mean(), ((tr < m).sum(1) == K).mean(), ((tr < m).sum(1) >= 9).mean()) for m in range(1, C + 1)])
def vector_tier(Gtr, Gte, N):
    S = np.unique(Gtr); h = np.isin(Gte, S).sum(1)
    return dict(BoverN=S.size / N, nn_cov=float(np.isin(Gte, S).mean()), q_full=float((h == K).mean()), q_ge09=float((h >= 9).mean()))
def at(curve, f):
    i = max(0, np.searchsorted(curve[:, 0], f, side="right") - 1)
    return dict(BoverN=float(curve[i, 0]), nn_cov=float(curve[i, 1]), q_full=float(curve[i, 2]), q_ge09=float(curve[i, 3]))

# ------------------------------------------------------------------ Quora (E4, E2)
if sys.argv[1] == "quora":
    ds = "quora"; X = L(ds, "base.npy"); N = len(X)
    RES["E4"] = {t: encoder_check(ds, t) for t in ("quora_dev", "quora_test")}
    Qtr, Gtr = L(ds, "q_quora_dev.npy"), L(ds, "gt_quora_dev.npy")
    Qte, Gte = L(ds, "q_quora_test.npy"), L(ds, "gt_quora_test.npy")
    sim = Qte @ Qtr.T; nn = sim.argmax(1); smax = sim.max(1)
    cached = Gtr[nn]                                           # results of the most similar train query
    rec_cache = np.array([len(set(a) & set(b)) / K for a, b in zip(cached, Gte)])
    vt = vector_tier(Gtr, Gte, N)                              # T1 = union of train NN (same memory as caching all results)
    h = np.isin(Gte, np.unique(Gtr)).sum(1)
    ths = np.round(np.arange(0.70, 1.001, 0.01), 2); rows = []
    for s in ths:
        hit = smax >= s
        rows.append((s, hit.mean(), (hit & (rec_cache >= 0.9)).mean(), (hit & (rec_cache < 0.9)).mean()))
    rows = np.array(rows)   # threshold, hit rate, good hits (recall>=0.9), bad hits (false accepts)
    np.savez(OUT / "quora_e2.npz", rows=rows, smax=smax, rec_cache=rec_cache, t1_hits=h)
    RES["E2"] = dict(train_queries=len(Qtr), test_queries=len(Qte), t1_vector=vt,
                     cache_best_good=float(rows[:, 2].max()), cache_rows={f"{r[0]:.2f}": dict(hit=r[1], good=r[2], false_hit=r[3]) for r in rows[::5]},
                     test_nn_train_query_sim_quantiles=np.quantile(smax, [.1, .25, .5, .75, .9]).round(3).tolist(),
                     corr_sim_vs_cache_recall=float(np.corrcoef(smax, rec_cache)[0, 1]))
    log("E2", json.dumps(RES["E2"]))

# ------------------------------------------------------------------ FEVER corpus (E4, E1, E3)
if sys.argv[1] == "fever":
    ds = "fever"; X = L(ds, "base.npy"); N = len(X); log("base", X.shape)
    RES["E4"] = {t: encoder_check(ds, t) for t in ("fever_test", "climate-fever_test")}
    Gf_tr, Gf_te, Gc = L(ds, "gt_fever_train.npy"), L(ds, "gt_fever_test.npy"), L(ds, "gt_climate-fever_test.npy")
    # E3: reuse vs number of train queries (vector-level T1 = union of train NN), test = fever test
    RES["E3"] = {}
    for n in (1000, 3000, 10000, 30000, 109810):
        sel = rng.choice(len(Gf_tr), n, replace=False); RES["E3"][n] = vector_tier(Gf_tr[sel], Gf_te, N); log("E3", n, RES["E3"][n])
    # E1: k-means cells
    C = 1024; samp = np.asarray(X[np.sort(rng.choice(N, 300_000, replace=False))]); cent = kmeans(samp, C); log("kmeans done")
    assign = np.empty(N, np.int32)
    for s in range(0, N, 500_000): assign[s:s + 500_000] = nearest(np.asarray(X[s:s + 500_000]), cent)
    size = np.bincount(assign, minlength=C); log("assigned")
    nC = len(Gc); fsub = Gf_te[rng.choice(len(Gf_te), nC, replace=False)]      # same #queries for fairness
    def conc(G):
        w = np.bincount(assign[G].ravel(), minlength=C); o = np.argsort(-w); cw = np.cumsum(w[o]) / w.sum(); cn = np.cumsum(size[o]) / N
        return dict(cells_touched=int((w > 0).sum()), D_frac_for_50pct_NN=float(cn[np.searchsorted(cw, .5)]),
                    D_frac_for_80pct_NN=float(cn[np.searchsorted(cw, .8)]), D_frac_for_90pct_NN=float(cn[np.searchsorted(cw, .9)]),
                    mean_cells_per_query=float(np.mean([np.unique(r).size for r in assign[G]])))
    RES["E1_concentration"] = {"fever": conc(fsub), "climate": conc(Gc)}; log("E1 conc", RES["E1_concentration"])
    # E1 per-context tier: split each query set 50/50
    curves = {}
    for name, G in (("fever", fsub), ("climate", Gc)):
        p = rng.permutation(len(G)); tr, te = G[p[: len(G) // 2]], G[p[len(G) // 2:]]
        curves[name] = cluster_tier(assign, size, tr, te, N)
    # cross: tier built from FEVER train queries (generic), tested on climate vs fever test
    curves["fever_train->climate"] = cluster_tier(assign, size, Gf_tr, Gc, N)
    curves["fever_train->fever"] = cluster_tier(assign, size, Gf_tr, Gf_te, N)
    np.savez(OUT / "fever_e1_curves.npz", **{k.replace(">", "").replace("-", "_"): v for k, v in curves.items()}, size=size)
    RES["E1_tier"] = {k: {f"{f}": at(v, f) for f in (0.01, 0.02, 0.05, 0.10, 0.20)} for k, v in curves.items()}
    log("E1 tier", json.dumps(RES["E1_tier"]))
    # E1 drift: T1 (cells, budget B) from FEVER train; stream FEVER test then Climate; re-layer from recent queries
    W = 50; nwin = 30; budget = 0.05
    def build(Gsrc):
        w = np.bincount(assign[Gsrc].ravel(), minlength=C); o = np.argsort(-w, kind="stable"); m = np.searchsorted(np.cumsum(size[o]) / N, budget, side="right")
        inT = np.zeros(C, bool); inT[o[:m]] = True; return inT
    fs = Gf_te[rng.permutation(len(Gf_te))[: nwin * W]]; cs = Gc[rng.permutation(len(Gc))[: nwin * W]]
    stream = np.concatenate([fs, cs]); n_w = len(stream) // W; switch = nwin
    T_static = build(Gf_tr); T = T_static.copy(); trace = []; relayer_at = None; hist = []
    for t in range(n_w):
        g = stream[t * W:(t + 1) * W]; rec_s = T_static[assign[g]].mean(1); rec_a = T[assign[g]].mean(1)
        rho_s, rho_a = float((rec_s < 0.9).mean()), float((rec_a < 0.9).mean()); trace.append((t, rho_s, rho_a, rec_s.mean(), rec_a.mean()))
        hist.append(g)
        base_rho = np.mean([x[2] for x in trace[:5]]) if len(trace) >= 5 else None
        if relayer_at is None and base_rho is not None and t >= 5 and rho_a > base_rho + 0.10:   # trigger
            relayer_at = t; T = build(np.concatenate(hist[-3:]))                                 # rebuild from recent queries
    trace = np.array(trace); np.savez(OUT / "fever_e1_drift.npz", trace=trace, switch=switch, relayer_at=-1 if relayer_at is None else relayer_at, W=W, budget=budget)
    RES["E1_drift"] = dict(window=W, budget=budget, switch_window=switch, relayer_window=relayer_at,
                           rho_static_before=float(trace[:switch, 1].mean()), rho_static_after=float(trace[switch:, 1].mean()),
                           rho_adaptive_after_relayer=float(trace[(relayer_at or switch) + 1:, 2].mean()))
    log("E1 drift", RES["E1_drift"])
json.dump(RES, open(OUT / f"beir_{sys.argv[1]}.json", "w"), indent=1, default=float); log("done")
