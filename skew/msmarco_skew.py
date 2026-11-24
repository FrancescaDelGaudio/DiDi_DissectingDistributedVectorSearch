#!/usr/bin/env python3
"""Skew / locality study on MSMARCO Web Search (768-d, 9376 real Bing queries, exact top-10 GT).
1) neighbor frequency (hubness) on 10M and 100M GT
2) vector-level coverage Cov(B): rank docs by train-query frequency, measure on held-out queries
3) spatial locality: k-means on 10M base; neighbors per query in how many clusters;
   cluster-level tier T1 chosen from train queries, coverage on test queries vs |T1|/N
Output: OUT/msmarco_*.json and .npz"""
import json, struct, sys, time
from pathlib import Path
import numpy as np, faiss

DIR = Path("/gpfs/scratch/delgaudio/vectors/MSMARCOWebSearch")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
D, K = 768, 10
rng = np.random.default_rng(0)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def load_gt(p):
    with open(p, "rb") as f:
        nq, k = struct.unpack("ii", f.read(8))
        ids = np.frombuffer(f.read(nq * k * 4), dtype=np.int32).reshape(nq, k)
    return ids

nq = load_gt(DIR / "computedGT_msmarco10M_9376q_k10.bin").shape[0]
perm = rng.permutation(nq); train, test = perm[: nq // 2], perm[nq // 2:]
res = {"nq": int(nq), "n_train": len(train), "n_test": len(test)}

def neighbor_stats(ids, N, tag):
    flat = ids.ravel()
    u, c = np.unique(flat, return_counts=True)
    slots = flat.size
    exp_distinct_uniform = N * (1 - (1 - 1 / N) ** slots)
    cs = np.sort(c)[::-1]
    r = dict(N=N, slots=int(slots), distinct=int(u.size), distinct_frac=u.size / slots,
             expected_distinct_if_uniform=exp_distinct_uniform,
             max_count=int(cs[0]), top10_counts=cs[:10].tolist(),
             slots_on_docs_seen_ge2=float(c[c >= 2].sum() / slots),
             docs_seen_ge2=int((c >= 2).sum()), docs_seen_ge5=int((c >= 5).sum()))
    # held-out vector-level coverage: rank by train frequency (greedy = top-B since objective is modular)
    ut, ct = np.unique(ids[train].ravel(), return_counts=True)
    order = ut[np.argsort(-ct, kind="stable")]
    test_flat = ids[test].ravel()
    Bs = np.unique(np.concatenate([np.logspace(0, np.log10(order.size), 40).astype(int), [order.size]]))
    cov = []
    for B in Bs:
        S = order[:B]
        cov.append(np.isin(test_flat, S).mean())
    r["heldout_cov_B"] = Bs.tolist(); r["heldout_cov"] = [float(x) for x in cov]
    r["heldout_cov_all_train_neighbors"] = float(cov[-1])
    r["heldout_cov_all_train_neighbors_BoverN"] = order.size / N
    # fraction of test queries with >=1 / all neighbors among train neighbors
    hit = np.isin(ids[test], order).sum(1)
    r["test_queries_any_nn_in_train_nn"] = float((hit >= 1).mean())
    r["test_queries_all10_in_train_nn"] = float((hit == K).mean())
    np.save(OUT / f"msmarco_{tag}_nn_counts.npy", cs)
    log(tag, json.dumps({k: v for k, v in r.items() if not k.startswith("heldout_cov_B") and k != "heldout_cov"}))
    return r

gt10 = load_gt(DIR / "computedGT_msmarco10M_9376q_k10.bin")
gt100 = load_gt(DIR / "computedGT_msmarco100M_9376q_k10.bin")
res["nn_10M"] = neighbor_stats(gt10, 10_000_000, "10M")
res["nn_100M"] = neighbor_stats(gt100, 100_000_000, "100M")
json.dump(res, open(OUT / "msmarco_skew.json", "w"), indent=1)

# ---------------- spatial locality on 10M ----------------
N = 10_000_000
log("loading 10M base")
X = np.fromfile(DIR / "base.10M.fbin", dtype=np.float32, offset=8, count=N * D).reshape(N, D)
Q = np.fromfile(DIR / "query.vectors.bin", dtype=np.float32, offset=8).reshape(-1, D)
norms = np.linalg.norm(X[rng.choice(N, 100000, replace=False)], axis=1)
res["base_norm_mean_std"] = [float(norms.mean()), float(norms.std())]
res["query_norm_mean_std"] = [float(np.linalg.norm(Q, axis=1).mean()), float(np.linalg.norm(Q, axis=1).std())]
log("norms", res["base_norm_mean_std"], res["query_norm_mean_std"])
faiss.omp_set_num_threads(int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "32")))
sample = X[np.sort(rng.choice(N, 500_000, replace=False))]
res["spatial"] = {}
def nearest(A, cent, bs=200_000):
    cn = (cent * cent).sum(1); out = np.empty(len(A), dtype=np.int32)
    for s in range(0, len(A), bs):
        out[s:s + bs] = np.argmin(cn[None, :] - 2 * (A[s:s + bs] @ cent.T), axis=1)
    return out
def kmeans_np(S, C, iters):
    cent = S[rng.choice(len(S), C, replace=False)].copy()
    for it in range(iters):
        a = nearest(S, cent); o = np.argsort(a, kind="stable"); a_s = a[o]
        starts = np.searchsorted(a_s, np.arange(C)); cnt = np.bincount(a, minlength=C)
        sums = np.add.reduceat(S[o], np.minimum(starts, len(S) - 1), axis=0)
        ok = cnt > 0; cent[ok] = sums[ok] / cnt[ok, None]
        empty = np.where(~ok)[0]
        if empty.size: cent[empty] = S[rng.choice(len(S), empty.size, replace=False)]
    log(f"kmeans C={C} done, empty last iter {empty.size}")
    return cent
for C in (1024, 4096):
    log("kmeans C", C)
    cent = kmeans_np(sample[: min(len(sample), 200 * C)], C, 12)
    assign = nearest(X, cent)
    log("assigned C", C)
    size = np.bincount(assign, minlength=C)
    ca = assign[gt10]                               # nq x 10 cluster ids of true neighbors
    ndist = np.array([np.unique(r).size for r in ca])
    top1 = np.array([np.bincount(r).max() for r in ca])
    # baseline: 10 random base vectors per query
    rnd = assign[rng.integers(0, N, size=(nq, K))]
    ndist_rnd = np.array([np.unique(r).size for r in rnd])
    # cluster of the query itself (nearest centroid, L2) -- does it contain the NN?
    qa = nearest(Q, cent)
    frac_in_qcluster = (ca == qa[:, None]).mean()
    # cluster-level tier: rank clusters by train-neighbor mass
    w_train = np.bincount(ca[train].ravel(), minlength=C).astype(float)
    order = np.argsort(-w_train, kind="stable")
    cumN = np.cumsum(size[order]) / N
    in_rank = np.empty(C, dtype=np.int64); in_rank[order] = np.arange(C)
    test_rank = in_rank[ca[test]]                    # rank of each test-neighbor cluster
    maxrank = test_rank.max(1)
    curve = []
    for m in range(1, C + 1):
        curve.append((cumN[m - 1], float((test_rank < m).mean()), float((maxrank < m).mean()),
                      float(((test_rank < m).sum(1) >= 9).mean())))
    curve = np.array(curve)  # cols: B/N, slot coverage, frac test queries recall@10=1, recall>=0.9
    # uniform baseline: rank clusters by size only uniform (random order)
    ro = rng.permutation(C); cumN_r = np.cumsum(size[ro]) / N
    in_r = np.empty(C, dtype=np.int64); in_r[ro] = np.arange(C)
    tr_r = in_r[ca[test]]
    curve_r = np.array([(cumN_r[m - 1], float((tr_r < m).mean()), float((tr_r.max(1) < m).mean())) for m in range(1, C + 1)])
    np.savez(OUT / f"msmarco_spatial_C{C}.npz", curve=curve, curve_rand=curve_r, size=size, ndist=ndist,
             ndist_rnd=ndist_rnd, top1=top1, w_train=w_train)
    def at(fr):  # coverage when T1 holds <= fr of D
        i = np.searchsorted(curve[:, 0], fr, side="right") - 1
        return dict(BoverN=float(curve[i, 0]), slot_cov=float(curve[i, 1]), q_full=float(curve[i, 2]), q_ge09=float(curve[i, 3]))
    r = dict(C=C, mean_distinct_clusters_per_query=float(ndist.mean()),
             median_distinct_clusters=float(np.median(ndist)),
             mean_distinct_clusters_random10=float(ndist_rnd.mean()),
             mean_max_neighbors_in_one_cluster=float(top1.mean()),
             frac_queries_all10_in_one_cluster=float((ndist == 1).mean()),
             frac_queries_le2_clusters=float((ndist <= 2).mean()),
             frac_nn_in_query_nearest_cluster=float(frac_in_qcluster),
             clusters_touched_by_any_nn=int((np.bincount(ca.ravel(), minlength=C) > 0).sum()),
             tier_at={f"{p}": at(p) for p in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)})
    res["spatial"][str(C)] = r
    log("C", C, json.dumps(r))
    json.dump(res, open(OUT / "msmarco_skew.json", "w"), indent=1)
log("done")
