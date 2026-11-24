#!/usr/bin/env python3
"""Skewed DEEP100M query workloads (8-node final experiment, 2026-11-10).

A workload = 10k queries: p*10k "hot" queries (noisy copies of H seed queries taken from
query.public.10K) + (1-p)*10k original queries, shuffled. Hot copy = normalize(seed + noise),
||noise|| = alpha * r10(seed), r10 = L2 distance of the seed's 10th true NN.

Subcommands
  calib     brute-force GT of a few copies for several alphas; reports how concentrated the copies'
            NNs are on one GP-ANN shard (k=8, o=0.2) and how different copies are from each other.
  hot       generate the 10k hot copies of one seed set (H seeds) + brute-force top-100 GT (sharded).
  assemble  build the workloads from hot sets + originals and export every system's formats.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

DEEP = Path("/gpfs/scratch/delgaudio/vectors/DEEP")
BASE = DEEP / "base.100M.fbin"
QUERY = DEEP / "query.public.10K.fbin"
GT = DEEP / "GT.100M.bin"
GP_PART = DEEP / "base.100M.partition.k=8.GP.o=0.2"
D, NQ, K = 96, 10000, 100
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ IO
def read_fbin(p):
    n, d = np.fromfile(p, dtype=np.int32, count=2)
    return np.fromfile(p, dtype=np.float32, offset=8).reshape(n, d)


def write_fbin(p, x):
    with open(p, "wb") as f:
        np.array(x.shape, dtype=np.int32).tofile(f)
        x.astype(np.float32).tofile(f)


def write_fvecs(p, x):
    n, d = x.shape
    out = np.empty((n, d + 1), dtype=np.float32)
    out[:, 0] = np.array([d], dtype=np.int32).view(np.float32)[0]
    out[:, 1:] = x
    out.tofile(p)


def read_gt(p):  # bigann: n,k int32 | ids int32[n*k] | sq-L2 dists float32[n*k]
    n, k = np.fromfile(p, dtype=np.int32, count=2)
    ids = np.fromfile(p, dtype=np.int32, count=n * k, offset=8).reshape(n, k)
    dis = np.fromfile(p, dtype=np.float32, count=n * k, offset=8 + 4 * n * k).reshape(n, k)
    return ids, dis


def write_gt(p, ids, dis):
    with open(p, "wb") as f:
        np.array(ids.shape, dtype=np.int32).tofile(f)
        ids.astype(np.int32).tofile(f)
        dis.astype(np.float32).tofile(f)


# ------------------------------------------------------------------ brute force
def brute_force(Q, k=K, chunk=1_000_000, qblock=128, threads=None):
    """Exact top-k by squared L2 over base.100M (memmap, base-chunk outer loop)."""
    threads = threads or os.cpu_count()
    base = np.memmap(BASE, dtype=np.float32, mode="r", offset=8).reshape(-1, D)
    nb = base.shape[0]
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    qn = (Q * Q).sum(1)
    best_d = np.full((len(Q), k), np.inf, dtype=np.float32)
    best_i = np.full((len(Q), k), -1, dtype=np.int64)
    blocks = [(s, min(s + qblock, len(Q))) for s in range(0, len(Q), qblock)]

    def work(se, B, bn, off):
        s, e = se
        sc = bn[None, :] - 2.0 * (Q[s:e] @ B.T)  # + qn later
        kk = min(k, sc.shape[1])
        p = np.argpartition(sc, kk - 1, axis=1)[:, :kk]
        d = np.take_along_axis(sc, p, 1)
        cd = np.concatenate([best_d[s:e], d], 1)
        ci = np.concatenate([best_i[s:e], p + off], 1)
        o = np.argpartition(cd, k - 1, axis=1)[:, :k]
        best_d[s:e] = np.take_along_axis(cd, o, 1)
        best_i[s:e] = np.take_along_axis(ci, o, 1)

    with ThreadPoolExecutor(threads) as ex:
        for off in range(0, nb, chunk):
            B = np.ascontiguousarray(base[off:off + chunk])
            bn = (B * B).sum(1)
            list(ex.map(lambda se: work(se, B, bn, off), blocks))
            if (off // chunk) % 10 == 0:
                log(f"  bf {off/nb*100:5.1f}%")
    o = np.argsort(best_d, 1)
    best_d = np.take_along_axis(best_d, o, 1) + qn[:, None]
    best_i = np.take_along_axis(best_i, o, 1)
    return best_i, np.maximum(best_d, 0)


# ------------------------------------------------------------------ hot copies
def make_copies(seeds_vec, r10, alpha, n, rng):
    """n copies, assigned round-robin to seeds (so any prefix is balanced across seeds)."""
    H = len(seeds_vec)
    owner = np.arange(n) % H
    g = rng.standard_normal((n, D)).astype(np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    x = seeds_vec[owner] + g * (alpha * r10[owner])[:, None]
    x /= np.linalg.norm(x, axis=1, keepdims=True)  # DEEP vectors are unit-norm
    return x.astype(np.float32), owner


def pick_seeds(rng, H, exclude=()):
    pool = np.setdiff1d(np.arange(NQ), np.array(list(exclude), dtype=np.int64))
    return np.sort(rng.choice(pool, H, replace=False))


def load_gp_shards():
    """uint8 bitmask per base vector: bit s set if the vector is in GP shard s (overlap allowed)."""
    mask = np.zeros(100_000_000, dtype=np.uint8)
    with open(GP_PART) as f:
        for s, line in enumerate(f):
            ids = np.array(line.split(), dtype=np.int64)
            mask[ids] |= np.uint8(1 << s)
            log(f"  GP shard {s}: {len(ids)} ids")
    return mask


# ------------------------------------------------------------------ subcommands
def cmd_calib(a):
    rng = np.random.default_rng(a.rng)
    Qo = read_fbin(QUERY)
    gti, gtd = read_gt(GT)
    seeds = pick_seeds(rng, a.nseeds)
    r10 = np.sqrt(gtd[:, 9])
    alphas = [float(x) for x in a.alphas.split(",")]
    allq, meta = [], []
    for al in alphas:
        x, owner = make_copies(Qo[seeds], r10[seeds], al, a.copies * len(seeds), rng)
        allq.append(x)
        meta += [(al, seeds[o]) for o in owner]
    Q = np.concatenate(allq)
    log(f"calib: {len(Q)} queries, seeds={seeds.tolist()}, alphas={alphas}")
    ids, dis = brute_force(np.concatenate([Q, Qo[seeds]]), k=10)
    chk_i, chk_d = ids[len(Q):], dis[len(Q):]
    ids, dis = ids[:len(Q)], dis[:len(Q)]
    rec = np.mean([len(set(chk_i[j]) & set(gti[s, :10])) / 10 for j, s in enumerate(seeds)])
    log(f"brute-force check vs GT.100M.bin: recall@10={rec:.3f} "
        f"max|d-gt|={np.abs(chk_d - gtd[seeds, :10]).max():.2e}")
    mask = load_gp_shards()
    seedmask = {s: mask[gti[s, :10]] for s in seeds}
    rows = []
    for al in alphas:
        for s in seeds:
            sel = [i for i, (aa, ss) in enumerate(meta) if aa == al and ss == s]
            nn = ids[sel]
            # dominant shard of the seed's own top-10
            cnt = [sum(bool(m & (1 << b)) for m in seedmask[s]) for b in range(8)]
            dom = int(np.argmax(cnt))
            in_dom = float(np.mean(mask[nn] & (1 << dom) > 0))
            seed_set = set(gti[s, :100].tolist())
            ov_seed = float(np.mean([len(seed_set & set(r.tolist())) / 10 for r in nn]))
            # pairwise top-10 overlap between copies (1.0 = duplicates)
            sets = [set(r.tolist()) for r in nn[:50]]
            pw = [len(sets[i] & sets[j]) / 10 for i in range(len(sets)) for j in range(i + 1, len(sets))]
            dist_seed = float(np.mean(np.linalg.norm(Q[sel] - Qo[s], axis=1)))
            rows.append(dict(alpha=al, seed=int(s), r10=float(r10[s]), dist_to_seed=dist_seed,
                             top10_in_seed_top100=ov_seed, pairwise_top10_overlap=float(np.mean(pw)),
                             nn_in_dominant_gp_shard=in_dom, dominant_shard=dom))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"{'alpha':>5} {'seed':>5} {'r10':>5} {'d(seed)':>7} {'inSeed100':>9} {'pairwise':>8} {'domShard':>8}")
    for r in rows:
        print(f"{r['alpha']:5.2f} {r['seed']:5d} {r['r10']:5.3f} {r['dist_to_seed']:7.3f} "
              f"{r['top10_in_seed_top100']:9.2f} {r['pairwise_top10_overlap']:8.2f} {r['nn_in_dominant_gp_shard']:8.2f}")


def cmd_hot(a):
    """Hot set `name`: H seeds, 10k copies, top-100 GT. --part i/n computes GT for one slice."""
    out = Path(a.outdir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "spec.json"
    if not spec.exists():  # first call creates seeds + copies deterministically
        rng = np.random.default_rng(a.rng)
        Qo = read_fbin(QUERY)
        _, gtd = read_gt(GT)
        exclude = json.loads(a.exclude) if a.exclude else []
        seeds = pick_seeds(rng, a.H, exclude)
        x, owner = make_copies(Qo[seeds], np.sqrt(gtd[seeds, 9]), a.alpha, NQ, rng)
        write_fbin(out / "hot.fbin", x)
        np.save(out / "owner.npy", seeds[owner])
        spec.write_text(json.dumps(dict(name=a.name, H=a.H, alpha=a.alpha, rng=a.rng,
                                        seeds=seeds.tolist()), indent=1))
        log(f"{a.name}: seeds {seeds.tolist()}")
    if a.part:
        i, n = map(int, a.part.split("/"))
        X = read_fbin(out / "hot.fbin")
        sl = np.array_split(np.arange(len(X)), n)[i]
        log(f"{a.name}: GT part {i}/{n} ({len(sl)} queries)")
        ids, dis = brute_force(X[sl])
        np.save(out / f"gt_ids_{i}.npy", ids.astype(np.int32))
        np.save(out / f"gt_dis_{i}.npy", dis)
        log("done")


# ------------------------------------------------------------------ region workloads (v2)
LEARN = DEEP / "learn.350M.fbin"


def sample_rows(path, n, nchunks, rng):
    """n rows from an fbin as nchunks contiguous blocks at random offsets (fast on GPFS)."""
    x = np.memmap(path, dtype=np.float32, mode="r", offset=8).reshape(-1, D)
    per = n // nchunks
    starts = np.sort(rng.choice(len(x) - per, nchunks, replace=False))
    ids = np.concatenate([np.arange(s, s + per) for s in starts])
    return np.concatenate([np.asarray(x[s:s + per]) for s in starts]), ids


def assign(X, C, bs=200_000):
    cn = (C * C).sum(1)
    return np.concatenate([np.argmin(cn[None, :] - 2.0 * X[i:i + bs] @ C.T, 1) for i in range(0, len(X), bs)])


def kmeans(X, k, iters, rng):
    sub = X[rng.choice(len(X), min(len(X), 50_000), replace=False)]  # k-means++ init on a subsample
    C = sub[rng.choice(len(sub), 1)]
    for _ in range(k - 1):
        d = np.maximum(np.min((sub * sub).sum(1)[:, None] - 2 * sub @ C.T + (C * C).sum(1)[None], 1), 0)
        C = np.vstack([C, sub[rng.choice(len(sub), p=d / d.sum())]])
    for it in range(iters):
        a = assign(X, C)
        newC = np.stack([X[a == c].mean(0) if np.any(a == c) else C[c] for c in range(k)])
        shift = np.abs(newC - C).max()
        C = newC
        log(f"  kmeans it {it}: sizes min/max {np.bincount(a, minlength=k).min()}/{np.bincount(a, minlength=k).max()} shift {shift:.2e}")
    return C.astype(np.float32)


def cmd_region(a):
    """k-means regions on base, hot query sets sampled from the learn set inside hot clusters."""
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.rng)
    cpath = out / f"centroids_C{a.C}.npy"
    Xb, _ = sample_rows(BASE, a.ntrain, 200, rng)
    if cpath.exists():
        C = np.load(cpath)
    else:
        log(f"kmeans C={a.C} on {len(Xb)} base vectors")
        C = kmeans(Xb, a.C, a.iters, rng)
        np.save(cpath, C)
    Xl, lid = sample_rows(LEARN, a.nlearn, 400, rng)
    al = assign(Xl, C)
    aq = assign(read_fbin(QUERY), C)
    learn_cnt = np.bincount(al, minlength=a.C)
    stats = dict(C=a.C, base_frac=(np.bincount(assign(Xb, C), minlength=a.C) / len(Xb)).round(4).tolist(),
                 learn_per_cluster=learn_cnt.tolist(),
                 query_per_cluster=np.bincount(aq, minlength=a.C).tolist())
    log(f"base fraction per cluster: {stats['base_frac']}")
    log(f"learn per cluster: {learn_cnt.tolist()}")
    perm = rng.permutation(a.C)
    sets, i = {}, 0
    for name, h in [("r1a", 1), ("r1b", 1), ("r1c", 1), ("r2", 2), ("r4", 4)]:
        sets[name] = sorted(perm[i:i + h].tolist())
        i += h
    for name, cl in sets.items():
        per = NQ // len(cl)
        picks = []
        for c in cl:
            cand = np.where(al == c)[0]
            if len(cand) < per:
                sys.exit(f"cluster {c}: only {len(cand)} learn vectors, need {per}; raise --nlearn")
            picks.append(rng.choice(cand, per, replace=False))
        order = np.stack(picks, 1).reshape(-1)  # round-robin over hot clusters: prefixes stay balanced
        d = out / "hot" / name
        d.mkdir(parents=True, exist_ok=True)
        write_fbin(d / "hot.fbin", Xl[order])
        np.save(d / "owner.npy", al[order])
        np.save(d / "learn_ids.npy", lid[order])
        (d / "spec.json").write_text(json.dumps(dict(name=name, kind="region", H=len(cl), clusters=cl,
                                                     alpha=None, seeds=[], C=a.C, rng=a.rng), indent=1))
        log(f"{name}: clusters {cl} ({per}/cluster)")
    (out / "regions.json").write_text(json.dumps(dict(stats, hot_sets=sets), indent=1))


def cmd_gt(a):
    d = Path(a.outdir) / a.name
    i, n = map(int, a.part.split("/"))
    X = read_fbin(d / "hot.fbin")
    sl = np.array_split(np.arange(len(X)), n)[i]
    log(f"{a.name}: GT part {i}/{n} ({len(sl)} queries)")
    ids, dis = brute_force(X[sl])
    np.save(d / f"gt_ids_{i}.npy", ids.astype(np.int32))
    np.save(d / f"gt_dis_{i}.npy", dis)
    log("done")


def cmd_check(a):
    """Diversity, difficulty and concentration of each hot set vs the original queries."""
    _, gtd = read_gt(GT)
    gti, _ = read_gt(GT)
    C = np.load(Path(a.outdir) / f"centroids_C{a.C}.npy")
    aq = assign(read_fbin(QUERY), C)
    mask = load_gp_shards()
    rng = np.random.default_rng(0)

    def shard_share(ids10):
        cnt = np.array([np.sum(mask[ids10.ravel()] & (1 << s) > 0) for s in range(8)], dtype=float)
        cnt /= cnt.sum()
        return np.sort(cnt)[::-1]

    def profile(name, ids, dis, extra=""):
        d = np.sqrt(np.maximum(dis, 0))
        sub = rng.choice(len(ids), min(200, len(ids)), replace=False)
        sets = [set(ids[j, :10].tolist()) for j in sub]
        pw = np.mean([len(sets[x] & sets[y]) / 10 for x in range(len(sets)) for y in range(x + 1, len(sets))])
        sh = shard_share(ids[:, :10])
        print(f"{name:22s} d1={np.median(d[:, 0]):.3f} d10={np.median(d[:, 9]):.3f} "
              f"d10/d1={np.median(d[:, 9] / np.maximum(d[:, 0], 1e-9)):.3f} pairwise={pw:.3f} "
              f"top1shard={sh[0]:.2f} top2={sh[:2].sum():.2f} top4={sh[:4].sum():.2f} dup={np.mean(d[:, 0] < 1e-4):.3f} {extra}")

    profile("original (uniform)", gti, gtd)
    for name in ["r1a", "r1b", "r1c", "r2", "r4"]:
        X, ids, dis, spec = load_hot(Path(a.outdir) / "hot" / name)
        inq = np.isin(aq, spec["clusters"])
        profile(f"orig in {spec['clusters']}", gti[inq], gtd[inq], f"(n={inq.sum()})")
        profile(f"{name} hot", ids, dis)


def load_hot(d):
    d = Path(d)
    X = read_fbin(d / "hot.fbin")
    parts = sorted(d.glob("gt_ids_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
    ids = np.concatenate([np.load(p) for p in parts])
    dis = np.concatenate([np.load(str(p).replace("gt_ids", "gt_dis")) for p in parts])
    assert len(ids) == len(X), f"{d}: GT incomplete ({len(ids)}/{len(X)})"
    return X, ids, dis, json.loads((d / "spec.json").read_text())


def export(wdir, name, Q, ids, dis, harmony_bench):
    """Write one workload in every system's input format."""
    wdir.mkdir(parents=True, exist_ok=True)
    n = len(Q)
    write_fbin(wdir / "query.fbin", Q)                                  # GP-ANN, DA, CoTra
    write_gt(wdir / "GT.bin", ids, dis)                                 # GP-ANN, DA (bigann, k=100)
    with open(wdir / "GT_cotra.bin", "wb") as f:                        # CoTra: header + ids only
        np.array([n, K], dtype=np.int32).tofile(f)
        ids.astype(np.int32).tofile(f)
    write_fvecs(wdir / "query.fvecs", Q)                                # Pyramid
    with open(wdir / "gt_k10.lshbox", "w") as f:                        # Pyramid (text, k=10)
        f.write(f"{n}\t10\n")
        for q in range(n):
            f.write(f"{q}\t" + "\t".join(f"{int(ids[q, j])}\t{float(dis[q, j])!r}" for j in range(10)) + "\n")
    # Harmony: ${BENCH}/${DATASET}/{origin,index,result}
    ds = f"DEEP100M_{name}"
    hd = Path(harmony_bench) / ds
    (hd / "origin").mkdir(parents=True, exist_ok=True)
    (hd / "result").mkdir(parents=True, exist_ok=True)
    src = Path(harmony_bench) / "DEEP100M"
    for link, target in [(hd / "origin" / f"{ds}_base.fvecs", src / "origin" / "DEEP100M_base.fvecs"),
                         (hd / "index", src / "index")]:
        if not link.is_symlink():
            link.symlink_to(target)
    write_fvecs(hd / "origin" / f"{ds}_query.fvecs", Q)
    for k in (10, 100):
        with open(hd / "result" / f"groundtruth_{k}.bin", "wb") as f:
            ids[:, :k].astype(np.int64).tofile(f)
            dis[:, :k].astype(np.float32).tofile(f)
    (wdir / "harmony_dataset.txt").write_text(ds + "\n")


def cmd_assemble(a):
    Qo = read_fbin(QUERY)
    gti, gtd = read_gt(GT)
    rng = np.random.default_rng(a.rng)
    plan = json.loads(Path(a.plan).read_text())  # [{"name":..., "hot":..., "p":...}, ...]
    manifest = []
    for w in plan:
        X, hi, hd_, spec = load_hot(Path(a.hotdir) / w["hot"])
        nh = int(round(w["p"] * NQ))
        no = NQ - nh
        orig = np.sort(rng.choice(np.setdiff1d(np.arange(NQ), spec["seeds"]), no, replace=False))
        Q = np.concatenate([X[:nh], Qo[orig]])
        ids = np.concatenate([hi[:nh], gti[orig]])
        dis = np.concatenate([hd_[:nh], gtd[orig]])
        src = np.concatenate([np.full(nh, -1), orig])  # -1 = hot copy, else original query id
        perm = rng.permutation(NQ)
        Q, ids, dis, src = Q[perm], ids[perm], dis[perm], src[perm]
        wdir = Path(a.outdir) / w["name"]
        export(wdir, w["name"], Q, ids, dis, a.harmony_bench)
        np.save(wdir / "source.npy", src)
        info = dict(w, H=spec["H"], alpha=spec["alpha"], seeds=spec["seeds"], n_hot=nh, n_orig=no)
        (wdir / "info.json").write_text(json.dumps(info, indent=1))
        manifest.append(info)
        log(f"{w['name']}: H={spec['H']} p={w['p']} hot={nh} orig={no}")
    (Path(a.outdir) / "manifest.json").write_text(json.dumps(manifest, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calib")
    c.add_argument("--nseeds", type=int, default=3)
    c.add_argument("--copies", type=int, default=100)
    c.add_argument("--alphas", default="0.25,0.5,1,1.5,2")
    c.add_argument("--rng", type=int, default=7)
    c.add_argument("--out", required=True)
    h = sub.add_parser("hot")
    h.add_argument("--name", required=True)
    h.add_argument("--H", type=int, required=True)
    h.add_argument("--alpha", type=float, required=True)
    h.add_argument("--rng", type=int, required=True)
    h.add_argument("--exclude", default="")
    h.add_argument("--part", default="")
    h.add_argument("--outdir", required=True)
    s = sub.add_parser("assemble")
    s.add_argument("--plan", required=True)
    s.add_argument("--hotdir", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--harmony-bench", default="/gpfs/scratch/delgaudio/vectors")
    s.add_argument("--rng", type=int, default=11)
    r = sub.add_parser("region")
    r.add_argument("--outdir", required=True)
    r.add_argument("--C", type=int, default=16)
    r.add_argument("--ntrain", type=int, default=1_000_000)
    r.add_argument("--iters", type=int, default=25)
    r.add_argument("--nlearn", type=int, default=4_000_000)
    r.add_argument("--rng", type=int, default=2026)
    g = sub.add_parser("gt")
    g.add_argument("--name", required=True)
    g.add_argument("--part", required=True)
    g.add_argument("--outdir", required=True)
    k = sub.add_parser("check")
    k.add_argument("--outdir", required=True)
    k.add_argument("--C", type=int, default=16)
    a = ap.parse_args()
    dict(calib=cmd_calib, hot=cmd_hot, assemble=cmd_assemble, region=cmd_region, gt=cmd_gt,
         check=cmd_check)[a.cmd](a)


if __name__ == "__main__":
    main()
