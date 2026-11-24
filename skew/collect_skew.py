#!/usr/bin/env python3
"""Collect skewed-workload runs (runs/full/<W>/) into one CSV and an iso-recall comparison.

Output:
  docs/skew_curves.csv   system,workload,config,recall10,qps,imbalance,source
  docs/skew_isorecall.csv  system,workload,target,qps,qps_uniform,ratio  (Pareto + log-linear interpolation)
Uniform reference: docs/final_curves.csv (DEEP100M-8n); workload "uniform_rerun" = fresh re-runs in runs/full/uniform.
"""
import csv, glob, math, os, re, sys

RUNS = "/gpfs/scratch/delgaudio/skew/runs/full"
DOCS = "/gpfs/users/delgaudio/exp_eval/docs"
NQ = 10000
TARGETS = [0.80, 0.85, 0.90, 0.95, 0.99]


def rd(p):
    with open(p, errors="replace") as f:
        return f.read()


def gp(w, d):
    out = []
    for p in sorted(glob.glob(f"{d}/gp_*.log")):
        t = rd(p)
        np_ = re.search(r"num_probes=(\d+) ef_search=(\d+)", t)
        q = re.search(r"Query QPS ([0-9.]+)", t)
        r = re.search(r"Mean recall@10 \(ID overlap\) ([0-9.]+)", t)
        l = re.search(r"max=(\d+) avg=(\d+)", t)
        if np_ and q and r:
            imb = int(l.group(1)) / int(l.group(2)) if l else ""
            out.append(("GP-ANN*", w, f"o=0.2 nprobe={np_.group(1)} ef={np_.group(2)}", float(r.group(1)),
                        float(q.group(1)), imb, os.path.basename(p)))
    return out


def pyr(w, d):
    out = []
    for p in sorted(glob.glob(f"{d}/pyramid_v2_sweep_*[0-9].tsv")):
        log = p.replace("pyramid_v2_sweep_skew_" + w + "_", "pyr_").replace(".tsv", ".log")
        wk = re.findall(r"avg_per_slave=([0-9.]+) min=\d+ max=(\d+)", rd(log)) if os.path.exists(log) else []
        rows = [r for r in csv.DictReader(open(p), delimiter="\t") if r.get("rc") == "0"]
        for i, r in enumerate(rows):
            imb = int(wk[i][1]) / float(wk[i][0]) if i < len(wk) else ""
            out.append(("Pyramid*", w, f"K={r['K']} ef={r['ef']}", float(r["recall_at_10"]), float(r["qps"]), imb,
                        os.path.basename(p)))
    return out


def hmy(w, d):
    out = []
    for p in sorted(glob.glob(f"{d}/hmy_*.out")):
        t = rd(p)
        q = re.search(r" query_time: ([0-9.]+)", t)
        r = re.search(r"^recall: ([0-9.]+)", t, re.M)
        n = re.search(r"^nprobe:(\d+)", t, re.M)
        if not (q and r and n):
            continue
        cv = [int(x) for x in re.findall(r"TeamQueryStats: team=\d+ .*compare_vectors=(\d+)", t)]
        imb = max(cv) / (sum(cv) / len(cv)) if cv else ""
        out.append(("Harmony", w, f"nprobe={n.group(1)}", float(r.group(1)), NQ / float(q.group(1)), imb,
                    os.path.basename(p)))
    return out


def cotra(w, d):
    out = []
    for p in sorted(glob.glob(f"{d}/cotra_*.out")):
        t = rd(p)
        load = {int(e): [int(x) for x in re.findall(r"m\d=(\d+)", rest)]
                for e, rest in re.findall(r"^(\d+)\tcore_load\t(.*)$", t, re.M)}
        for e, r, q in re.findall(r"^(\d+) \t ([0-9.]+) \t 0\.00 \t ([0-9.]+)", t, re.M):
            v = load.get(int(e), [])
            imb = max(v) / (sum(v) / len(v)) if v else ""
            out.append(("CoTra", w, f"ef={e}", float(r), float(q), imb, os.path.basename(p)))
    return out


def da(w, d):
    out = []
    for p in sorted(glob.glob(f"{d}/da_*.log")):
        t = rd(p)
        imbs = re.findall(r"score_imbalance\(mean max/mean per round\)=([0-9.]+)", t)
        for i, m in enumerate(re.finditer(r"SWEEP_SUMMARY k=10 BW=(\d+) H=(\d+) L=(\d+) .*?recall@10=([0-9.]+) .*?qps=([0-9.]+)", t)):
            bw, h, l, r, q = m.groups()
            out.append(("DistributedANN*", w, f"BW={bw} H={h} L={l}", float(r), float(q),
                        float(imbs[i]) if i < len(imbs) else "", os.path.basename(p)))
    return out


def pareto(pts):
    pts = sorted(pts, key=lambda x: -x[0])  # recall desc
    front, best = [], -1
    for r, q in pts:
        if q > best:
            front.append((r, q)); best = q
    return sorted(front)


def iso(pts, t):
    f = pareto(pts)
    if not f or f[-1][0] < t or min(p[0] for p in pts) > t:
        return None  # target not bracketed by measured points -> not covered
    if f[0][0] >= t:
        return f[0][1]  # fastest point already has recall >= t (lower-recall points are dominated)
    for (r0, q0), (r1, q1) in zip(f, f[1:]):
        if r0 <= t <= r1:
            if r1 == r0:
                return max(q0, q1)
            a = (t - r0) / (r1 - r0)
            return math.exp(math.log(q0) + a * (math.log(q1) - math.log(q0)))
    return f[-1][1] if f[-1][0] == t else None


def main():
    rows = []
    for d in sorted(glob.glob(f"{RUNS}/*/")):
        w = os.path.basename(d.rstrip("/"))
        w = "uniform_rerun" if w == "uniform" else w
        for fn in (gp, pyr, hmy, cotra, da):
            rows += fn(w, d)
    for r in csv.DictReader(open(f"{DOCS}/final_curves.csv")):
        if r["setup"] == "DEEP100M-8n" and r["recall10"]:
            rows.append((r["system"], "uniform", r["config"], float(r["recall10"]), float(r["qps"]), "", r["source"]))
    with open(f"{DOCS}/skew_curves.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["system", "workload", "config", "recall10", "qps", "imbalance", "source"])
        cw.writerows(rows)
    curves = {}
    for s, w, c, r, q, *_ in rows:
        curves.setdefault((s, w), []).append((r, q))
    iso_rows = []
    for (s, w), pts in sorted(curves.items()):
        for t in TARGETS:
            q, qu = iso(pts, t), iso(curves.get((s, "uniform"), []), t)
            iso_rows.append((s, w, t, q and round(q, 1), qu and round(qu, 1), round(q / qu, 3) if q and qu else ""))
    with open(f"{DOCS}/skew_isorecall.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["system", "workload", "target", "qps", "qps_uniform", "ratio"])
        cw.writerows(iso_rows)
    # console summary
    ws = sorted({w for _, w in curves})
    print("ratio QPS(skew)/QPS(uniform) at iso-recall ('-' = target not covered)")
    for s in sorted({s for s, _ in curves}):
        print(f"\n{s}")
        print("  " + "workload".ljust(14) + "".join(f"{t:>8}" for t in TARGETS) + "   pts  rec[min,max]  imb(max)")
        for w in ws:
            if (s, w) not in curves:
                continue
            pts = curves[(s, w)]
            imbs = [r[5] for r in rows if r[0] == s and r[1] == w and r[5] != ""]
            vals = []
            for t in TARGETS:
                q, qu = iso(pts, t), iso(curves.get((s, "uniform"), []), t)
                vals.append(f"{q / qu:8.2f}" if q and qu else (f"{q:8.0f}" if q and w == "uniform" else "       -"))
            print("  " + w.ljust(14) + "".join(vals) +
                  f"   {len(pts):3d}  [{min(p[0] for p in pts):.3f},{max(p[0] for p in pts):.3f}]"
                  + (f"  {max(imbs):.2f}" if imbs else ""))


if __name__ == "__main__":
    main()
