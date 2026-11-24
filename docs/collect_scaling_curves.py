import csv, glob, os, re, sys
from collections import defaultdict
SCR="/gpfs/scratch/delgaudio"; EXP="/gpfs/users/delgaudio/exp_eval"
pts=defaultdict(dict)  # (setup,system) -> config -> (recall,qps)
V=f"{SCR}/vectors"
gp={"DEEP10M-8n":f"{V}/DEEP/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-deep10M-k8_*.log",
    "TTI10M-8n":f"{V}/TTI/new_experiments_gpann_fixed/curve_parallel/logs/gp-curve-tti10M-k8_*.log",
    "DEEP100M-4n":f"{V}/DEEP/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-deep100M-k4_*.log",
    "DEEP100M-6n":f"{V}/DEEP/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-deep100M-k6_*.log",
    "TTI100M-4n":f"{V}/TTI/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-tti100M-k4_*.log",
    "TTI100M-6n":f"{V}/TTI/distbench_100M_ef_nprobe/curve_parallel/logs/gp-curve-tti100M-k6_*.log"}
for s,p in gp.items():
    for f in glob.glob(p):
        t=open(f,errors="replace").read()
        c=re.search(r"nprobe=(\d+) ef=(\d+)",t); r=re.search(r"recall@10 \(ID overlap\) ([0-9.]+)",t); q=re.search(r"Query QPS ([0-9.]+)",t)
        if c and r and q: pts[(s,"GP-ANN*")][c.group(0)]=(float(r[1]),float(q[1]))
for s,tag in {"DEEP10M-8n":"deep10m_8n","TTI10M-8n":"tti10m_8n","DEEP100M-4n":"deep100m_4n","DEEP100M-6n":"deep100m_6n","TTI100M-4n":"tti100m_4n","TTI100M-6n":"tti100m_6n"}.items():
    for f in glob.glob(f"{SCR}/dhnsw/logs/sweep_v2/pyramid_v2_sweep_{tag}_*[0-9].tsv"):
        for r in csv.DictReader(open(f),delimiter="\t"):
            if r["rc"]=="0" and r["recall_at_10"] not in("","NA") and r["qps"] not in("","NA"):
                pts[(s,"Pyramid*")][f"K={r['K']} ef={r['ef']}"]=(float(r["recall_at_10"]),float(r["qps"]))
for s,tag in {"DEEP10M-8n":"deep10m-w8","TTI10M-8n":"tti10m-w8","DEEP100M-4n":"deep100m-w4-*","DEEP100M-6n":"deep100m-w6-*","TTI100M-4n":"tti100m-w4-*","TTI100M-6n":"tti100m-w6-*","TTI100M-8n":"tti100m-w8-*"}.items():
    for f in glob.glob(f"{SCR}/dann/logs/scaling_queries/da-q-{tag}_*.log"):
        for line in open(f,errors="replace"):
            m=re.search(r"SWEEP_SUMMARY k=10 BW=(\d+) H=(\d+) L=(\d+) .*recall@10=([0-9.]+) elapsed_s=[0-9.]+ qps=([0-9.]+)",line)
            if m: pts[(s,"DA*")][f"BW={m[1]} H={m[2]} L={m[3]}"]=(float(m[4]),float(m[5]))
L=f"{EXP}/CoTra/script_slurm/logs"
for s,pat in {"DEEP10M-8n":"scala_anns_deep10M_8p-*.out","TTI10M-8n":"scala_anns_tti10M_8p-*.out","DEEP100M-4n":"scala_anns_deep100M_4p-*.out","DEEP100M-6n":"scala_anns_6p-cotra6_deep100_k10-6975613.out","TTI100M-4n":"scala_anns_tti100M_4p-*.out","TTI100M-6n":"scala_anns_tti100M_6p-*.out"}.items():
    for f in glob.glob(f"{L}/{pat}"):
        for line in open(f,errors="replace"):
            m=re.match(r"^\s*(\d+)\s+([0-9.]+)\s+0\.00\s+([0-9.]+)\s*$",line)
            if m: pts[(s,"CoTra")][f"ef={m[1]}"]=(float(m[2]),float(m[3]))
for s,pre in {"DEEP10M-8n":"deep10m_8n","TTI10M-8n":"tti10m_8n","DEEP100M-4n":"deep100m_4n","DEEP100M-6n":"deep100m_6n","TTI100M-4n":"tti100m_4n","TTI100M-6n":"tti100m_6n","TTI100M-8n":"tti100m_8n"}.items():
    for f in sorted(glob.glob(f"{EXP}/Harmony/slurm_script/log/harmony_{pre}_omp16_*.out"),key=os.path.getmtime):
        t=open(f,errors="replace").read()
        h=re.search(r"HARMONY_K=(\d+) HARMONY_NPROBE=(\d+)",t); r=re.search(r"recall: ([0-9.]+)",t); qt=re.search(r" query_time: ([0-9.]+)",t); nq=re.search(r"nq:(\d+)",t)
        if h and h[1]=="10" and r and qt and float(qt[1])>0:
            pts[(s,"Harmony")][f"nprobe={h[2]}"]=(float(r[1]),(int(nq[1]) if nq else 10000)/float(qt[1]))
T=[.8,.9,.95,.99]
setups=["DEEP10M-8n","DEEP100M-4n","DEEP100M-6n","TTI10M-8n","TTI100M-4n","TTI100M-6n","TTI100M-8n"]
fmt=lambda q:f"{q/1000:.1f}k" if q>=1000 else f"{q:.0f}"
for s in setups:
    print("##",s)
    for sy in ["GP-ANN*","Pyramid*","CoTra","DA*","Harmony"]:
        d=pts.get((s,sy))
        if not d: print(f"{sy:9} -"); continue
        v=list(d.values()); mr=max(x[0] for x in v)
        cells=[]
        for t in T:
            c=[q for r,q in v if r>=t]
            cells.append(fmt(max(c)) if c else "n/r")
        print(f"{sy:9} pts={len(v):3} minR={min(x[0] for x in v):.3f} maxR={mr:.3f} | "+" ".join(f"{t}:{c}" for t,c in zip(T,cells)))
w=csv.writer(open(sys.argv[1],"w")); w.writerow("setup system config recall10 qps".split())
for (s,sy),d in sorted(pts.items()):
    for c,(r,q) in sorted(d.items(),key=lambda x:x[1][0]): w.writerow([s,sy,c,r,round(q,1)])
