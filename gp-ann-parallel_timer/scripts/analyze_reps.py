#!/usr/bin/env python3
"""GP-ANN* repetitions: per target, medians over the repeats of the query time and of its split.

Per run (from the "#[prof ]" rows of the 8 ranks):
  T          = max over ranks of total_s (query phase of the slowest rank)
  compute_r  = route_wall_s + post_req_s + search_wall_s + merge_s of rank r
  comm_wait  = mean over ranks of (total_s - compute_r): communication + protocol + waiting
  max_removable = T - max_r compute_r: the most that ANY communication improvement can save
                  (the run cannot end before the busiest rank has done its own work)
Usage: python3 analyze_reps.py <gp_reps_..._prof.tsv> <gp_reps_....tsv>
"""
import re
import statistics as st
import sys
from collections import defaultdict


def main():
    prof_path, summary_path = sys.argv[1], sys.argv[2]
    runs = defaultdict(list)
    for line in open(prof_path).readlines()[1:]:
        np_, ef, rep, prof = line.rstrip("\n").split("\t", 3)
        kv = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", prof)}
        runs[(int(np_), int(ef), int(rep))].append(kv)
    summ = {}
    for line in open(summary_path).readlines()[1:]:
        p = line.rstrip("\n").split("\t")
        summ[(int(p[0]), int(p[1]), int(p[2]))] = p

    per_cfg = defaultdict(list)
    for key, ranks in runs.items():
        if len(ranks) != 8:
            print(f"WARN: {key} has {len(ranks)} rank rows, skipped", file=sys.stderr)
            continue
        T = max(r["total_s"] for r in ranks)
        comp = [r["route_wall_s"] + r["post_req_s"] + r["search_wall_s"] + r["merge_s"] for r in ranks]
        comm_wait = st.mean(r["total_s"] - c for r, c in zip(ranks, comp))
        s = summ.get(key)
        recall = float(s[4]) if s and s[4] != "NA" else float("nan")
        query_s = float(s[5]) if s and s[5] != "NA" else float("nan")
        per_cfg[key[:2]].append(dict(T=T, query_s=query_s, recall=recall, comm_wait=comm_wait,
                                     route=st.mean(r["route_wall_s"] for r in ranks),
                                     search=st.mean(r["search_wall_s"] for r in ranks),
                                     max_removable=T - max(comp)))

    print("nprobe\tef\treps\trecall_med\tquery_s_med\tquery_s_min\tquery_s_max\troute_s_med\tsearch_s_med\t"
          "comm_wait_s_med\tcomm_wait_pct\tmax_removable_s_med\tmax_removable_pct")
    for (np_, ef), v in sorted(per_cfg.items()):
        med = lambda k: st.median(x[k] for x in v)
        T = med("T")
        print(f"{np_}\t{ef}\t{len(v)}\t{med('recall'):.4f}\t{med('query_s'):.4f}\t{min(x['query_s'] for x in v):.4f}\t"
              f"{max(x['query_s'] for x in v):.4f}\t{med('route'):.4f}\t{med('search'):.4f}\t{med('comm_wait'):.4f}\t"
              f"{100 * med('comm_wait') / T:.1f}\t{med('max_removable'):.4f}\t{100 * med('max_removable') / T:.1f}")


if __name__ == "__main__":
    main()
