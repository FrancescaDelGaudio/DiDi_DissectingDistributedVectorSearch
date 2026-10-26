#!/usr/bin/env python3
"""Pyramid* runs (real or PYR_COMM_ONLY=1): per (K, ef), medians over the repeats.

Per run, from the "#[prof ]" lines of sender, receiver and shards:
  wall_s        receiver query-phase wall time (the system's query time)
  sender_*      sender total, time in MPI_Send (thread 0), time waiting for routed queries,
                MPI_Send time per task message
  shard_*       mean over shards: active window, search utilisation, thread-0 MPI recv / send
Usage: python3 analyze_pyramid.py <prof.tsv>   (columns: K, ef, rep, prof_line)
"""
import re
import statistics as st
import sys
from collections import defaultdict


def main():
    runs = defaultdict(lambda: {"slaves": []})
    for line in open(sys.argv[1]).readlines()[1:]:
        k, ef, rep, prof = line.rstrip("\n").split("\t", 3)
        kv = {a: float(b) for a, b in re.findall(r"(\w+)=([\d.]+)", prof)}
        r = runs[(int(k), int(ef), int(rep))]
        if prof.startswith("sender"):
            r["sender"] = kv
        elif prof.startswith("receiver"):
            r["receiver"] = kv
        elif prof.startswith("slave"):
            r["slaves"].append(kv)
    per_cfg = defaultdict(list)
    for key, r in runs.items():
        if "sender" not in r or "receiver" not in r or not r["slaves"]:
            print(f"WARN: {key} incomplete, skipped", file=sys.stderr)
            continue
        s, sl = r["sender"], r["slaves"]
        msgs = max(1.0, s.get("task_msgs", 1.0))
        per_cfg[key[:2]].append(dict(
            wall=r["receiver"]["wall_s"], s_total=s["total_s"], s_send=s["send_s"], s_wait=s.get("send_wait_s", 0.0),
            us_msg=s["send_s"] * 1e6 / msgs, msgs=msgs,
            window=st.mean(x["window_s"] for x in sl), util=st.mean(x["search_util"] for x in sl),
            sh_recv=st.mean(x["recv_s"] for x in sl), sh_send=st.mean(x["send_s"] for x in sl)))
    print("K\tef\treps\twall_s_med\twall_s_min\twall_s_max\tsender_total_s\tsender_send_s\tsender_wait_s\t"
          "send_us_per_msg\ttask_msgs\tshard_window_s\tshard_search_util\tshard_recv_s\tshard_send_s")
    for (k, ef), v in sorted(per_cfg.items()):
        med = lambda a: st.median(x[a] for x in v)
        print(f"{k}\t{ef}\t{len(v)}\t{med('wall'):.4f}\t{min(x['wall'] for x in v):.4f}\t{max(x['wall'] for x in v):.4f}\t"
              f"{med('s_total'):.4f}\t{med('s_send'):.4f}\t{med('s_wait'):.4f}\t{med('us_msg'):.1f}\t{med('msgs'):.0f}\t"
              f"{med('window'):.4f}\t{med('util'):.3f}\t{med('sh_recv'):.4f}\t{med('sh_send'):.4f}")


if __name__ == "__main__":
    main()
