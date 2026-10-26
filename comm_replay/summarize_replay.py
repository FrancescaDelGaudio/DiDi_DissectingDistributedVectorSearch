#!/usr/bin/env python3
"""Table of transport_replay results: per trace, MPI vs RDMA (median over passes) and wire bound.

Reads the "#[replay]" lines of one or more logs (several replays per log are fine).
    rdma_gain_s = mpi_median_s - rdma_median_s   (what replacing MPI with RDMA saves on the transfers)
Usage: python3 summarize_replay.py <log> [<log> ...] [> table.tsv]
"""
import re
import sys


def main():
    rows = []
    cur = None
    for path in sys.argv[1:]:
        for line in open(path):
            m = re.search(r"#\[replay\] trace=(\S+)", line)
            if m:
                cur = {"trace": m.group(1), "res": {}}
                rows.append(cur)
                continue
            m = re.search(r"#\[replay\] mode=(\w+) phase=(\S+) min_s=([\d.]+) median_s=([\d.]+) mean_s=([\d.]+) wire_s=([\d.]+)", line)
            if m and cur is not None:
                cur["res"][(m.group(1), m.group(2))] = {"min": float(m.group(3)), "median": float(m.group(4)),
                                                        "wire": float(m.group(6))}
    print("trace\tphase\tmpi_median_s\tmpi_min_s\trdma_median_s\trdma_min_s\twire_s\trdma_gain_s\tmpi_over_rdma")
    for r in rows:
        phases = []
        for (mode, ph) in r["res"]:
            if ph not in phases:
                phases.append(ph)
        for ph in phases:
            a, b = r["res"].get(("mpi", ph)), r["res"].get(("rdma", ph))
            if not a or not b:
                continue
            ratio = a["median"] / b["median"] if b["median"] > 0 else float("nan")
            print(f"{r['trace']}\t{ph}\t{a['median']:.6f}\t{a['min']:.6f}\t{b['median']:.6f}\t{b['min']:.6f}\t"
                  f"{a['wire']:.6f}\t{a['median'] - b['median']:.6f}\t{ratio:.2f}")


if __name__ == "__main__":
    main()
