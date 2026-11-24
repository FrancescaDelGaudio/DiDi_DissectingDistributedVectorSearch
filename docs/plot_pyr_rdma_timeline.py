#!/usr/bin/env python3
"""Pyramid* over MPI vs native RDMA: where the query time goes, along the sender's timeline (single pass,
real runs of job 7017381 (corrected launch), median of 6 per variant and target). Same format as plot_rdma_split.py
(half a column, per target a stacked bar for MPI "M" and RDMA "R", % of the MPI query time W):
  Sending              time the sender spends in Send (MPI_Send or RDMA write): the protocol cost
  Waiting for routing  time the sender waits for queries from its routing threads
  Shards finishing     W - sender total: shards still searching after the last query was sent
Segments: medians over the runs, scaled so that each bar sums to the median W of its variant.
(The comm-zoom split cannot be computed for RDMA: there routing and search overlap, so
W - routing - search is negative.) Writes subfigs/pyr_rdma_split.pdf/.png
"""
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_gp_mpi_vs_rdma import INK, SURFACE, clean, style

HERE = Path(__file__).resolve().parent
LOG = Path("/gpfs/scratch/delgaudio/comm_replay/logs/pyr_ab_7017381.log")
CFG = {(3, 50): "0.80", (10, 100): "0.90", (20, 200): "0.95", (100, 800): "0.99"}
TARGETS = ["0.80", "0.90", "0.95", "0.99"]
CATS = [("send", "Sending", "#eda100"), ("wait", "Waiting for routing", "#2a78d6"),
        ("tail", "Shards finishing", "#1baf7a")]


def load():
    cur, runs = None, defaultdict(dict)
    for line in open(LOG, errors="replace"):
        m = re.search(r"---- (\w+) mode=(\w+) K=(\d+) ef=(\d+) rep=(\d+) ----", line)
        if m:
            key = (int(m.group(3)), int(m.group(4)))
            cur = (CFG[key], m.group(1).upper(), m.group(2), int(m.group(5))) if key in CFG else None
            continue
        if not cur or cur[2] != "real":
            continue
        kv = {k: float(x) for k, x in re.findall(r"(\w+)=([\d.eE+-]+)", line)}
        if line.startswith("#[prof ] sender"):
            runs[cur].update(total=kv["total_s"], send=kv["send_s"], wait=kv["send_wait_s"])
        elif line.startswith("#[prof ] receiver"):
            runs[cur]["W"] = kv["wall_s"]
    seg = defaultdict(lambda: defaultdict(list))
    for (t, v, _, _), r in runs.items():
        seg[(t, v)]["send"].append(1e3 * r["send"])
        seg[(t, v)]["wait"].append(1e3 * r["wait"])
        seg[(t, v)]["tail"].append(1e3 * max(0.0, r["W"] - r["total"]))
        seg[(t, v)]["W"].append(1e3 * r["W"])
    return {k: {c: st.median(x) for c, x in d.items()} | {"n": len(d["W"])} for k, d in seg.items()}


def main():
    d = load()
    style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(1.65, 2.2), constrained_layout=True)
    w, gap, ticks = 0.40, 0.04, []
    for j, t in enumerate(TARGETS):
        w_mpi = d[(t, "MPI")]["W"]
        for k, v in enumerate(("MPI", "RDMA")):
            x = j + (k - 0.5) * (w + gap)
            bottom = 0.0
            # medians of the segments, scaled so that the bar sums to the median query time W of the variant
            scale = d[(t, v)]["W"] / sum(d[(t, v)][key] for key, _, _ in CATS)
            for key, _l, colour in CATS:
                h = 100 * scale * d[(t, v)][key] / w_mpi
                ax.bar(x, h, bottom=bottom, width=w, color=colour, edgecolor=SURFACE, linewidth=0.4)
                bottom += h
            ticks.append((x, v[0]))
            print(f"{t} {v}: " + "  ".join(f"{key} {100 * d[(t, v)][key] / w_mpi:5.1f}%" for key, _, _ in CATS)
                  + f"  W {d[(t, v)]['W']:.0f} ms = {100 * d[(t, v)]['W'] / w_mpi:.1f}% (n={d[(t, v)]['n']})")
        ax.text(j, -0.085, t[1:], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5)
    ax.set_xticks([x for x, _ in ticks], [l for _, l in ticks])
    ax.tick_params(axis="x", length=0, pad=1.5, labelsize=6)
    ax.set_xlabel("Recall@10", labelpad=12)
    ax.set_ylabel("% of MPI query time")
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in reversed(CATS)], frameon=False, loc="upper center",
              ncol=1, handlelength=0.9, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2)
    clean(ax)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "subfigs" / f"pyr_rdma_split.{ext}", dpi=200)


if __name__ == "__main__":
    main()
