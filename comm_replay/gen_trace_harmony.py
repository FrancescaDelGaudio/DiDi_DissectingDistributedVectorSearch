#!/usr/bin/env python3
"""Harmony transport-replay trace: the same exchanges as Harmony-parallel_timer/bench/harmony_comm_replay.cpp
(group mode, rank 0 = master, ranks 1..8 = workers, teams of 2 workers = ranks 2t-1, 2t), as a trace.

Segments (as the phases of harmony_comm_replay):
  dist  : bcast nq/k, bcast queries (nq*d float), bcast nprobe lists (nq*nprobe int64),
          master -> each worker: compare sizes (nq int64) and prefix sums (nq+1 int64), bcast heap tops (nq float)
  team  : per group g and block b, in every team the first worker of the block (alternating) sends
          the block's partial distances (team_floats / (groups*blocks) floats) to its teammate
  heaps : per block the last worker of each team sends its heap (block*k float, block*k int64) to the
          master; per group the master forwards the group's heaps to the 2 workers of each of the
          next 3 teams (tops: group float, distances: group*k float, ids: group*k int64)
Per-team volumes are the compared-vector counts of the real run (TeamQueryStats), as in
bench/sbatch_comm_replay_deep100m_6n_ext.sh.

Usage: python3 gen_trace_harmony.py <out.trace> <nprobe> <t1> <t2> <t3> <t4> [nq=10000 d=96 k=10 groups=4 blocks=5]
"""
import sys


def main():
    out = sys.argv[1]
    nprobe = int(sys.argv[2])
    team_floats = [int(x) for x in sys.argv[3:7]]
    nq, d, k, groups, blocks = ([int(x) for x in sys.argv[7:12]] + [10000, 96, 10, 4, 5][len(sys.argv[7:12]):])
    teams, team_size = 4, 2
    workers = teams * team_size
    group_size = nq // groups
    block_size = group_size // blocks
    msgs_per_team = groups * blocks

    lines = [f"# Harmony group-mode exchanges, nprobe={nprobe} team_floats={team_floats}",
             f"ranks {workers + 1}"]

    def p2p(msgs):
        lines.append(f"p2p {len(msgs)}")
        lines.extend(f"{s} {t} {b}" for s, t, b in msgs)

    lines.append("barrier dist")
    lines.append("bcast 0 16")
    lines.append(f"bcast 0 {nq * d * 4}")
    lines.append(f"bcast 0 {nq * nprobe * 8}")
    p2p([(0, w, nq * 8) for w in range(1, workers + 1)])
    p2p([(0, w, (nq + 1) * 8) for w in range(1, workers + 1)])
    lines.append(f"bcast 0 {nq * 4}")

    lines.append("barrier team")
    for g in range(groups):
        for b in range(blocks):
            first = b % 2
            msgs = []
            for t in range(teams):
                src = 1 + t * team_size + first
                dst = src + 1 if first == 0 else src - 1
                msgs.append((src, dst, (team_floats[t] // msgs_per_team) * 4))
            p2p(msgs)

    lines.append("barrier heaps")
    for g in range(groups):
        for b in range(blocks):
            last = 1 - (b % 2)
            srcs = [1 + t * team_size + last for t in range(teams)]
            p2p([(s, 0, block_size * k * 4) for s in srcs])
            p2p([(s, 0, block_size * k * 8) for s in srcs])
        for hop in range(teams - 1):
            dst_team = (g + hop + 1) % teams
            for r in range(team_size):
                dst = 1 + dst_team * team_size + r
                p2p([(0, dst, group_size * 4)])
                p2p([(0, dst, group_size * k * 4)])
                p2p([(0, dst, group_size * k * 8)])

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{out}: nprobe={nprobe}, team message {team_floats[0] // msgs_per_team * 4 / 1e6:.1f} MB (team 1)")


if __name__ == "__main__":
    main()
