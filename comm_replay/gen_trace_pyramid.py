#!/usr/bin/env python3
"""Pyramid* transport-replay trace from the task destinations dumped by the sender (PYR_DUMP_DEST).

Pyramid's query phase is a pipeline: the sender (rank N) sends one task message (task_B bytes)
per (query, woken shard) with a blocking MPI_Send, in routing order; every shard (ranks 0..N-1)
sends one result message (result_B bytes) per task to the receiver (rank N+1). The trace has
one timed segment per stream (the real run overlaps them; replayed one after the other):
    barrier tasks    stream <sender -> shard, in the dumped order>
    barrier results  stream <shard -> receiver, one per task>

Usage: python3 gen_trace_pyramid.py <dest_file> <num_shards> <out.trace> [result_B=104]
"""
import re
import sys


def main():
    dest_file, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    result_b = int(sys.argv[4]) if len(sys.argv) > 4 else 104
    lines = open(dest_file).read().splitlines()
    m = re.match(r"# sender (\d+) task_B (\d+) messages (\d+)", lines[0])
    if not m:
        sys.exit(f"{dest_file}: missing header")
    sender, task_b, nmsg = int(m.group(1)), int(m.group(2)), int(m.group(3))
    dests = [int(x) for x in lines[1:] if x.strip()]
    if len(dests) != nmsg or sender != n:
        sys.exit(f"{dest_file}: {len(dests)} destinations (header {nmsg}), sender {sender} (expected {n})")
    receiver = n + 1
    with open(out, "w") as f:
        f.write(f"# Pyramid* streams from {dest_file}\nranks {n + 2}\n")
        f.write(f"barrier tasks\nstream {len(dests)}\n")
        f.writelines(f"{sender} {d} {task_b}\n" for d in dests)
        counts = [dests.count(s) for s in range(n)]
        f.write(f"barrier results\nstream {sum(counts)}\n")
        for s in range(n):
            f.writelines(f"{s} {receiver} {result_b}\n" for _ in range(counts[s]))
    print(f"{out}: {len(dests)} tasks of {task_b} B, per shard {counts}")


if __name__ == "__main__":
    main()
