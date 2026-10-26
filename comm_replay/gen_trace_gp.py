#!/usr/bin/env python3
"""GP-ANN* transport-replay trace from the "#[bytes]" lines of one DistributedBench run (GP_LOG_BYTES=1).

GP-ANN's query phase moves two batches of messages: every rank sends one buffer with all its
requests to each target rank (message-queue flush, one MPI_Isend per destination), then one
buffer with all its responses to each origin rank. The trace has one timed segment per batch:
    barrier req   p2p <requests>
    barrier resp  p2p <responses>
message-queue's termination protocol (rounds of MPI_Iallreduce) is not in the trace: it is the
protocol part, measured by the complete replay (GP_COMM_ONLY=1).

Usage: python3 gen_trace_gp.py <run.log> <out.trace>
"""
import re
import sys


def main():
    log, out = sys.argv[1], sys.argv[2]
    req, resp = {}, {}
    for line in open(log):
        m = re.search(r"#\[bytes\] rank(\d+) req_to=([\d,]+) resp_to=([\d,]+)", line)
        if m:
            r = int(m.group(1))
            req[r] = [int(x) for x in m.group(2).split(",")]
            resp[r] = [int(x) for x in m.group(3).split(",")]
    if not req:
        sys.exit(f"no '#[bytes]' lines in {log} (run with GP_LOG_BYTES=1)")
    n = len(req[0])
    if sorted(req) != list(range(n)):
        sys.exit(f"{log}: expected '#[bytes]' lines for ranks 0..{n - 1}, got {sorted(req)}")
    with open(out, "w") as f:
        f.write(f"# GP-ANN* exchanges from {log}\nranks {n}\n")
        for name, mat in (("req", req), ("resp", resp)):
            msgs = [(s, d, mat[s][d]) for s in range(n) for d in range(n) if s != d and mat[s][d] > 0]
            f.write(f"barrier {name}\np2p {len(msgs)}\n")
            f.writelines(f"{s} {d} {b}\n" for s, d, b in msgs)
    tot = sum(sum(v) for v in req.values()) + sum(sum(v) for v in resp.values())
    print(f"{out}: {n} ranks, {tot / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
