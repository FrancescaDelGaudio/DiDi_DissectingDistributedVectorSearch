// Transport-only replay: the messages of a real run (from a trace), no computation, executed
// once with MPI (the same calls the system makes) and once with native RDMA (verbs, same scheme
// as CoTra: RC queue pairs, pre-registered buffers, RDMA_WRITE_WITH_IMM, polling the receive CQ).
// The difference between the two is what replacing MPI with RDMA saves on moving the data.
//
// Trace format (text, '#' starts a comment):
//   ranks <N>                      must equal the MPI world size
//   barrier <phase>                starts a timed segment (MPI_Barrier, then timer), label <phase>
//   p2p <n>                        n point-to-point messages, one per line: <src> <dst> <bytes>
//                                    MPI: Irecv/Isend + Waitall   RDMA: one write-with-imm each
//   a2a                            then N lines of N byte counts (row = sender)
//                                    MPI: Alltoall of counts + Alltoallv (as DA-batch)
//                                    RDMA: one write-with-imm per non-zero off-diagonal entry
//   stream <n>                     n messages in sending order, one per line: <src> <dst> <bytes>
//                                    MPI: one blocking MPI_Send per message (as Pyramid), receivers
//                                    MPI_Recv(MPI_ANY_SOURCE) in arrival order; a rank either sends
//                                    or receives in a stream   RDMA: one write-with-imm per message
//   bcast <root> <bytes>           MPI: MPI_Bcast   RDMA: root writes to every other rank
//   allreduce <bytes>              MPI: MPI_Allreduce (SUM, doubles)   RDMA: every rank writes to every other
// Steps of one segment run back to back (no barrier between them), in trace order.
// Segment time = slowest rank (MPI_Reduce MAX after the segment, outside the timer).
// A pass = all segments; pass 0 is a warm-up; MPI and RDMA passes alternate (ABBA order).
// Every received RDMA message is checked (first/last 8 bytes carry sender and step).
//
// Usage: mpirun -n <N> transport_replay <trace> [passes=7] [link_gbps=100]
// Env:   RDMA_DEV (default mlx5_0), RDMA_PORT (default 1), RDMA_GID_INDEX (RoCE only, default 0)
// Output (rank 0): "#[replay]" lines per (mode, phase) with min / median / mean seconds per pass,
//                  plus the wire lower bound per phase (busiest link bytes / link rate).
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_rank = 0;

#define CHECK(cond, msg)                                                                       \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::fprintf(stderr, "rank %d FATAL: %s (%s:%d)\n", g_rank, msg, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

constexpr int kRecvDepth = 4096;  // receive WRs kept posted on every QP (streams of small messages)
constexpr int kSendDepth = 256;
constexpr size_t kInline = 128;   // RDMA writes up to this size are sent inline

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- trace -------------------------------------------------------------------------------

enum class StepType { P2P, STREAM, A2A, BCAST, ALLREDUCE };

struct Msg {
    int src, dst;
    size_t bytes;
};

struct Step {
    StepType type;
    std::vector<Msg> msgs;                   // P2P; for A2A/BCAST/ALLREDUCE: the equivalent messages
    std::vector<std::vector<size_t>> mat;    // A2A
    int root = 0;                            // BCAST
    size_t bytes = 0;                        // BCAST / ALLREDUCE
};

struct Segment {
    std::string phase;
    std::vector<Step> steps;
};

struct Trace {
    int ranks = 0;
    std::vector<Segment> segs;
};

Trace read_trace(const std::string& path) {
    std::ifstream in(path);
    CHECK(in.good(), "cannot open trace");
    Trace t;
    std::string line;
    auto next = [&](std::istringstream& ss) {
        while (std::getline(in, line)) {
            const size_t h = line.find('#');
            if (h != std::string::npos) line.resize(h);
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            ss = std::istringstream(line);
            return true;
        }
        return false;
    };
    std::istringstream ss;
    while (next(ss)) {
        std::string kw;
        ss >> kw;
        if (kw == "ranks") {
            ss >> t.ranks;
        } else if (kw == "barrier") {
            Segment s;
            ss >> s.phase;
            t.segs.push_back(s);
        } else {
            CHECK(t.ranks > 0 && !t.segs.empty(), "trace: step before 'ranks' / first 'barrier'");
            Step st;
            if (kw == "p2p" || kw == "stream") {
                st.type = kw == "p2p" ? StepType::P2P : StepType::STREAM;
                int n = 0;
                ss >> n;
                for (int i = 0; i < n; ++i) {
                    std::istringstream ms;
                    CHECK(next(ms), "trace: truncated p2p/stream");
                    Msg m;
                    ms >> m.src >> m.dst >> m.bytes;
                    CHECK(m.src >= 0 && m.src < t.ranks && m.dst >= 0 && m.dst < t.ranks, "trace: bad rank");
                    if (m.bytes > 0 && m.src != m.dst) st.msgs.push_back(m);
                }
                if (st.type == StepType::STREAM) {
                    std::vector<char> sends(t.ranks, 0), recvs(t.ranks, 0);
                    for (const Msg& m : st.msgs) sends[m.src] = recvs[m.dst] = 1;
                    for (int r = 0; r < t.ranks; ++r) CHECK(!(sends[r] && recvs[r]), "trace: a rank both sends and receives in a stream");
                }
            } else if (kw == "a2a") {
                st.type = StepType::A2A;
                st.mat.assign(t.ranks, std::vector<size_t>(t.ranks, 0));
                for (int s = 0; s < t.ranks; ++s) {
                    std::istringstream ms;
                    CHECK(next(ms), "trace: truncated a2a");
                    for (int d = 0; d < t.ranks; ++d) ms >> st.mat[s][d];
                    for (int d = 0; d < t.ranks; ++d)
                        if (d != s && st.mat[s][d] > 0) st.msgs.push_back({s, d, st.mat[s][d]});
                }
            } else if (kw == "bcast") {
                st.type = StepType::BCAST;
                ss >> st.root >> st.bytes;
                for (int d = 0; d < t.ranks; ++d)
                    if (d != st.root && st.bytes > 0) st.msgs.push_back({st.root, d, st.bytes});
            } else if (kw == "allreduce") {
                st.type = StepType::ALLREDUCE;
                ss >> st.bytes;
                st.bytes = std::max<size_t>(8, st.bytes / 8 * 8);
                for (int s = 0; s < t.ranks; ++s)
                    for (int d = 0; d < t.ranks; ++d)
                        if (s != d) st.msgs.push_back({s, d, st.bytes});
            } else {
                CHECK(false, ("trace: unknown keyword " + kw).c_str());
            }
            t.segs.back().steps.push_back(std::move(st));
        }
    }
    CHECK(t.ranks > 0, "trace: missing 'ranks'");
    return t;
}

// ---- RDMA --------------------------------------------------------------------------------

struct QpInfo {
    uint16_t lid;
    uint32_t qpn, psn, rkey;
    uint64_t addr;
    uint8_t gid[16];
};

struct Rdma {
    int rank = 0, size = 0;
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq *send_cq = nullptr, *recv_cq = nullptr;
    ibv_port_attr port_attr{};
    int port = 1, gid_index = 0;
    std::vector<ibv_qp*> qp;
    std::vector<char*> send_buf, recv_buf;   // per peer, sized for the largest message of the trace
    std::vector<size_t> send_cap, recv_cap;
    std::vector<ibv_mr*> send_mr, recv_mr;
    std::vector<QpInfo> remote;
    int outstanding = 0;
    std::map<uint32_t, int> arrived;         // step id -> messages arrived (early arrivals included)
    std::map<uint64_t, int> arrived_from;    // (step id, sender) -> count, for the payload check

    void init(int rank_, int size_, const std::vector<size_t>& out_cap, const std::vector<size_t>& in_cap) {
        rank = rank_;
        size = size_;
        const char* dev_name = std::getenv("RDMA_DEV") ? std::getenv("RDMA_DEV") : "mlx5_0";
        port = std::getenv("RDMA_PORT") ? std::atoi(std::getenv("RDMA_PORT")) : 1;
        gid_index = std::getenv("RDMA_GID_INDEX") ? std::atoi(std::getenv("RDMA_GID_INDEX")) : 0;
        int num_dev = 0;
        ibv_device** devs = ibv_get_device_list(&num_dev);
        CHECK(devs && num_dev > 0, "no RDMA device");
        for (int i = 0; i < num_dev; ++i)
            if (std::string(ibv_get_device_name(devs[i])) == dev_name) ctx = ibv_open_device(devs[i]);
        ibv_free_device_list(devs);
        CHECK(ctx, "cannot open RDMA device");
        CHECK(ibv_query_port(ctx, port, &port_attr) == 0, "ibv_query_port");
        CHECK(port_attr.state == IBV_PORT_ACTIVE, "port not active");
        pd = ibv_alloc_pd(ctx);
        CHECK(pd, "ibv_alloc_pd");
        send_cq = ibv_create_cq(ctx, kSendDepth * 2, nullptr, nullptr, 0);
        recv_cq = ibv_create_cq(ctx, kRecvDepth * size, nullptr, nullptr, 0);
        CHECK(send_cq && recv_cq, "ibv_create_cq");

        qp.assign(size, nullptr);
        send_buf.assign(size, nullptr);
        recv_buf.assign(size, nullptr);
        send_mr.assign(size, nullptr);
        recv_mr.assign(size, nullptr);
        send_cap = out_cap;
        recv_cap = in_cap;
        std::vector<QpInfo> mine(size), theirs(size);
        ibv_gid gid{};
        if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET)
            CHECK(ibv_query_gid(ctx, port, gid_index, &gid) == 0, "ibv_query_gid");
        auto alloc_reg = [&](size_t bytes, int access, char*& buf, ibv_mr*& mr) {
            bytes = std::max<size_t>(bytes, 64);
            buf = static_cast<char*>(std::aligned_alloc(4096, (bytes + 4095) / 4096 * 4096));
            CHECK(buf, "aligned_alloc");
            std::memset(buf, 0, bytes);   // touch: no page faults inside the timed region
            mr = ibv_reg_mr(pd, buf, bytes, access);
            CHECK(mr, "ibv_reg_mr");
        };
        for (int p = 0; p < size; ++p) {
            if (p == rank) continue;
            alloc_reg(send_cap[p], IBV_ACCESS_LOCAL_WRITE, send_buf[p], send_mr[p]);
            alloc_reg(recv_cap[p], IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE, recv_buf[p], recv_mr[p]);
            ibv_qp_init_attr attr{};
            attr.send_cq = send_cq;
            attr.recv_cq = recv_cq;
            attr.qp_type = IBV_QPT_RC;
            attr.cap.max_send_wr = kSendDepth;
            attr.cap.max_recv_wr = kRecvDepth;
            attr.cap.max_send_sge = 1;
            attr.cap.max_recv_sge = 1;
            attr.cap.max_inline_data = kInline;
            qp[p] = ibv_create_qp(pd, &attr);
            CHECK(qp[p], "ibv_create_qp");
            ibv_qp_attr init{};
            init.qp_state = IBV_QPS_INIT;
            init.port_num = port;
            init.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
            CHECK(ibv_modify_qp(qp[p], &init, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                  "modify QP to INIT");
            mine[p].lid = port_attr.lid;
            mine[p].qpn = qp[p]->qp_num;
            mine[p].psn = static_cast<uint32_t>(1000 + rank * 64 + p) & 0xffffff;
            mine[p].rkey = recv_mr[p]->rkey;
            mine[p].addr = reinterpret_cast<uint64_t>(recv_buf[p]);
            std::memcpy(mine[p].gid, gid.raw, 16);
        }
        // Bootstrap only (not timed): entry p goes to rank p.
        MPI_Alltoall(mine.data(), sizeof(QpInfo), MPI_BYTE, theirs.data(), sizeof(QpInfo), MPI_BYTE, MPI_COMM_WORLD);
        remote = theirs;
        for (int p = 0; p < size; ++p) {
            if (p == rank) continue;
            ibv_qp_attr rtr{};
            rtr.qp_state = IBV_QPS_RTR;
            rtr.path_mtu = port_attr.active_mtu;
            rtr.dest_qp_num = remote[p].qpn;
            rtr.rq_psn = remote[p].psn;
            rtr.max_dest_rd_atomic = 1;
            rtr.min_rnr_timer = 1;   // 0.01 ms: a receiver short of receive WRs is retried quickly
            rtr.ah_attr.dlid = remote[p].lid;
            rtr.ah_attr.port_num = port;
            if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
                rtr.ah_attr.is_global = 1;
                std::memcpy(rtr.ah_attr.grh.dgid.raw, remote[p].gid, 16);
                rtr.ah_attr.grh.sgid_index = gid_index;
                rtr.ah_attr.grh.hop_limit = 1;
            }
            CHECK(ibv_modify_qp(qp[p], &rtr,
                                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                    IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0,
                  "modify QP to RTR");
            ibv_qp_attr rts{};
            rts.qp_state = IBV_QPS_RTS;
            rts.timeout = 14;
            rts.retry_cnt = 7;
            rts.rnr_retry = 7;
            rts.sq_psn = mine[p].psn;
            rts.max_rd_atomic = 1;
            CHECK(ibv_modify_qp(qp[p], &rts,
                                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                    IBV_QP_MAX_QP_RD_ATOMIC) == 0,
                  "modify QP to RTS");
            for (int i = 0; i < kRecvDepth; ++i) post_recv(p);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void post_recv(int p) {
        ibv_recv_wr wr{};
        ibv_recv_wr* bad = nullptr;
        wr.wr_id = static_cast<uint64_t>(p);
        CHECK(ibv_post_recv(qp[p], &wr, &bad) == 0, "ibv_post_recv");
    }

    void poll_sends(int at_least) {
        ibv_wc wc[32];
        int done = 0;
        while (done < at_least) {
            const int n = ibv_poll_cq(send_cq, 32, wc);
            CHECK(n >= 0, "ibv_poll_cq(send)");
            for (int i = 0; i < n; ++i) CHECK(wc[i].status == IBV_WC_SUCCESS, ibv_wc_status_str(wc[i].status));
            done += n;
            outstanding -= n;
        }
    }

    // Write `bytes` from send_buf[p] into peer p's recv_buf[me]; imm = step id (24 bit) | sender (8 bit).
    void write(int p, size_t bytes, uint32_t step) {
        CHECK(bytes <= send_cap[p], "message larger than the send buffer");
        CHECK(bytes <= 0xffffffffu, "message larger than 4 GiB");
        while (outstanding >= kSendDepth) poll_sends(1);
        ibv_sge sge{};
        sge.addr = reinterpret_cast<uint64_t>(send_buf[p]);
        sge.length = static_cast<uint32_t>(bytes);
        sge.lkey = send_mr[p]->lkey;
        ibv_send_wr wr{};
        ibv_send_wr* bad = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.send_flags = IBV_SEND_SIGNALED | (bytes <= kInline ? IBV_SEND_INLINE : 0);
        wr.imm_data = htonl(((step & 0xffffffu) << 8) | static_cast<uint32_t>(rank));
        wr.wr.rdma.remote_addr = remote[p].addr;
        wr.wr.rdma.rkey = remote[p].rkey;
        CHECK(ibv_post_send(qp[p], &wr, &bad) == 0, "ibv_post_send");
        ++outstanding;
    }

    // Busy-poll until `expected` messages of `step` have arrived; later steps are counted too.
    void wait_step(uint32_t step, int expected) {
        ibv_wc wc[32];
        while (arrived[step & 0xffffffu] < expected) {
            const int n = ibv_poll_cq(recv_cq, 32, wc);
            CHECK(n >= 0, "ibv_poll_cq(recv)");
            for (int i = 0; i < n; ++i) {
                CHECK(wc[i].status == IBV_WC_SUCCESS, ibv_wc_status_str(wc[i].status));
                CHECK(wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM, "unexpected recv opcode");
                const uint32_t imm = ntohl(wc[i].imm_data);
                const int sender = static_cast<int>(imm & 0xff);
                arrived[imm >> 8]++;
                post_recv(sender);
            }
        }
        arrived.erase(step & 0xffffffu);
    }
};

void stamp(char* buf, size_t bytes, uint32_t a, uint32_t b) {
    const uint32_t tag[2] = {a, b};
    std::memcpy(buf, tag, std::min<size_t>(8, bytes));
    if (bytes >= 16) std::memcpy(buf + bytes - 8, tag, 8);
}

// The buffer of sender s may already hold a later step (RDMA senders can run ahead within a
// segment): the check accepts any step >= the current one.
bool stamp_ok(const char* buf, size_t bytes, uint32_t sender, uint32_t step) {
    if (bytes < 8) return true;
    uint32_t a[2];
    std::memcpy(a, buf, 8);
    if (a[0] != sender || a[1] < step) return false;
    if (bytes >= 16) {
        uint32_t c[2];
        std::memcpy(c, buf + bytes - 8, 8);
        if (c[0] != sender || c[1] < step) return false;
    }
    return true;
}

// ---- execution ---------------------------------------------------------------------------

struct Runner {
    int rank = 0, size = 0;
    const Trace* t = nullptr;
    Rdma rdma;
    // MPI buffers: the RDMA ones are reused for p2p/bcast (MPI registers them on first use);
    // contiguous ones for the collectives.
    std::vector<char> a2a_send, a2a_recv, coll_buf, red_in, red_out, stream_buf;

    void run_step_mpi(const Step& st, int step_id) {
        const int tag = step_id % 30000;
        switch (st.type) {
            case StepType::P2P: {
                std::vector<MPI_Request> reqs;
                for (const Msg& m : st.msgs)
                    if (m.dst == rank) {
                        reqs.emplace_back();
                        MPI_Irecv(rdma.recv_buf[m.src], static_cast<int>(m.bytes), MPI_BYTE, m.src, tag, MPI_COMM_WORLD, &reqs.back());
                    }
                for (const Msg& m : st.msgs)
                    if (m.src == rank) {
                        reqs.emplace_back();
                        MPI_Isend(rdma.send_buf[m.dst], static_cast<int>(m.bytes), MPI_BYTE, m.dst, tag, MPI_COMM_WORLD, &reqs.back());
                    }
                MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
                break;
            }
            case StepType::STREAM: {
                int expected = 0;
                for (const Msg& m : st.msgs)
                    if (m.dst == rank) ++expected;
                for (const Msg& m : st.msgs)
                    if (m.src == rank)
                        MPI_Send(rdma.send_buf[m.dst], static_cast<int>(m.bytes), MPI_BYTE, m.dst, tag, MPI_COMM_WORLD);
                for (int i = 0; i < expected; ++i)
                    MPI_Recv(stream_buf.data(), static_cast<int>(stream_buf.size()), MPI_BYTE, MPI_ANY_SOURCE, tag,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            }
            case StepType::A2A: {
                // Same calls as DA-batch's exchange(): Alltoall of the counts, then Alltoallv.
                std::vector<int> sc(size), sd(size), rc(size), rd(size);
                size_t off = 0;
                for (int d = 0; d < size; ++d) {
                    sc[d] = static_cast<int>(st.mat[rank][d]);
                    sd[d] = static_cast<int>(off);
                    off += st.mat[rank][d];
                }
                MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
                size_t roff = 0;
                for (int s = 0; s < size; ++s) {
                    rd[s] = static_cast<int>(roff);
                    roff += static_cast<size_t>(rc[s]);
                }
                MPI_Alltoallv(a2a_send.data(), sc.data(), sd.data(), MPI_BYTE, a2a_recv.data(), rc.data(), rd.data(), MPI_BYTE,
                              MPI_COMM_WORLD);
                break;
            }
            case StepType::BCAST:
                MPI_Bcast(coll_buf.data(), static_cast<int>(st.bytes), MPI_BYTE, st.root, MPI_COMM_WORLD);
                break;
            case StepType::ALLREDUCE:
                MPI_Allreduce(red_in.data(), red_out.data(), static_cast<int>(st.bytes / 8), MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                break;
        }
    }

    void run_step_rdma(const Step& st, uint32_t step_id) {
        int expected = 0;
        for (const Msg& m : st.msgs)
            if (m.src == rank) {
                stamp(rdma.send_buf[m.dst], m.bytes, static_cast<uint32_t>(rank), step_id);
                rdma.write(m.dst, m.bytes, step_id);
            } else if (m.dst == rank) {
                ++expected;
            }
        rdma.wait_step(step_id, expected);
        for (const Msg& m : st.msgs)
            if (m.dst == rank) CHECK(stamp_ok(rdma.recv_buf[m.src], m.bytes, m.src, step_id), "bad RDMA payload");
        if (rdma.outstanding > 0) rdma.poll_sends(rdma.outstanding);   // send buffers reusable
    }

    // One pass over all segments; returns the slowest-rank time of every segment.
    std::vector<double> pass(bool use_rdma, uint32_t& step_counter) {
        std::vector<double> seg_t(t->segs.size(), 0.0);
        for (size_t s = 0; s < t->segs.size(); ++s) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = now();
            for (const Step& st : t->segs[s].steps) {
                const uint32_t id = step_counter++;
                if (use_rdma)
                    run_step_rdma(st, id);
                else
                    run_step_mpi(st, static_cast<int>(id));
            }
            const double local = now() - t0;
            MPI_Allreduce(&local, &seg_t[s], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }
        return seg_t;
    }
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    CHECK(argc >= 2, "usage: transport_replay <trace> [passes=7] [link_gbps=100]");
    const std::string trace_path = argv[1];
    const int passes = argc > 2 ? std::atoi(argv[2]) : 7;
    const double link_gbps = argc > 3 ? std::atof(argv[3]) : 100.0;
    CHECK(passes >= 2, "need at least 2 passes (pass 0 is a warm-up)");

    const Trace trace = read_trace(trace_path);
    CHECK(trace.ranks == size, "trace 'ranks' differs from the MPI world size");
    const int rank = g_rank;

    // Buffer sizes: largest message per (me -> peer) and (peer -> me); collectives: largest total.
    std::vector<size_t> out_cap(size, 0), in_cap(size, 0);
    size_t a2a_out = 0, a2a_in = 0, coll = 8, red = 8, stream = 8;
    for (const Segment& seg : trace.segs)
        for (const Step& st : seg.steps) {
            if (st.type == StepType::STREAM)
                for (const Msg& m : st.msgs) stream = std::max(stream, m.bytes);
            for (const Msg& m : st.msgs) {
                if (m.src == rank) out_cap[m.dst] = std::max(out_cap[m.dst], m.bytes);
                if (m.dst == rank) in_cap[m.src] = std::max(in_cap[m.src], m.bytes);
            }
            if (st.type == StepType::A2A) {
                size_t o = 0, i = 0;
                for (int p = 0; p < size; ++p) {
                    o += st.mat[rank][p];
                    i += st.mat[p][rank];
                }
                a2a_out = std::max(a2a_out, o);
                a2a_in = std::max(a2a_in, i);
            }
            if (st.type == StepType::BCAST) coll = std::max(coll, st.bytes);
            if (st.type == StepType::ALLREDUCE) red = std::max(red, st.bytes);
        }

    Runner run;
    run.rank = rank;
    run.size = size;
    run.t = &trace;
    run.a2a_send.assign(std::max<size_t>(a2a_out, 8), 0);
    run.a2a_recv.assign(std::max<size_t>(a2a_in, 8), 0);
    run.coll_buf.assign(coll, 0);
    run.red_in.assign(red, 0);
    run.red_out.assign(red, 0);
    run.stream_buf.assign(stream, 0);
    run.rdma.init(rank, size, out_cap, in_cap);

    char host[256];
    gethostname(host, sizeof(host));
    std::vector<char> hosts(256 * size);
    MPI_Gather(host, 256, MPI_CHAR, hosts.data(), 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Wire lower bound per segment: busiest rank's max(bytes out, bytes in) over the network.
    // Messages between two ranks on the same node do not use the network (MPI: shared memory).
    std::vector<double> seg_wire(trace.segs.size(), 0.0);
    if (rank == 0) {
        for (size_t s = 0; s < trace.segs.size(); ++s) {
            std::vector<double> out(size, 0), in(size, 0);
            for (const Step& st : trace.segs[s].steps)
                for (const Msg& m : st.msgs) {
                    if (std::strncmp(&hosts[256 * m.src], &hosts[256 * m.dst], 256) == 0) continue;
                    out[m.src] += m.bytes;
                    in[m.dst] += m.bytes;
                }
            double busiest = 0;
            for (int p = 0; p < size; ++p) busiest = std::max({busiest, out[p], in[p]});
            seg_wire[s] = busiest / (link_gbps * 1e9 / 8.0);
        }
        size_t nsteps = 0, nmsgs = 0;
        double bytes = 0;
        for (const Segment& seg : trace.segs)
            for (const Step& st : seg.steps) {
                ++nsteps;
                nmsgs += st.msgs.size();
                for (const Msg& m : st.msgs) bytes += m.bytes;
            }
        std::printf("#[replay] trace=%s ranks=%d segments=%zu steps=%zu messages=%zu total_MB=%.2f passes=%d link_gbps=%.0f "
                    "link_layer=%s\n",
                    trace_path.c_str(), size, trace.segs.size(), nsteps, nmsgs, bytes / 1e6, passes - 1, link_gbps,
                    run.rdma.port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "InfiniBand" : "Ethernet");
        for (int p = 0; p < size; ++p) std::printf("#[replay] rank%d host=%s\n", p, &hosts[256 * p]);
        std::fflush(stdout);
    }

    // Passes: 0 = warm-up for both modes; then ABBA order (mpi, rdma, rdma, mpi, ...).
    uint32_t step_counter = 1;
    std::vector<std::vector<double>> times[2];   // [mode][pass][segment]
    for (int p = 0; p < passes; ++p) {
        const bool rdma_first = (p % 2 == 1);
        for (int k = 0; k < 2; ++k) {
            const bool use_rdma = (k == 0) ? rdma_first : !rdma_first;
            auto seg_t = run.pass(use_rdma, step_counter);
            if (p > 0) times[use_rdma ? 1 : 0].push_back(seg_t);
        }
    }

    if (rank == 0) {
        std::vector<std::string> phases;
        for (const Segment& seg : trace.segs)
            if (std::find(phases.begin(), phases.end(), seg.phase) == phases.end()) phases.push_back(seg.phase);
        phases.push_back("TOTAL");
        for (int mode = 0; mode < 2; ++mode) {
            for (const std::string& ph : phases) {
                std::vector<double> per_pass;
                for (const auto& seg_t : times[mode]) {
                    double sum = 0;
                    for (size_t s = 0; s < trace.segs.size(); ++s)
                        if (ph == "TOTAL" || trace.segs[s].phase == ph) sum += seg_t[s];
                    per_pass.push_back(sum);
                }
                double wire = 0;
                for (size_t s = 0; s < trace.segs.size(); ++s)
                    if (ph == "TOTAL" || trace.segs[s].phase == ph) wire += seg_wire[s];
                double mn = *std::min_element(per_pass.begin(), per_pass.end()), mean = 0;
                for (double v : per_pass) mean += v;
                mean /= per_pass.size();
                std::printf("#[replay] mode=%s phase=%s min_s=%.6f median_s=%.6f mean_s=%.6f wire_s=%.6f\n",
                            mode ? "rdma" : "mpi", ph.c_str(), mn, median(per_pass), mean, wire);
            }
        }
        std::printf("#[replay] all checks passed\n");
        std::fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
