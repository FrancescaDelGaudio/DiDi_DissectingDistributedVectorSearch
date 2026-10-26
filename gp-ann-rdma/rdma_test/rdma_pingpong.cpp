// Step 1 of GP-ANN over RDMA: verbs connection setup + transport comparison, no ANN code.
// Same scheme as CoTra (RC queue pairs, pre-registered buffers, RDMA_WRITE_WITH_IMM, polling the
// receive CQ), but the QP info is exchanged with MPI (bootstrap only, never timed).
//
// Tests (every rank pair / all ranks, one rank per node):
//   pingpong : rank 0 <-> rank 1, half round trip, RDMA write-with-imm vs MPI_Send/MPI_Recv
//   alltoall : every rank sends S bytes to every other rank at the same time and waits for all
//              N-1 buffers (the GP-ANN query-phase pattern: one large buffer per peer),
//              RDMA vs MPI_Isend/MPI_Irecv vs MPI_Alltoall; time = slowest rank, median of reps
// Every received buffer is checked (first and last word carry sender, iteration).
//
// Usage: mpirun -n <N> --map-by ppr:1:node rdma_pingpong [max_bytes=4194304]
// Env:   RDMA_DEV (default mlx5_0), RDMA_PORT (default 1), RDMA_GID_INDEX (RoCE only, default 0)
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
#include <string>
#include <vector>

namespace {

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "rank %d FATAL: %s (%s:%d)\n", g_rank, msg,     \
                         __FILE__, __LINE__);                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

int g_rank = 0;
constexpr int kRecvDepth = 128;   // receive WRs kept posted on every QP
constexpr int kSendDepth = 128;

struct QpInfo {
    uint16_t lid;
    uint32_t qpn;
    uint32_t psn;
    uint32_t rkey;
    uint64_t addr;
    uint8_t gid[16];
};

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Rdma {
    int rank = 0, size = 0;
    size_t max_bytes = 0;
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* send_cq = nullptr;
    ibv_cq* recv_cq = nullptr;
    ibv_port_attr port_attr{};
    int port = 1;
    int gid_index = 0;
    std::vector<ibv_qp*> qp;          // qp[peer], nullptr for self
    std::vector<char*> send_buf;      // send_buf[peer]: local source of writes to peer
    std::vector<char*> recv_buf;      // recv_buf[peer]: written by peer
    std::vector<ibv_mr*> send_mr, recv_mr;
    std::vector<QpInfo> remote;       // remote[peer]: peer's recv_buf for me
    int outstanding_sends = 0;

    void init(int rank_, int size_, size_t max_bytes_) {
        rank = rank_;
        size = size_;
        max_bytes = max_bytes_;
        const char* dev_name = std::getenv("RDMA_DEV") ? std::getenv("RDMA_DEV") : "mlx5_0";
        port = std::getenv("RDMA_PORT") ? std::atoi(std::getenv("RDMA_PORT")) : 1;
        gid_index = std::getenv("RDMA_GID_INDEX") ? std::atoi(std::getenv("RDMA_GID_INDEX")) : 0;

        int num_dev = 0;
        ibv_device** devs = ibv_get_device_list(&num_dev);
        CHECK(devs && num_dev > 0, "no RDMA device");
        for (int i = 0; i < num_dev; ++i) {
            if (std::string(ibv_get_device_name(devs[i])) == dev_name) ctx = ibv_open_device(devs[i]);
        }
        ibv_free_device_list(devs);
        CHECK(ctx, "cannot open RDMA device");
        CHECK(ibv_query_port(ctx, port, &port_attr) == 0, "ibv_query_port");
        CHECK(port_attr.state == IBV_PORT_ACTIVE, "port not active");
        pd = ibv_alloc_pd(ctx);
        CHECK(pd, "ibv_alloc_pd");
        send_cq = ibv_create_cq(ctx, kSendDepth * size, nullptr, nullptr, 0);
        recv_cq = ibv_create_cq(ctx, kRecvDepth * size, nullptr, nullptr, 0);
        CHECK(send_cq && recv_cq, "ibv_create_cq");

        qp.assign(size, nullptr);
        send_buf.assign(size, nullptr);
        recv_buf.assign(size, nullptr);
        send_mr.assign(size, nullptr);
        recv_mr.assign(size, nullptr);
        std::vector<QpInfo> mine(size), theirs(size);
        ibv_gid gid{};
        if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
            CHECK(ibv_query_gid(ctx, port, gid_index, &gid) == 0, "ibv_query_gid");
        }
        for (int p = 0; p < size; ++p) {
            if (p == rank) continue;
            send_buf[p] = static_cast<char*>(std::aligned_alloc(4096, max_bytes));
            recv_buf[p] = static_cast<char*>(std::aligned_alloc(4096, max_bytes));
            CHECK(send_buf[p] && recv_buf[p], "aligned_alloc");
            std::memset(send_buf[p], 0, max_bytes);
            std::memset(recv_buf[p], 0, max_bytes);
            send_mr[p] = ibv_reg_mr(pd, send_buf[p], max_bytes, IBV_ACCESS_LOCAL_WRITE);
            recv_mr[p] = ibv_reg_mr(pd, recv_buf[p], max_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
            CHECK(send_mr[p] && recv_mr[p], "ibv_reg_mr");

            ibv_qp_init_attr attr{};
            attr.send_cq = send_cq;
            attr.recv_cq = recv_cq;
            attr.qp_type = IBV_QPT_RC;
            attr.cap.max_send_wr = kSendDepth;
            attr.cap.max_recv_wr = kRecvDepth;
            attr.cap.max_send_sge = 1;
            attr.cap.max_recv_sge = 1;
            qp[p] = ibv_create_qp(pd, &attr);
            CHECK(qp[p], "ibv_create_qp");

            ibv_qp_attr init{};
            init.qp_state = IBV_QPS_INIT;
            init.pkey_index = 0;
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
        // Bootstrap: entry p of my array goes to rank p (and I receive what each peer prepared for me).
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
            rtr.min_rnr_timer = 12;
            rtr.ah_attr.dlid = remote[p].lid;
            rtr.ah_attr.sl = 0;
            rtr.ah_attr.src_path_bits = 0;
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

    // write-with-imm consumes one receive WR on the target; no data lands in it (zero SGEs).
    void post_recv(int p) {
        ibv_recv_wr wr{};
        ibv_recv_wr* bad = nullptr;
        wr.wr_id = static_cast<uint64_t>(p);
        wr.sg_list = nullptr;
        wr.num_sge = 0;
        CHECK(ibv_post_recv(qp[p], &wr, &bad) == 0, "ibv_post_recv");
    }

    // RDMA write of `bytes` from send_buf[p] into peer p's recv_buf[me]; imm = my rank.
    void post_write(int p, size_t bytes) {
        while (outstanding_sends >= kSendDepth) poll_sends(1);
        ibv_sge sge{};
        sge.addr = reinterpret_cast<uint64_t>(send_buf[p]);
        sge.length = static_cast<uint32_t>(bytes);
        sge.lkey = send_mr[p]->lkey;
        ibv_send_wr wr{};
        ibv_send_wr* bad = nullptr;
        wr.wr_id = static_cast<uint64_t>(p);
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.imm_data = htonl(static_cast<uint32_t>(rank));
        wr.wr.rdma.remote_addr = remote[p].addr;
        wr.wr.rdma.rkey = remote[p].rkey;
        CHECK(ibv_post_send(qp[p], &wr, &bad) == 0, "ibv_post_send");
        ++outstanding_sends;
    }

    void poll_sends(int at_least) {
        ibv_wc wc[16];
        int done = 0;
        while (done < at_least) {
            int n = ibv_poll_cq(send_cq, 16, wc);
            CHECK(n >= 0, "ibv_poll_cq(send)");
            for (int i = 0; i < n; ++i) CHECK(wc[i].status == IBV_WC_SUCCESS, ibv_wc_status_str(wc[i].status));
            done += n;
            outstanding_sends -= n;
        }
    }

    void drain_sends() {
        if (outstanding_sends > 0) poll_sends(outstanding_sends);
    }

    // Busy-poll the receive CQ for one completed write-with-imm; returns the sender rank.
    int wait_recv() {
        ibv_wc wc;
        int n;
        do {
            n = ibv_poll_cq(recv_cq, 1, &wc);
        } while (n == 0);
        CHECK(n == 1, "ibv_poll_cq(recv)");
        CHECK(wc.status == IBV_WC_SUCCESS, ibv_wc_status_str(wc.status));
        CHECK(wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM, "unexpected recv opcode");
        const int sender = static_cast<int>(ntohl(wc.imm_data));
        post_recv(sender);
        return sender;
    }
};

void stamp(char* buf, size_t bytes, uint32_t sender, uint32_t iter) {
    const uint32_t tag[2] = {sender, iter};
    std::memcpy(buf, tag, std::min<size_t>(8, bytes));
    if (bytes >= 16) std::memcpy(buf + bytes - 8, tag, 8);
}

bool stamp_ok(const char* buf, size_t bytes, uint32_t sender, uint32_t iter) {
    uint32_t a[2] = {0, 0}, b[2] = {sender, iter};
    std::memcpy(a, buf, std::min<size_t>(8, bytes));
    if (bytes < 8) return std::memcmp(a, b, bytes) == 0;
    if (a[0] != sender || a[1] != iter) return false;
    if (bytes >= 16) {
        std::memcpy(a, buf + bytes - 8, 8);
        if (a[0] != sender || a[1] != iter) return false;
    }
    return true;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int iters_for(size_t bytes) {
    if (bytes <= 4096) return 2000;
    if (bytes <= 65536) return 500;
    if (bytes <= 1 << 20) return 100;
    return 30;
}

void pingpong(Rdma& r, size_t max_bytes) {
    const int rank = r.rank;
    if (rank == 0) std::printf("#pingpong rank0<->rank1, half round trip (us), median of 5 runs\n%10s %12s %12s %8s\n",
                               "bytes", "rdma_us", "mpi_us", "mpi/rdma");
    std::vector<char> mpi_send(max_bytes), mpi_recv(max_bytes);
    for (size_t bytes = 8; bytes <= max_bytes; bytes *= 8) {
        const int iters = iters_for(bytes);
        std::vector<double> t_rdma, t_mpi;
        for (int run = 0; run < 5; ++run) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = now();
            if (rank <= 1) {
                const int peer = 1 - rank;
                for (int it = 0; it < iters; ++it) {
                    const uint32_t tag = static_cast<uint32_t>(run * 100000 + it);
                    if (rank == 0) {
                        stamp(r.send_buf[peer], bytes, rank, tag);
                        r.post_write(peer, bytes);
                        CHECK(r.wait_recv() == peer, "pingpong: wrong sender");
                        CHECK(stamp_ok(r.recv_buf[peer], bytes, peer, tag), "pingpong: bad payload (rdma)");
                    } else {
                        CHECK(r.wait_recv() == peer, "pingpong: wrong sender");
                        CHECK(stamp_ok(r.recv_buf[peer], bytes, peer, tag), "pingpong: bad payload (rdma)");
                        stamp(r.send_buf[peer], bytes, rank, tag);
                        r.post_write(peer, bytes);
                    }
                    if (r.outstanding_sends >= kSendDepth / 2) r.drain_sends();
                }
                r.drain_sends();
            }
            t_rdma.push_back((now() - t0) / iters / 2 * 1e6);

            MPI_Barrier(MPI_COMM_WORLD);
            t0 = now();
            if (rank <= 1) {
                const int peer = 1 - rank;
                for (int it = 0; it < iters; ++it) {
                    const uint32_t tag = static_cast<uint32_t>(run * 100000 + it);
                    if (rank == 0) {
                        stamp(mpi_send.data(), bytes, rank, tag);
                        MPI_Send(mpi_send.data(), static_cast<int>(bytes), MPI_BYTE, peer, 0, MPI_COMM_WORLD);
                        MPI_Recv(mpi_recv.data(), static_cast<int>(bytes), MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        CHECK(stamp_ok(mpi_recv.data(), bytes, peer, tag), "pingpong: bad payload (mpi)");
                    } else {
                        MPI_Recv(mpi_recv.data(), static_cast<int>(bytes), MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        CHECK(stamp_ok(mpi_recv.data(), bytes, peer, tag), "pingpong: bad payload (mpi)");
                        stamp(mpi_send.data(), bytes, rank, tag);
                        MPI_Send(mpi_send.data(), static_cast<int>(bytes), MPI_BYTE, peer, 0, MPI_COMM_WORLD);
                    }
                }
            }
            t_mpi.push_back((now() - t0) / iters / 2 * 1e6);
        }
        if (rank == 0) {
            const double a = median(t_rdma), b = median(t_mpi);
            std::printf("%10zu %12.2f %12.2f %8.2f\n", bytes, a, b, b / a);
        }
    }
}

// One exchange: all ranks send `bytes` to every peer, then wait for N-1 buffers. Slowest rank.
double max_over_ranks(double v) {
    double m = 0;
    MPI_Allreduce(&v, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return m;
}

void alltoall(Rdma& r, size_t max_bytes) {
    const int rank = r.rank, size = r.size;
    if (size < 2) return;
    if (rank == 0) std::printf("#alltoall %d ranks, every rank sends S bytes to each peer, slowest rank (ms), median of reps\n"
                               "%10s %10s %10s %10s %10s %10s\n",
                               size, "bytes", "rdma_ms", "isend_ms", "a2a_ms", "rdma_GBs", "isend/rdma");
    std::vector<char> ms(max_bytes * size), mr(max_bytes * size);
    std::vector<MPI_Request> reqs(2 * size);
    for (size_t bytes = 65536; bytes <= max_bytes; bytes *= 4) {
        const int reps = bytes <= (1 << 20) ? 50 : 20;
        std::vector<double> t_rdma, t_isend, t_a2a;
        for (int rep = 0; rep < reps; ++rep) {
            const uint32_t tag = static_cast<uint32_t>(rep);
            // RDMA
            for (int p = 0; p < size; ++p)
                if (p != rank) stamp(r.send_buf[p], bytes, rank, tag);
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = now();
            for (int k = 1; k < size; ++k) r.post_write((rank + k) % size, bytes);
            for (int k = 1; k < size; ++k) r.wait_recv();
            r.drain_sends();
            double t = now() - t0;
            for (int p = 0; p < size; ++p)
                if (p != rank) CHECK(stamp_ok(r.recv_buf[p], bytes, p, tag), "alltoall: bad payload (rdma)");
            t_rdma.push_back(max_over_ranks(t));

            // MPI Isend/Irecv (one buffer per peer, like message-queue's flush)
            for (int p = 0; p < size; ++p)
                if (p != rank) stamp(ms.data() + p * bytes, bytes, rank, tag);
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = now();
            int nr = 0;
            for (int k = 1; k < size; ++k) {
                const int p = (rank + k) % size;
                MPI_Irecv(mr.data() + p * bytes, static_cast<int>(bytes), MPI_BYTE, p, 1, MPI_COMM_WORLD, &reqs[nr++]);
            }
            for (int k = 1; k < size; ++k) {
                const int p = (rank + k) % size;
                MPI_Isend(ms.data() + p * bytes, static_cast<int>(bytes), MPI_BYTE, p, 1, MPI_COMM_WORLD, &reqs[nr++]);
            }
            MPI_Waitall(nr, reqs.data(), MPI_STATUSES_IGNORE);
            t = now() - t0;
            for (int p = 0; p < size; ++p)
                if (p != rank) CHECK(stamp_ok(mr.data() + p * bytes, bytes, p, tag), "alltoall: bad payload (isend)");
            t_isend.push_back(max_over_ranks(t));

            // MPI_Alltoall
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = now();
            MPI_Alltoall(ms.data(), static_cast<int>(bytes), MPI_BYTE, mr.data(), static_cast<int>(bytes), MPI_BYTE, MPI_COMM_WORLD);
            t = now() - t0;
            t_a2a.push_back(max_over_ranks(t));
        }
        if (rank == 0) {
            const double a = median(t_rdma), b = median(t_isend), c = median(t_a2a);
            const double gbs = static_cast<double>(bytes) * (size - 1) / a / 1e9;
            std::printf("%10zu %10.3f %10.3f %10.3f %10.2f %10.2f\n", bytes, a * 1e3, b * 1e3, c * 1e3, gbs, b / a);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const size_t max_bytes = argc > 1 ? std::stoull(argv[1]) : (4u << 20);
    char host[256];
    gethostname(host, sizeof(host));
    std::printf("rank %d host %s\n", g_rank, host);
    std::fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    Rdma r;
    const double t0 = now();
    r.init(g_rank, size, max_bytes);
    const double t_setup = max_over_ranks(now() - t0);
    if (g_rank == 0) {
        std::printf("#setup ok: %d ranks, link_layer=%s, active_mtu=%d, setup %.3f s (not part of any timing)\n", size,
                    r.port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "InfiniBand" : "Ethernet",
                    128 << r.port_attr.active_mtu, t_setup);
        std::fflush(stdout);
    }
    if (size >= 2) pingpong(r, max_bytes);
    std::fflush(stdout);
    alltoall(r, max_bytes);
    std::fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    if (g_rank == 0) std::printf("#all checks passed\n");
    MPI_Finalize();
    return 0;
}
