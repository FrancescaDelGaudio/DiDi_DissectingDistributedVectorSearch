#pragma once
// GP_RDMA: native RDMA (libibverbs) transport for the MPI calls that GP-ANN* makes during its query
// phase. Only the transport changes: message-queue (buffering, probing, request handling, termination
// detection with repeated global reductions) and GP-ANN* run unchanged; the MPI functions they call are
// redirected here by the macros at the end of this file, with the same signatures and semantics:
//
//   MPI_Isend                       copy into a pre-registered staging ring, RDMA_WRITE_WITH_IMM into the
//                                   receiver's ring (one ring per sender), completes on the send CQ
//   MPI_Improbe / MPI_Get_count /   poll the receive CQ; the immediate data gives sender and ring offset,
//   MPI_Imrecv / MPI_Mrecv          the 16-byte header length and tag; receiving copies the payload out of
//                                   the ring and returns the space to the sender (one-sided write of the
//                                   consumed counter into the sender's credit array)
//   MPI_Test / Testsome / Testany   completion of the requests above
//   / Waitall
//   MPI_Iallreduce (SUM, in place)  every rank writes its contribution to every other rank (write with
//                                   immediate), completes when all contributions of that collective arrived
//   MPI_Gather (root 0)             every rank writes its value to the root
// Same scheme as CoTra: reliable connected queue pairs, buffers registered once at start-up, writes with
// immediate data, busy polling of the completion queues, all from the main thread (MPI_THREAD_FUNNELED).
// Queue pairs are exchanged once with MPI_Alltoall in gp_rdma::init(), outside the timed query phase.
//
// Env: GP_RDMA_DEV (default mlx5_0), GP_RDMA_PORT (1), GP_RDMA_RING_MB (128, ring per peer and direction).
#include <mpi.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

namespace gp_rdma {

constexpr int kRecvDepth = 4096;        // receive WRs kept posted per QP (one per incoming write with imm)
constexpr int kSendDepth = 512;         // send WRs per QP
constexpr size_t kAlign = 64;           // ring slots are 64-byte aligned; imm carries offset / 64 (24 bits)
constexpr int kCollSlots = 64;          // collective slots (seq % kCollSlots), contributions of up to 48 B
constexpr size_t kCollBytes = 64;
constexpr uint32_t kMagic = 0x47505244;  // "GPRD"

struct Hdr {
    uint32_t magic;
    uint32_t len;   // payload bytes
    int32_t tag;
    uint32_t pad;
};

struct Req {
    enum Kind { SEND, DONE, COLL, GATHER_SEND } kind;
    bool done = false;
    // collectives
    uint64_t seq = 0;
    void* buf = nullptr;
    int count = 0;
    int pending_sends = 0;
};

struct Msg {
    int src;
    uint64_t start;   // ring counter of the slot
    uint32_t off;
};

struct PeerInfo {
    uint16_t lid;
    uint32_t qpn, psn;
    uint8_t gid[16];
    uint64_t ring_addr;   uint32_t ring_rkey;     // my ring that receives from this peer
    uint64_t credit_addr; uint32_t credit_rkey;   // my credit array (entry = peer's rank)
    uint64_t coll_addr;   uint32_t coll_rkey;     // my collective region
};

struct State {
    bool on = false;
    int rank = 0, size = 0;
    size_t ring = 0;
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq *scq = nullptr, *rcq = nullptr;
    ibv_port_attr port{};
    int port_num = 1;
    std::vector<ibv_qp*> qp;
    char *rx = nullptr, *tx = nullptr;               // [peer][ring]
    ibv_mr *rx_mr = nullptr, *tx_mr = nullptr;
    volatile uint64_t* credit = nullptr;              // credit[p] = bytes of my stream consumed by p
    ibv_mr* credit_mr = nullptr;
    uint64_t* credit_out = nullptr;                   // staging for the consumed counters I write to others
    ibv_mr* credit_out_mr = nullptr;
    char* coll = nullptr;                              // [slot][sender][kCollBytes]
    ibv_mr* coll_mr = nullptr;
    char* coll_out = nullptr;                          // [slot][kCollBytes]
    ibv_mr* coll_out_mr = nullptr;
    std::vector<PeerInfo> remote;
    std::vector<uint64_t> head;                        // bytes written to p (incl. padding)
    std::vector<uint64_t> consumed;                    // bytes of p's stream I consumed
    std::deque<Msg> arrived;                           // messages in arrival order
    std::vector<uint32_t> coll_seen;                   // [slot * size + sender] = seq (24 bit) + 1
    uint64_t seq = 0;                                  // collectives issued so far (same order everywhere)
    int outstanding = 0;
};

inline State g;
// use_mpi = true: every redirected call goes to the real MPI function instead (GP_ALT_TRANSPORT in
// distributed_bench.cpp switches it between batches, with no request outstanding, so that MPI and RDMA
// batches run in the same process on the same loaded index).
inline bool use_mpi = false;

inline void die(const char* what) {
    std::fprintf(stderr, "rank %d gp_rdma FATAL: %s\n", g.rank, what);
    MPI_Abort(MPI_COMM_WORLD, 3);
}
#define GP_RDMA_CHECK(c, m) do { if (!(c)) ::gp_rdma::die(m); } while (0)

inline size_t round_up(size_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

inline void post_recv(int p) {
    ibv_recv_wr wr{}, *bad = nullptr;
    wr.wr_id = static_cast<uint64_t>(p);
    GP_RDMA_CHECK(ibv_post_recv(g.qp[p], &wr, &bad) == 0, "ibv_post_recv");
}

// Send-side progress: reap completions of writes; wr_id = Req* for MPI sends, 0 otherwise.
inline void poll_send() {
    ibv_wc wc[32];
    int n;
    while ((n = ibv_poll_cq(g.scq, 32, wc)) > 0) {
        for (int i = 0; i < n; ++i) {
            GP_RDMA_CHECK(wc[i].status == IBV_WC_SUCCESS, ibv_wc_status_str(wc[i].status));
            if (wc[i].wr_id) {
                auto* r = reinterpret_cast<Req*>(wc[i].wr_id);
                if (--r->pending_sends <= 0 && r->kind != Req::COLL) r->done = true;
            }
        }
        g.outstanding -= n;
    }
    GP_RDMA_CHECK(n >= 0, "ibv_poll_cq(send)");
}

// Receive-side progress: messages go to the arrival queue, collective contributions are marked.
inline void poll_recv() {
    ibv_wc wc[32];
    int n;
    while ((n = ibv_poll_cq(g.rcq, 32, wc)) > 0) {
        for (int i = 0; i < n; ++i) {
            GP_RDMA_CHECK(wc[i].status == IBV_WC_SUCCESS, ibv_wc_status_str(wc[i].status));
            const uint32_t imm = ntohl(wc[i].imm_data);
            const int src = static_cast<int>((imm >> 24) & 0x7f);
            if (imm >> 31) {   // collective contribution
                const uint32_t s = imm & 0xffffff;
                g.coll_seen[(s % kCollSlots) * g.size + src] = s + 1;
            } else {
                g.arrived.push_back(Msg{src, 0, (imm & 0xffffff) * static_cast<uint32_t>(kAlign)});
            }
            post_recv(src);
        }
    }
    GP_RDMA_CHECK(n >= 0, "ibv_poll_cq(recv)");
}

inline void progress() {
    poll_send();
    poll_recv();
}

inline void post_write(int p, void* local, uint32_t lkey, size_t len, uint64_t raddr, uint32_t rkey, uint32_t imm,
                  bool with_imm, uint64_t wr_id) {
    while (g.outstanding >= kSendDepth) progress();
    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(local);
    sge.length = static_cast<uint32_t>(len);
    sge.lkey = lkey;
    ibv_send_wr wr{}, *bad = nullptr;
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = with_imm ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | (len <= 64 ? IBV_SEND_INLINE : 0);
    wr.imm_data = htonl(imm);
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey = rkey;
    GP_RDMA_CHECK(ibv_post_send(g.qp[p], &wr, &bad) == 0, "ibv_post_send");
    ++g.outstanding;
}

// ---- start-up / shutdown (outside the timed phase) -------------------------------------------------
inline void init(MPI_Comm comm) {
    MPI_Comm_rank(comm, &g.rank);
    MPI_Comm_size(comm, &g.size);
    GP_RDMA_CHECK(g.size <= 128, "at most 128 ranks (7-bit sender in the immediate)");
    const char* dev = std::getenv("GP_RDMA_DEV") ? std::getenv("GP_RDMA_DEV") : "mlx5_0";
    g.port_num = std::getenv("GP_RDMA_PORT") ? std::atoi(std::getenv("GP_RDMA_PORT")) : 1;
    g.ring = static_cast<size_t>(std::getenv("GP_RDMA_RING_MB") ? std::atol(std::getenv("GP_RDMA_RING_MB")) : 128) << 20;
    GP_RDMA_CHECK(g.ring / kAlign <= (1u << 24), "ring larger than 1 GiB");
    int nd = 0;
    ibv_device** devs = ibv_get_device_list(&nd);
    for (int i = 0; i < nd; ++i)
        if (std::string(ibv_get_device_name(devs[i])) == dev) g.ctx = ibv_open_device(devs[i]);
    ibv_free_device_list(devs);
    GP_RDMA_CHECK(g.ctx, "cannot open the RDMA device");
    GP_RDMA_CHECK(ibv_query_port(g.ctx, g.port_num, &g.port) == 0 && g.port.state == IBV_PORT_ACTIVE, "port");
    g.pd = ibv_alloc_pd(g.ctx);
    g.scq = ibv_create_cq(g.ctx, kSendDepth * g.size, nullptr, nullptr, 0);
    g.rcq = ibv_create_cq(g.ctx, kRecvDepth * g.size, nullptr, nullptr, 0);
    GP_RDMA_CHECK(g.pd && g.scq && g.rcq, "pd / cq");

    auto alloc = [&](size_t bytes, int access, ibv_mr*& mr) {
        char* p = static_cast<char*>(std::aligned_alloc(4096, (bytes + 4095) / 4096 * 4096));
        GP_RDMA_CHECK(p, "aligned_alloc");
        std::memset(p, 0, bytes);
        mr = ibv_reg_mr(g.pd, p, bytes, access);
        GP_RDMA_CHECK(mr, "ibv_reg_mr");
        return p;
    };
    const int rw = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    g.rx = alloc(g.ring * g.size, rw, g.rx_mr);
    g.tx = alloc(g.ring * g.size, IBV_ACCESS_LOCAL_WRITE, g.tx_mr);
    g.credit = reinterpret_cast<volatile uint64_t*>(alloc(sizeof(uint64_t) * g.size, rw, g.credit_mr));
    g.credit_out = reinterpret_cast<uint64_t*>(alloc(sizeof(uint64_t) * g.size, IBV_ACCESS_LOCAL_WRITE, g.credit_out_mr));
    g.coll = alloc(kCollSlots * g.size * kCollBytes, rw, g.coll_mr);
    g.coll_out = alloc(kCollSlots * kCollBytes, IBV_ACCESS_LOCAL_WRITE, g.coll_out_mr);
    g.head.assign(g.size, 0);
    g.consumed.assign(g.size, 0);
    g.coll_seen.assign(kCollSlots * g.size, 0);

    g.qp.assign(g.size, nullptr);
    std::vector<PeerInfo> mine(g.size), theirs(g.size);
    ibv_gid gid{};
    if (g.port.link_layer == IBV_LINK_LAYER_ETHERNET) ibv_query_gid(g.ctx, g.port_num, 0, &gid);
    for (int p = 0; p < g.size; ++p) {
        if (p == g.rank) continue;
        ibv_qp_init_attr a{};
        a.send_cq = g.scq;
        a.recv_cq = g.rcq;
        a.qp_type = IBV_QPT_RC;
        a.cap.max_send_wr = kSendDepth;
        a.cap.max_recv_wr = kRecvDepth;
        a.cap.max_send_sge = 1;
        a.cap.max_recv_sge = 1;
        a.cap.max_inline_data = 64;
        g.qp[p] = ibv_create_qp(g.pd, &a);
        GP_RDMA_CHECK(g.qp[p], "ibv_create_qp");
        ibv_qp_attr in{};
        in.qp_state = IBV_QPS_INIT;
        in.port_num = g.port_num;
        in.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        GP_RDMA_CHECK(ibv_modify_qp(g.qp[p], &in, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                      "QP INIT");
        PeerInfo& m = mine[p];
        m.lid = g.port.lid;
        m.qpn = g.qp[p]->qp_num;
        m.psn = static_cast<uint32_t>(4000 + g.rank * 131 + p) & 0xffffff;
        std::memcpy(m.gid, gid.raw, 16);
        m.ring_addr = reinterpret_cast<uint64_t>(g.rx + static_cast<size_t>(p) * g.ring);
        m.ring_rkey = g.rx_mr->rkey;
        m.credit_addr = reinterpret_cast<uint64_t>(g.credit);
        m.credit_rkey = g.credit_mr->rkey;
        m.coll_addr = reinterpret_cast<uint64_t>(g.coll);
        m.coll_rkey = g.coll_mr->rkey;
    }
    MPI_Alltoall(mine.data(), sizeof(PeerInfo), MPI_BYTE, theirs.data(), sizeof(PeerInfo), MPI_BYTE, comm);
    g.remote = theirs;
    for (int p = 0; p < g.size; ++p) {
        if (p == g.rank) continue;
        ibv_qp_attr r{};
        r.qp_state = IBV_QPS_RTR;
        r.path_mtu = g.port.active_mtu;
        r.dest_qp_num = g.remote[p].qpn;
        r.rq_psn = g.remote[p].psn;
        r.max_dest_rd_atomic = 1;
        r.min_rnr_timer = 1;
        r.ah_attr.dlid = g.remote[p].lid;
        r.ah_attr.port_num = g.port_num;
        if (g.port.link_layer == IBV_LINK_LAYER_ETHERNET) {
            r.ah_attr.is_global = 1;
            std::memcpy(r.ah_attr.grh.dgid.raw, g.remote[p].gid, 16);
            r.ah_attr.grh.hop_limit = 1;
        }
        GP_RDMA_CHECK(ibv_modify_qp(g.qp[p], &r, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                                     IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0,
                      "QP RTR");
        ibv_qp_attr s{};
        s.qp_state = IBV_QPS_RTS;
        s.timeout = 14;
        s.retry_cnt = 7;
        s.rnr_retry = 7;
        s.sq_psn = mine[p].psn;
        s.max_rd_atomic = 1;
        GP_RDMA_CHECK(ibv_modify_qp(g.qp[p], &s, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                                     IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0,
                      "QP RTS");
        for (int i = 0; i < kRecvDepth; ++i) post_recv(p);
    }
    MPI_Barrier(comm);
    g.on = true;
    if (g.rank == 0)
        std::printf("#[rdma] transport=verbs device=%s link_layer=%s ring_MB_per_peer=%zu ranks=%d\n", dev,
                    g.port.link_layer == IBV_LINK_LAYER_INFINIBAND ? "InfiniBand" : "Ethernet", g.ring >> 20, g.size);
}

inline void finalize() {
    if (!g.on) return;
    while (g.outstanding > 0) progress();
    for (auto* q : g.qp)
        if (q) ibv_destroy_qp(q);
    for (ibv_mr* m : {g.rx_mr, g.tx_mr, g.credit_mr, g.credit_out_mr, g.coll_mr, g.coll_out_mr}) ibv_dereg_mr(m);
    ibv_destroy_cq(g.scq);
    ibv_destroy_cq(g.rcq);
    ibv_dealloc_pd(g.pd);
    ibv_close_device(g.ctx);
    g.on = false;
}

inline int type_size(MPI_Datatype t) {
    int s = 0;
    PMPI_Type_size(t, &s);
    return s;
}

inline MPI_Request as_request(Req* r) { return reinterpret_cast<MPI_Request>(r); }
inline Req* as_req(MPI_Request r) { return reinterpret_cast<Req*>(r); }

// ---- point to point --------------------------------------------------------------------------------
inline int Isend(const void* buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm, MPI_Request* request) {
    if (use_mpi) return ::MPI_Isend(buf, count, dt, dest, tag, comm, request);
    GP_RDMA_CHECK(dest != g.rank && dest >= 0 && dest < g.size, "Isend: bad destination");
    const size_t len = static_cast<size_t>(count) * type_size(dt);
    const size_t need = round_up(sizeof(Hdr) + len);
    GP_RDMA_CHECK(need <= g.ring, "message larger than the ring (raise GP_RDMA_RING_MB)");
    uint64_t start = g.head[dest];
    const size_t pos = start % g.ring;
    if (pos + need > g.ring) start += g.ring - pos;   // does not fit before the end: skip to the ring start
    while (start + need - g.credit[dest] > g.ring) progress();   // wait for the receiver to free space
    const size_t off = start % g.ring;
    char* slot = g.tx + static_cast<size_t>(dest) * g.ring + off;
    Hdr h{kMagic, static_cast<uint32_t>(len), tag, 0};
    std::memcpy(slot, &h, sizeof(Hdr));
    std::memcpy(slot + sizeof(Hdr), buf, len);
    auto* r = new Req{Req::SEND};
    r->pending_sends = 1;
    const uint32_t imm = (static_cast<uint32_t>(g.rank) << 24) | static_cast<uint32_t>(off / kAlign);
    post_write(dest, slot, g.tx_mr->lkey, sizeof(Hdr) + len, g.remote[dest].ring_addr + off, g.remote[dest].ring_rkey, imm, true,
          reinterpret_cast<uint64_t>(r));
    g.head[dest] = start + need;
    *request = as_request(r);
    return MPI_SUCCESS;
}

inline int Improbe(int source, int tag, MPI_Comm comm, int* flag, MPI_Message* message, MPI_Status* status) {
    if (use_mpi) return ::MPI_Improbe(source, tag, comm, flag, message, status);
    progress();
    for (auto it = g.arrived.begin(); it != g.arrived.end(); ++it) {
        if (source != MPI_ANY_SOURCE && it->src != source) continue;
        const Hdr* h = reinterpret_cast<const Hdr*>(g.rx + static_cast<size_t>(it->src) * g.ring + it->off);
        GP_RDMA_CHECK(h->magic == kMagic, "Improbe: corrupted ring slot");
        if (tag != MPI_ANY_TAG && h->tag != tag) continue;
        if (status != MPI_STATUS_IGNORE) {
            status->MPI_SOURCE = it->src;
            status->MPI_TAG = h->tag;
            status->MPI_ERROR = MPI_SUCCESS;
            status->_ucount = h->len;
        }
        *message = reinterpret_cast<MPI_Message>(new Msg(*it));
        g.arrived.erase(it);
        *flag = 1;
        return MPI_SUCCESS;
    }
    *flag = 0;
    return MPI_SUCCESS;
}

inline int Get_count(const MPI_Status* status, MPI_Datatype dt, int* count) {
    if (use_mpi) return ::MPI_Get_count(status, dt, count);
    *count = static_cast<int>(status->_ucount / type_size(dt));
    return MPI_SUCCESS;
}

// Copy the payload out of the ring and give the space back to the sender (in-order per sender).
inline void receive_matched(void* buf, int count, MPI_Datatype dt, MPI_Message* message) {
    Msg* m = reinterpret_cast<Msg*>(*message);
    const Hdr* h = reinterpret_cast<const Hdr*>(g.rx + static_cast<size_t>(m->src) * g.ring + m->off);
    const size_t len = static_cast<size_t>(count) * type_size(dt);
    GP_RDMA_CHECK(len == h->len, "receive: size differs from the probed message");
    std::memcpy(buf, reinterpret_cast<const char*>(h) + sizeof(Hdr), len);
    // the slot starts at m->off; account for the padding skipped at the end of the ring
    uint64_t start = g.consumed[m->src];
    const size_t pos = start % g.ring;
    if (m->off < pos) start += g.ring - pos;
    start += m->off - start % g.ring;
    g.consumed[m->src] = start + round_up(sizeof(Hdr) + len);
    g.credit_out[m->src] = g.consumed[m->src];
    post_write(m->src, &g.credit_out[m->src], g.credit_out_mr->lkey, sizeof(uint64_t),
          g.remote[m->src].credit_addr + sizeof(uint64_t) * g.rank, g.remote[m->src].credit_rkey, 0, false, 0);
    delete m;
    *message = MPI_MESSAGE_NULL;
}

inline int Imrecv(void* buf, int count, MPI_Datatype dt, MPI_Message* message, MPI_Request* request) {
    if (use_mpi) return ::MPI_Imrecv(buf, count, dt, message, request);
    receive_matched(buf, count, dt, message);
    auto* r = new Req{Req::DONE};
    r->done = true;
    *request = as_request(r);
    return MPI_SUCCESS;
}

inline int Mrecv(void* buf, int count, MPI_Datatype dt, MPI_Message* message, MPI_Status* status_) {
    if (use_mpi) return ::MPI_Mrecv(buf, count, dt, message, status_);
    receive_matched(buf, count, dt, message);
    return MPI_SUCCESS;
}

// ---- collectives -----------------------------------------------------------------------------------
inline char* coll_slot(uint64_t seq, int sender) {
    return g.coll + ((seq % kCollSlots) * g.size + sender) * kCollBytes;
}

inline bool coll_from(uint64_t seq, int sender) {
    return g.coll_seen[(seq % kCollSlots) * g.size + sender] == static_cast<uint32_t>((seq & 0xffffff) + 1);
}

// Write `bytes` of `data` as contribution `seq` to every rank in [targets].
inline void coll_send(uint64_t seq, const void* data, size_t bytes, const std::vector<int>& targets, Req* r) {
    char* out = g.coll_out + (seq % kCollSlots) * kCollBytes;
    std::memcpy(out, data, bytes);
    std::memcpy(coll_slot(seq, g.rank), data, bytes);
    const uint32_t imm = 0x80000000u | (static_cast<uint32_t>(g.rank) << 24) | static_cast<uint32_t>(seq & 0xffffff);
    for (int p : targets) {
        if (r) ++r->pending_sends;
        post_write(p, out, g.coll_out_mr->lkey, bytes,
              g.remote[p].coll_addr + ((seq % kCollSlots) * g.size + g.rank) * kCollBytes, g.remote[p].coll_rkey, imm, true,
              reinterpret_cast<uint64_t>(r));
    }
}

inline bool allreduce_done(Req* r) {
    if (r->pending_sends > 0) return false;
    for (int p = 0; p < g.size; ++p)
        if (p != g.rank && !coll_from(r->seq, p)) return false;
    auto* out = static_cast<uint64_t*>(r->buf);
    for (int i = 0; i < r->count; ++i) {
        uint64_t s = 0;
        for (int p = 0; p < g.size; ++p) s += reinterpret_cast<const uint64_t*>(coll_slot(r->seq, p))[i];
        out[i] = s;
    }
    return true;
}

inline int Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype dt, MPI_Op op, MPI_Comm comm,
                      MPI_Request* request) {
    if (use_mpi) return ::MPI_Iallreduce(sendbuf, recvbuf, count, dt, op, comm, request);
    GP_RDMA_CHECK(sendbuf == MPI_IN_PLACE && op == MPI_SUM && type_size(dt) == 8 &&
                      static_cast<size_t>(count) * 8 <= kCollBytes,
                  "Iallreduce: only in-place SUM of 8-byte integers is implemented");
    auto* r = new Req{Req::COLL};
    r->seq = g.seq++;
    r->buf = recvbuf;
    r->count = count;
    std::vector<int> others;
    for (int p = 0; p < g.size; ++p)
        if (p != g.rank) others.push_back(p);
    coll_send(r->seq, recvbuf, static_cast<size_t>(count) * 8, others, r);
    *request = as_request(r);
    return MPI_SUCCESS;
}

inline int Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount_,
                  MPI_Datatype recvtype_, int root, MPI_Comm comm) {
    if (use_mpi) return ::MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount_, recvtype_, root, comm);
    const size_t bytes = static_cast<size_t>(sendcount) * type_size(sendtype);
    GP_RDMA_CHECK(bytes <= kCollBytes, "Gather: contribution larger than a collective slot");
    const uint64_t seq = g.seq++;
    if (g.rank != root) {
        Req r{Req::GATHER_SEND};
        coll_send(seq, sendbuf, bytes, {root}, &r);
        while (r.pending_sends > 0) progress();
        return MPI_SUCCESS;
    }
    std::memcpy(coll_slot(seq, g.rank), sendbuf, bytes);
    for (int p = 0; p < g.size; ++p) {
        if (p == root) continue;
        while (!coll_from(seq, p)) progress();
    }
    for (int p = 0; p < g.size; ++p) std::memcpy(static_cast<char*>(recvbuf) + p * bytes, coll_slot(seq, p), bytes);
    return MPI_SUCCESS;
}

// ---- completion ------------------------------------------------------------------------------------
inline bool finished(Req* r) {
    if (r->kind == Req::COLL) return allreduce_done(r);
    return r->done;
}

inline int Test(MPI_Request* request, int* flag, MPI_Status* status_) {
    if (use_mpi) return ::MPI_Test(request, flag, status_);
    if (*request == MPI_REQUEST_NULL) {
        *flag = 1;
        return MPI_SUCCESS;
    }
    progress();
    Req* r = as_req(*request);
    *flag = finished(r) ? 1 : 0;
    if (*flag) {
        delete r;
        *request = MPI_REQUEST_NULL;
    }
    return MPI_SUCCESS;
}

inline int Testsome(int incount, MPI_Request requests[], int* outcount, int indices[], MPI_Status statuses_[]) {
    if (use_mpi) return ::MPI_Testsome(incount, requests, outcount, indices, statuses_);
    progress();
    int active = 0, done = 0;
    for (int i = 0; i < incount; ++i) {
        if (requests[i] == MPI_REQUEST_NULL) continue;
        ++active;
        Req* r = as_req(requests[i]);
        if (finished(r)) {
            delete r;
            requests[i] = MPI_REQUEST_NULL;
            indices[done++] = i;
        }
    }
    *outcount = active ? done : MPI_UNDEFINED;
    return MPI_SUCCESS;
}

inline int Testany(int count, MPI_Request requests[], int* index, int* flag, MPI_Status* status_) {
    if (use_mpi) return ::MPI_Testany(count, requests, index, flag, status_);
    progress();
    bool active = false;
    for (int i = 0; i < count; ++i) {
        if (requests[i] == MPI_REQUEST_NULL) continue;
        active = true;
        Req* r = as_req(requests[i]);
        if (finished(r)) {
            delete r;
            requests[i] = MPI_REQUEST_NULL;
            *index = i;
            *flag = 1;
            return MPI_SUCCESS;
        }
    }
    *index = MPI_UNDEFINED;
    *flag = active ? 0 : 1;
    return MPI_SUCCESS;
}

inline int Waitall(int count, MPI_Request requests[], MPI_Status statuses_[]) {
    if (use_mpi) return ::MPI_Waitall(count, requests, statuses_);
    for (int i = 0; i < count; ++i) {
        while (requests[i] != MPI_REQUEST_NULL) {
            Req* r = as_req(requests[i]);
            if (finished(r)) {
                delete r;
                requests[i] = MPI_REQUEST_NULL;
            } else {
                progress();
            }
        }
    }
    return MPI_SUCCESS;
}

}  // namespace gp_rdma

// Redirect the calls of message-queue and of the query phase (the code is otherwise unchanged).
#define MPI_Isend gp_rdma::Isend
#define MPI_Improbe gp_rdma::Improbe
#define MPI_Get_count gp_rdma::Get_count
#define MPI_Imrecv gp_rdma::Imrecv
#define MPI_Mrecv gp_rdma::Mrecv
#define MPI_Test gp_rdma::Test
#define MPI_Testsome gp_rdma::Testsome
#define MPI_Testany gp_rdma::Testany
#define MPI_Waitall gp_rdma::Waitall
#define MPI_Iallreduce gp_rdma::Iallreduce
#define MPI_Gather gp_rdma::Gather
