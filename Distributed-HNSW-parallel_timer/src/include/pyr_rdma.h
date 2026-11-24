#pragma once
// PYR_RDMA: native RDMA (libibverbs) transport for the point-to-point calls of the Pyramid* query phase.
// Only the transport changes: same messages (task_message 408 B, result_message), same destinations,
// tags, ordering and blocking semantics; routing, search, merge, pings and shutdown are unchanged.
//   MPI_Send   -> copy into the pre-registered staging ring of the destination, RDMA_WRITE_WITH_IMM into
//                 the receiver's ring (returns once copied, like an eager MPI_Send; waits only if the
//                 destination ring is full)
//   MPI_Iprobe -> poll the receive CQ, look for a matching (source, tag) message, do not consume it
//   MPI_Recv   -> poll until a matching message arrived, copy it out, give the ring space back
// Same layer as GP-ANN* (rdma_transport.h): RC queue pairs, buffers registered once in gp_rdma::init()
// (outside the timed phase), writes with immediate data, busy polling, main thread only (FUNNELED).
// Index construction, barriers and everything outside the query phase stay on MPI.
// Env: GP_RDMA_DEV (mlx5_0), GP_RDMA_PORT (1), GP_RDMA_RING_MB (ring per peer and direction; 16 is plenty).
#define GP_RDMA_NO_MACROS
#include "rdma_transport.h"

namespace gp_rdma {

inline std::deque<Msg>::iterator find_arrived(int source, int tag) {
    for (auto it = g.arrived.begin(); it != g.arrived.end(); ++it) {
        if (source != MPI_ANY_SOURCE && it->src != source) continue;
        const Hdr* h = reinterpret_cast<const Hdr*>(g.rx + static_cast<size_t>(it->src) * g.ring + it->off);
        GP_RDMA_CHECK(h->magic == kMagic, "probe: corrupted ring slot");
        if (tag != MPI_ANY_TAG && h->tag != tag) continue;
        return it;
    }
    return g.arrived.end();
}

inline void fill_status(MPI_Status* status, const Msg& m) {
    if (status == MPI_STATUS_IGNORE) return;
    const Hdr* h = reinterpret_cast<const Hdr*>(g.rx + static_cast<size_t>(m.src) * g.ring + m.off);
    status->MPI_SOURCE = m.src;
    status->MPI_TAG = h->tag;
    status->MPI_ERROR = MPI_SUCCESS;
    status->_ucount = h->len;
}

inline int Send(const void* buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm) {
    GP_RDMA_CHECK(dest != g.rank && dest >= 0 && dest < g.size, "Send: bad destination");
    const size_t len = static_cast<size_t>(count) * type_size(dt);
    const size_t need = round_up(sizeof(Hdr) + len);
    GP_RDMA_CHECK(need <= g.ring, "message larger than the ring (raise GP_RDMA_RING_MB)");
    uint64_t start = g.head[dest];
    const size_t pos = start % g.ring;
    if (pos + need > g.ring) start += g.ring - pos;
    while (start + need - g.credit[dest] > g.ring) progress();
    const size_t off = start % g.ring;
    char* slot = g.tx + static_cast<size_t>(dest) * g.ring + off;
    Hdr h{kMagic, static_cast<uint32_t>(len), tag, 0};
    std::memcpy(slot, &h, sizeof(Hdr));
    std::memcpy(slot + sizeof(Hdr), buf, len);
    const uint32_t imm = (static_cast<uint32_t>(g.rank) << 24) | static_cast<uint32_t>(off / kAlign);
    post_write(dest, slot, g.tx_mr->lkey, sizeof(Hdr) + len, g.remote[dest].ring_addr + off, g.remote[dest].ring_rkey,
               imm, true, 0);
    g.head[dest] = start + need;
    poll_send();
    return MPI_SUCCESS;
}

inline int Iprobe(int source, int tag, MPI_Comm, int* flag, MPI_Status* status) {
    progress();
    auto it = find_arrived(source, tag);
    *flag = it != g.arrived.end();
    if (*flag) fill_status(status, *it);
    return MPI_SUCCESS;
}

inline int Recv(void* buf, int count, MPI_Datatype dt, int source, int tag, MPI_Comm, MPI_Status* status) {
    std::deque<Msg>::iterator it;
    while ((it = find_arrived(source, tag)) == g.arrived.end()) progress();
    const Hdr* h = reinterpret_cast<const Hdr*>(g.rx + static_cast<size_t>(it->src) * g.ring + it->off);
    GP_RDMA_CHECK(h->len <= static_cast<size_t>(count) * type_size(dt), "Recv: message larger than the buffer");
    fill_status(status, *it);
    MPI_Message msg = reinterpret_cast<MPI_Message>(new Msg(*it));
    const int bytes = static_cast<int>(h->len);
    g.arrived.erase(it);
    receive_matched(buf, bytes, MPI_BYTE, &msg);
    return MPI_SUCCESS;
}

}  // namespace gp_rdma

#define PYR_Send gp_rdma::Send
#define PYR_Recv gp_rdma::Recv
#define PYR_Iprobe gp_rdma::Iprobe
