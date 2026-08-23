#include "shard_bundle.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>

#include "../IndexFlat.h"
#include "../FaissAssert.h"
#include "../utils.h"

namespace auncel_dist {

namespace {

static const uint32_t kProfileMagic = 0x41554e43;  // "AUNC"
static const uint32_t kProfileVersion = 2;

// Heap-owned GT buffers restored from disk (one per loaded profile).
struct ProfileGtStorage {
    std::vector<float> train_D;
    std::vector<faiss::idx> train_I;
};
std::vector<ProfileGtStorage*> g_profile_gt_storage;

template <typename T>
void write_pod(FILE* f, const T& v) {
    fwrite(&v, sizeof(T), 1, f);
}

template <typename T>
void read_pod(FILE* f, T& v) {
    fread(&v, sizeof(T), 1, f);
}

void write_vector_float(FILE* f, const std::vector<float>& v) {
    uint64_t n = v.size();
    write_pod(f, n);
    if (n > 0)
        fwrite(v.data(), sizeof(float), n, f);
}

void read_vector_float(FILE* f, std::vector<float>& v) {
    uint64_t n = 0;
    read_pod(f, n);
    v.resize(n);
    if (n > 0)
        fread(v.data(), sizeof(float), n, f);
}

void write_trace(FILE* f, const faiss::Trace& tr) {
    write_pod(f, tr.nprobe);
    write_pod(f, tr.bs);
    write_vector_float(f, tr.stds);
    uint64_t npairs = tr.trace.size();
    write_pod(f, npairs);
    for (const auto& p : tr.trace) {
        write_pod(f, p.first);
        write_pod(f, p.second);
    }
}

void read_trace(FILE* f, faiss::Trace& tr) {
    read_pod(f, tr.nprobe);
    read_pod(f, tr.bs);
    read_vector_float(f, tr.stds);
    uint64_t npairs = 0;
    read_pod(f, npairs);
    tr.trace.resize(npairs);
    for (uint64_t i = 0; i < npairs; i++) {
        read_pod(f, tr.trace[i].first);
        read_pod(f, tr.trace[i].second);
    }
}

}  // namespace

std::string faissindex_path(const std::string& prefix) {
    return prefix + ".faissindex";
}

std::string profile_path(const std::string& prefix) {
    return prefix + ".profile";
}

std::string meta_path(const std::string& prefix) {
    return prefix + ".meta";
}

void recompute_interdis_cem(faiss::IndexIVF* ix) {
    faiss::IndexFlat* flat = dynamic_cast<faiss::IndexFlat*>(ix->quantizer);
    FAISS_THROW_IF_NOT_MSG(flat, "IVF quantizer must be IndexFlat");
    const size_t nlist = ix->nlist;
    const size_t d = ix->d;
    ix->interdis_cem.resize(nlist * (nlist - 1) / 2);
    if (ix->metric_type != faiss::METRIC_INNER_PRODUCT) {
        faiss::fvec_inter_vecs(
                ix->interdis_cem.data(), flat->xb.data(), nlist, d);
        return;
    }
    std::vector<float> centroids(flat->xb);
    for (size_t i = 0; i < nlist; i++) {
        float* st = centroids.data() + i * d;
        float norm = sqrtf(faiss::fvec_norm_L2sqr(st, d));
        if (norm > 0) {
            for (size_t j = 0; j < d; j++)
                st[j] /= norm;
        }
    }
    faiss::fvec_inter_vecs_IP(
            ix->interdis_cem.data(), centroids.data(), nlist, d);
    for (size_t i = 0; i < ix->interdis_cem.size(); i++)
        ix->interdis_cem[i] = std::acos(ix->interdis_cem[i]);
}

void write_error_profile(const faiss::error_pro* profile, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "could not open %s for write\n", path);
        abort();
    }
    write_pod(f, kProfileMagic);
    write_pod(f, kProfileVersion);
    write_pod(f, profile->std_m);
    write_pod(f, profile->multipler);
    write_pod(f, profile->arcos_size);
    write_vector_float(f, profile->arcos_list);
    write_pod(f, profile->nlist);
    write_pod(f, profile->max_topk);
    write_pod(f, profile->d);
    write_pod(f, profile->train_num);
    write_pod(f, static_cast<uint32_t>(profile->m_type));
    uint64_t ntr = profile->traces.size();
    write_pod(f, ntr);
    for (uint64_t i = 0; i < ntr; i++)
        write_trace(f, profile->traces[i]);

    const uint64_t gt_elems = profile->train_num * profile->max_topk;
    const uint8_t has_gt = (profile->train_D != nullptr && profile->train_I != nullptr);
    write_pod(f, has_gt);
    if (has_gt) {
        fwrite(profile->train_D, sizeof(float), gt_elems, f);
        fwrite(profile->train_I, sizeof(faiss::idx), gt_elems, f);
    }
    fclose(f);
}

faiss::error_pro* read_error_profile(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s for read\n", path);
        abort();
    }
    uint32_t magic = 0, version = 0;
    read_pod(f, magic);
    read_pod(f, version);
    if (magic != kProfileMagic || (version != 1 && version != kProfileVersion)) {
        fprintf(stderr, "bad profile file %s (magic/version)\n", path);
        abort();
    }
    faiss::error_pro* profile = new faiss::error_pro;
    read_pod(f, profile->std_m);
    read_pod(f, profile->multipler);
    read_pod(f, profile->arcos_size);
    read_vector_float(f, profile->arcos_list);
    read_pod(f, profile->nlist);
    read_pod(f, profile->max_topk);
    read_pod(f, profile->d);
    read_pod(f, profile->train_num);
    uint32_t m_type = 0;
    read_pod(f, m_type);
    profile->m_type = static_cast<faiss::error_pro::metric>(m_type);
    uint64_t ntr = 0;
    read_pod(f, ntr);
    profile->traces.resize(ntr);
    for (uint64_t i = 0; i < ntr; i++)
        read_trace(f, profile->traces[i]);

    if (version >= 2) {
        uint8_t has_gt = 0;
        read_pod(f, has_gt);
        if (has_gt) {
            const uint64_t gt_elems = profile->train_num * profile->max_topk;
            auto* store = new ProfileGtStorage;
            store->train_D.resize(gt_elems);
            store->train_I.resize(gt_elems);
            fread(store->train_D.data(), sizeof(float), gt_elems, f);
            fread(store->train_I.data(), sizeof(faiss::idx), gt_elems, f);
            profile->train_D = store->train_D.data();
            profile->train_I = store->train_I.data();
            g_profile_gt_storage.push_back(store);
        }
    }
    fclose(f);
    return profile;
}

void write_meta(const BundleMeta& meta, const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "could not open %s for write\n", path);
        abort();
    }
    fprintf(f, "train_nq %zu\n", meta.train_nq);
    fprintf(f, "k %zu\n", meta.k);
    fprintf(f, "d %zu\n", meta.d);
    fprintf(f, "nb %zu\n", meta.nb);
    fprintf(f, "index_key %s\n", meta.index_key.c_str());
    fprintf(f, "hyperparam_id %d\n", meta.hyperparam_id);
    fprintf(f, "multipler %.9g\n", meta.multipler);
    fprintf(f, "std_m %.9g\n", meta.std_m);
    fprintf(f, "calibrated %d\n", meta.calibrated);
    fclose(f);
}

BundleMeta read_meta(const char* path) {
    BundleMeta meta;
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "could not open %s for read\n", path);
        abort();
    }
    std::string key;
    while (in >> key) {
        if (key == "train_nq")
            in >> meta.train_nq;
        else if (key == "k")
            in >> meta.k;
        else if (key == "d")
            in >> meta.d;
        else if (key == "nb")
            in >> meta.nb;
        else if (key == "index_key")
            in >> meta.index_key;
        else if (key == "hyperparam_id")
            in >> meta.hyperparam_id;
        else if (key == "multipler")
            in >> meta.multipler;
        else if (key == "std_m")
            in >> meta.std_m;
        else if (key == "calibrated")
            in >> meta.calibrated;
    }
    return meta;
}

void save_bundle(
        faiss::Index* index,
        const BundleMeta& meta,
        const std::string& prefix) {
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
    FAISS_THROW_IF_NOT_MSG(ivf && ivf->t, "index must be IVF with trained error profile");
    const std::string idx_path = faissindex_path(prefix);
    const std::string prof_path = profile_path(prefix);
    const std::string meta_file = meta_path(prefix);
    printf("Writing Faiss index -> %s\n", idx_path.c_str());
    faiss::write_index(index, idx_path.c_str());
    printf("Writing error profile -> %s\n", prof_path.c_str());
    write_error_profile(ivf->t, prof_path.c_str());
    printf("Writing metadata -> %s\n", meta_file.c_str());
    write_meta(meta, meta_file.c_str());
}

faiss::Index* load_bundle(const std::string& prefix, BundleMeta* meta_out) {
    const std::string idx_path = faissindex_path(prefix);
    const std::string prof_path = profile_path(prefix);
    const std::string meta_file = meta_path(prefix);
    printf("Loading Faiss index <- %s\n", idx_path.c_str());
    faiss::Index* index = faiss::read_index(idx_path.c_str());
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
    FAISS_THROW_IF_NOT_MSG(ivf, "bundle index must be IndexIVF");
    recompute_interdis_cem(ivf);
    printf("Loading error profile <- %s\n", prof_path.c_str());
    ivf->t = read_error_profile(prof_path.c_str());
    ivf->t->interdis_cem = ivf->interdis_cem.data();
    if (meta_out)
        *meta_out = read_meta(meta_file.c_str());
    return index;
}

void attach_error_sys(
        faiss::Error_sys* err_sys,
        faiss::Index* index,
        const BundleMeta& meta) {
    err_sys->index = dynamic_cast<faiss::IndexIVF*>(index);
    err_sys->key = "IVF";
    err_sys->train_num = meta.train_nq;
    err_sys->max_topk = meta.k;
    err_sys->is_trained = true;
}

void bind_profile_ground_truth(
        faiss::error_pro* profile,
        size_t train_nq,
        size_t max_topk,
        const float* gt_dis,
        const faiss::Index::idx_t* gt_ids) {
    FAISS_THROW_IF_NOT_MSG(
            profile && gt_dis && gt_ids,
            "profile and ground truth pointers must be non-null");
    profile->train_num = train_nq;
    profile->max_topk = max_topk;
    profile->train_D = gt_dis;
    profile->train_I = gt_ids;
    profile->query_topk = max_topk;
}

bool bundle_exists(const std::string& prefix) {
    FILE* f = fopen(faissindex_path(prefix).c_str(), "rb");
    if (!f)
        return false;
    fclose(f);
    f = fopen(profile_path(prefix).c_str(), "rb");
    if (!f)
        return false;
    fclose(f);
    f = fopen(meta_path(prefix).c_str(), "r");
    if (!f)
        return false;
    fclose(f);
    return true;
}

}  // namespace auncel_dist
