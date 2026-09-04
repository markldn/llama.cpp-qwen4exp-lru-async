#include "llama-moe-expert-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h" // typedefs only (ggml_backend_cuda_host_register_mapped_t) -- the actual symbols are
                        // resolved dynamically via ggml_backend_reg_get_proc_address below, never linked directly,
                        // since the CUDA/HIP backend is a separate .so loaded at runtime (ggml_backend_load_all()).

#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct layer_state {
    llama_moe_cache_layer pub;

    // non-owning; the moe_cache that owns this layer also owns one
    // ggml_backend_t per device (see moe_cache::backends) that the worker
    // thread dispatches this layer's async uploads through.
    ggml_backend_t backend = nullptr;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)

    std::vector<bool>     slot_in_flight; // slot has an upload pending

    // scratch for flush_table's async dev_table write: must stay alive until
    // that copy actually completes (the caller synchronizes before this
    // could be overwritten again), so it can't be a function-local temporary.
    std::vector<int32_t>  table_scratch;

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

struct moe_cache {
    int32_t n_slots     = 0;
    int32_t max_inserts = 2;

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    // one ggml_backend_t per device with cached layers, used by the worker
    // thread to issue async (non-blocking) copies -- see upload_slice below.
    // Owned here; freed in llama_moe_cache_shutdown.
    std::map<ggml_backend_dev_t, ggml_backend_t> backends;
    // guards all use of `backends` (dispatch + synchronize): the worker
    // thread (uploads) and the main decode thread (flush_table's dev_table
    // write, from llama_moe_cache_step()) both use the same per-device
    // backend/stream, and ggml's CUDA/HIP backend context isn't documented
    // as safe for concurrent enqueue from two host threads without this.
    std::mutex backend_mtx;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph can never read a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;
};

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;

// Standalone host-memory pinning (llama_moe_pin_offloaded_experts), separate
// from the GPU-resident cache above: guards its own idempotency and owns its
// own unregister list, since it needs to work even when the cache itself is
// disabled (n_slots == 0).
std::mutex g_pin_mtx;
bool g_pin_done = false;
std::vector<std::pair<ggml_backend_cuda_host_unregister_mapped_t, void *>> g_pinned;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens > 5) {
        return; // batch/prefill: the cache graph is not built there, don't pollute the LRU
    }

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss++;
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
    }
}

// Async: the CUDA/HIP backend's plain ggml_backend_tensor_set() does
// cudaMemcpyAsync() immediately followed by a full cudaStreamSynchronize()
// -- i.e. every "async" upload was actually blocking the worker thread on
// its own individual copy, one at a time, with no pipelining. Measured at
// ~70,000 such round-trips in a 20s decode window. Using the _async form
// here (no implicit sync) and having the caller batch many of these before
// one explicit ggml_backend_synchronize() lets the driver queue and overlap
// them instead.
void upload_slice(ggml_backend_t backend, ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    ggml_backend_tensor_set_async(backend, dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
}

// Rewrites the WHOLE table from the current (in-memory, authoritative)
// expert_slot array in one write per table, instead of one tiny
// ggml_backend_tensor_set per changed entry. A layer can pick up several
// evictions and several publishes in a single step() call (up to
// max_inserts of each); batching means that costs 2 dispatches total for
// the layer regardless of how many entries moved, not up to 2*2*max_inserts.
// dev_table goes through the same async path as upload_slice -- this runs
// on the main decode thread (from llama_moe_cache_step()), so avoiding a
// blocking sync per dirty layer matters for decode latency directly, not
// just worker throughput. host_table is a plain host-to-host memcpy either
// way (no HSA round-trip), so it stays on the synchronous call.
void flush_table(layer_state & ls, int32_t n_slots) {
    const size_t n_expert = ls.expert_slot.size();
    if (ls.table_scratch.size() != n_expert) {
        ls.table_scratch.resize(n_expert);
    }
    for (size_t e = 0; e < n_expert; ++e) {
        ls.table_scratch[e] = ls.expert_slot[e] >= 0 ? ls.expert_slot[e] : n_slots;
    }
    ggml_backend_tensor_set_async(ls.backend, ls.pub.dev_table, ls.table_scratch.data(), 0, n_expert*sizeof(int32_t));
    ggml_backend_tensor_set(ls.pub.host_table, ls.table_scratch.data(), 0, n_expert*sizeof(int32_t));
}

// One-time host-memory pin+map (page-lock the tensor's existing mmap'd
// range in place -- no copy, no extra RAM) for a weight tensor that gets
// read from repeatedly on the host->device path: either the cache's async
// worker slicing individual experts out of it, or ggml-backend-sched's
// op-offload path streaming the whole tensor to a GPU for a large batch.
// Without this, every such copy has to pin the source range first
// (hsa_amd_memory_lock_to_pool) since it's raw mmap'd GGUF data the driver
// has never seen before -- measured at ~13us/call for the cache's small
// per-expert slices, and unpinned host->device copies more broadly go
// through the driver's bounce buffer at roughly half bandwidth instead of
// direct async DMA. Registering once here makes the driver treat the range
// as already pinned for every later copy out of it. The returned mapped
// device pointer isn't used for anything -- registration's side effect
// (pinning) is the whole point, not the mapping. Appends to `out` on
// success so the caller can release it at shutdown; silently leaves the
// tensor unpinned (falling back to the per-call pin/unpin tax) if the
// active backend doesn't expose this registration API.
void pin_host_source(ggml_backend_dev_t dev, ggml_tensor * w,
        std::vector<std::pair<ggml_backend_cuda_host_unregister_mapped_t, void *>> & out) {
    if (w->data == nullptr) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto register_fn   = (ggml_backend_cuda_host_register_mapped_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_register_mapped");
    auto unregister_fn = (ggml_backend_cuda_host_unregister_mapped_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unregister_mapped");
    if (!register_fn || !unregister_fn) {
        LLAMA_LOG_WARN("%s: backend %s does not expose mapped host-memory registration; "
                        "%s will pay a pin/unpin tax on every host<->device copy\n",
                        __func__, ggml_backend_dev_name(dev), w->name);
        return;
    }
    void * device_ptr = nullptr;
    if (!register_fn(w->data, ggml_nbytes(w), &device_ptr)) {
        LLAMA_LOG_WARN("%s: failed to pin+map %s (%.1f MiB); it will pay a pin/unpin tax on every "
                        "host<->device copy instead\n", __func__, w->name, ggml_nbytes(w) / 1024.0 / 1024.0);
        return;
    }
    out.emplace_back(unregister_fn, w->data);
}

} // namespace

void llama_moe_pin_offloaded_experts(const llama_model & model) {
    std::lock_guard<std::mutex> lock(g_pin_mtx);
    if (g_pin_done) {
        return;
    }
    if (model.devices.empty()) {
        g_pin_done = true; // nothing GPU-side to resolve the pin API through
        return;
    }
    ggml_backend_dev_t dev = model.devices[0].dev; // host pinning isn't per-device; any loaded GPU backend's registry works

    size_t n_pinned = 0;
    size_t bytes_pinned = 0;
    for (const auto & l : model.layers) {
        if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps) {
            continue;
        }
        if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
            continue; // dry-run / memory-estimation model: weights not loaded
        }
        if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
            continue; // not offloaded to host: nothing to pin
        }
        const size_t before = g_pinned.size();
        pin_host_source(dev, l.ffn_up_exps,   g_pinned);
        pin_host_source(dev, l.ffn_gate_exps, g_pinned);
        pin_host_source(dev, l.ffn_down_exps, g_pinned);
        for (size_t i = before; i < g_pinned.size(); ++i) {
            n_pinned++;
        }
        bytes_pinned += ggml_nbytes(l.ffn_up_exps) + ggml_nbytes(l.ffn_gate_exps) + ggml_nbytes(l.ffn_down_exps);
    }

    if (n_pinned == 0) {
        return; // NOT g_pin_done = true: this model (e.g. a fully GPU-resident MTP
                 // draft head) genuinely has nothing to pin, but a later call for
                 // the real target model's context should still get to try
    }
    // WARN, not INFO: this fork's LLAMA_LOG_INFO from this file doesn't surface at
    // the server's default verbosity (a logging-tag/verbosity filtering nuance),
    // same reason the cache's own "active" announcement below uses WARN too.
    LLAMA_LOG_WARN("%s: pinned %zu host-resident MoE expert tensor(s), %.1f MiB, for full-bandwidth "
                    "host<->device DMA (cache uploads and/or ggml-backend-sched's op-offload streaming)\n",
                    __func__, n_pinned, bytes_pinned / 1024.0 / 1024.0);
    g_pin_done = true;
}

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    [&]() {
        if (n_slots <= 0) {
            g_init_done = true;
            return;
        }

        auto * mc = new moe_cache();
        mc->n_slots = n_slots;
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router) --
        // this is what makes a multi-GPU tensor-split placement correct: a
        // layer whose OWN (non-expert) tensors landed on GPU1 gets its cache
        // pool on GPU1 too, not forced onto whichever device happened to be
        // first. No explicit per-layer device lookup needed.
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                continue; // experts already on a device: nothing to cache
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: --moe-expert-cache-experts=%d set but no host-resident expert layers found - disabled\n", __func__, n_slots);
            delete mc;
            return; // NOT g_init_done = true: a later call (the real target
                     // model's context, if this was the MTP draft's) may
                     // still find real candidates and should get to try.
        }

        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots  = n_slots;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], n_slots + 1);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], n_slots + 1);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], n_slots + 1);
                    ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                    ggml_format_name(ls->pub.up_c,      "moe_cache_up.%d",   c.il);
                    ggml_format_name(ls->pub.gate_c,    "moe_cache_gate.%d", c.il);
                    ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);

                    // Source-tensor pinning happens in llama_moe_pin_offloaded_experts,
                    // called before this from llama-context.cpp -- not here, since that
                    // pass covers every offloaded expert layer regardless of whether the
                    // cache ends up tracking it.
                    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);

                    ggml_backend_t & backend = mc->backends[dev];
                    if (!backend) {
                        backend = ggml_backend_dev_init(dev, nullptr);
                    }
                    ls->backend = backend;
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            mc->bufs.push_back(buf);
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        if (!ok) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
        }

        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            ls.slot_expert.assign(n_slots, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(n_slots, 0);
            ls.slot_in_flight.assign(n_slots, false);

            std::vector<int32_t> dummy(n_expert, n_slots);
            ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
        }

        mc->worker = std::thread([mc]() {
            for (;;) {
                std::vector<upload_job> batch;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                    if (mc->stop) {
                        return;
                    }
                    // drain everything currently queued into one batch: each
                    // upload_slice below is now a non-blocking dispatch, so
                    // the whole batch's copies can be enqueued back-to-back
                    // and confirmed with one synchronize per backend instead
                    // of one blocking round-trip per tensor per expert.
                    batch.reserve(mc->todo.size());
                    while (!mc->todo.empty()) {
                        batch.push_back(mc->todo.front());
                        mc->todo.pop_front();
                    }
                }

                {
                    std::lock_guard<std::mutex> blk(mc->backend_mtx);
                    for (const auto & j : batch) {
                        auto & ls = mc->layers[j.layer_idx];
                        upload_slice(ls.backend, ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
                        upload_slice(ls.backend, ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
                        upload_slice(ls.backend, ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
                    }
                    for (auto & [dev, backend] : mc->backends) {
                        (void) dev;
                        ggml_backend_synchronize(backend);
                    }
                }

                std::lock_guard<std::mutex> lk(mc->wmtx);
                for (auto j : batch) {
                    j.done = true;
                    mc->done.push_back(j);
                }
            }
        });

        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        g_cache = mc;
        g_init_done = true;

        LLAMA_LOG_WARN("%s: MoE expert cache active (async): %zu layer(s) x %d slots across %zu device(s), "
                        "%d insert(s)/layer/step, %.1f MiB device memory\n",
                        __func__, mc->layers.size(), n_slots, groups.size(), mc->max_inserts, vram/1024.0/1024.0);
    }();
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

void llama_moe_cache_step() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    // layers whose expert_slot table changed this call (eviction and/or
    // publish) -- flushed to device/host tables once each at the end,
    // instead of one tensor_set per changed entry.
    std::vector<bool> dirty(mc->layers.size(), false);

    // 1) publish completed uploads (sync point: no graph is executing)
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & j : mc->done) {
            auto & ls = mc->layers[j.layer_idx];
            ls.slot_expert[j.slot]     = j.expert;
            ls.expert_slot[j.expert]   = j.slot;
            ls.slot_last_use[j.slot]   = ++mc->clock;
            ls.slot_in_flight[j.slot]  = false;
            dirty[j.layer_idx] = true;
        }
        mc->done.clear();
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int budget = mc->max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it, --budget) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0) {
                continue;
            }

            // victim: an empty non-in-flight slot if any, else the LRU non-in-flight slot
            int32_t slot = -1;
            uint64_t best = UINT64_MAX;
            for (int32_t s = 0; s < mc->n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                dirty[li] = true;
            }
            ls.slot_in_flight[slot] = true;

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            mc->todo.push_back({li, id, slot});
        }
        ls.pending.clear();
    }
    mc->wcv.notify_one();

    // 3) one table write per dirty layer, covering every change from both
    //    steps above (publishes and evictions can land on the same layer
    //    in the same call -- expert_slot already reflects the net result).
    // The dev_table write is async (see flush_table) -- synchronize before
    // returning so the update is actually visible before the next decode's
    // graph reads it, and so table_scratch is safe to reuse next step().
    {
        bool any_dirty = false;
        std::lock_guard<std::mutex> blk(mc->backend_mtx);
        for (size_t li = 0; li < mc->layers.size(); ++li) {
            if (dirty[li]) {
                flush_table(mc->layers[li], mc->n_slots);
                any_dirty = true;
            }
        }
        if (any_dirty) {
            for (auto & [dev, backend] : mc->backends) {
                (void) dev;
                ggml_backend_synchronize(backend);
            }
        }
    }

    if (mc->n_steps % 512 == 0) {
        uint64_t h = 0, m = 0;
        for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
        LLAMA_LOG_DEBUG("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0);
    }
}

void llama_moe_cache_gpu_lock() {
    if (g_cache) {
        g_cache->backend_mtx.lock();
    }
}

void llama_moe_cache_gpu_unlock() {
    if (g_cache) {
        g_cache->backend_mtx.unlock();
    }
}

void llama_moe_cache_shutdown() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    g_cache = nullptr;
    ggml_set_moe_obs_callback(nullptr, nullptr);
    {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->stop = true;
    }
    mc->wcv.notify_one();
    if (mc->worker.joinable()) {
        mc->worker.join();
    }
    for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
    for (auto * c : mc->ctxs) { ggml_free(c); }
    for (auto & [dev, backend] : mc->backends) { (void) dev; ggml_backend_free(backend); }
    delete mc;
}
