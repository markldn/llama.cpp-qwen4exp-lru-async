#pragma once

// GPU-resident LRU cache for MoE expert weights that live in host memory (via
// -ot ...exps=CPU / --n-cpu-moe). Decode on a host-offloaded MoE layer is
// bound by host RAM bandwidth: every token streams the routed experts'
// weights from system RAM. This caches recently-used experts in VRAM.
//
// This is the second design (2026-09-03), replacing the original
// synchronous-fetch-on-miss one: that version's ggml_moe_expert_copy ran
// in-graph, so every decode step blocked on a PCIe copy for any miss
// (rocprofv3 measured it at ~35% of total GPU kernel time -- more than the
// actual MoE matmuls). This version never blocks decode on a miss:
//
//  - per cached layer, companion tensors up_c/gate_c/down_c of shape
//    [ne0, ne1, n_slots+1] live in the device buffer of that layer's router
//    (llama_moe_cache_init groups host-resident layers by the buffer TYPE of
//    their ffn_gate_inp -- correct per-device placement in a multi-GPU split
//    falls out of this for free, no explicit device-list plumbing needed);
//    slot n_slots is permanently zero (the "dummy" slot).
//  - an I32 table[n_expert] maps expert id -> slot, or n_slots when uncached.
//    One copy on device (read by ggml_get_rows to remap ids for the
//    cache-side mul_mat_id chain) and one on host (read by the CPU
//    mul_mat_id via src[3] -- see ggml_compute_forward_mul_mat_id in
//    ggml-cpu.c -- to SKIP cached ids there, zeroing their dst rows).
//  - the two down-projection outputs are summed: uncached ids contribute 0
//    through the cache chain (their remapped slot is the all-zero dummy
//    slot) and cached ids contribute 0 through the CPU chain (skipped), so
//    the result is exact either way -- each expert is computed on exactly
//    one of the two chains.
//  - a miss is never fetched inline. llama_moe_cache_step(), called once per
//    llama_context::decode() after the graph has finished executing (a safe
//    sync point -- no graph is running), evicts LRU victims and hands their
//    replacement slice-copies to a background worker thread. The new
//    mapping is only published (table updated) once the worker reports the
//    copy done, at a LATER step() call -- a running graph can never observe
//    a torn slot. Uploads are throttled (n_moe_cache_inserts per layer per
//    step) so a cold cache can't saturate the host<->GPU link.
//  - the CPU mul_mat_id path computing a miss THIS step is exactly the
//    stock no-cache path (same cost as caching being off for that expert) --
//    the cache only ever removes work, never adds a stall.
//
// Ported from the mechanism in ggml-org/llama.cpp#27861 (open PR, same core
// design), adapted to this fork's qwen4exp/build_moe_ffn call shape and
// kept single-threaded-simple where the original PR already was.
//
// Enabled via --moe-expert-cache-experts N (slots/layer) and
// --moe-expert-cache-inserts N (uploads/layer/step, default 2).

#include <cstdint>

struct llama_model;
struct ggml_tensor;

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0;

    // host-resident source weights (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // device-resident cache slots, ne[2] == n_slots + 1 (last slot all zeros)
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;
};

// One-time host-memory pin (page-lock, in place -- no copy, no extra RAM) of
// every --n-cpu-moe-offloaded expert weight tensor, independent of whether
// the GPU-resident cache below is enabled. ggml-backend-sched's op-offload
// path (GGML_OP_OFFLOAD_MIN_BATCH*) streams these same host-resident weights
// to a GPU on the fly for large batches (prefill) regardless of the cache;
// from unpinned (plain mmap'd) memory that copy goes through the driver's
// bounce buffer at roughly half bandwidth. Call this once per model load,
// before llama_moe_cache_init if the cache is also enabled -- that function
// no longer pins its own tensors, it relies on this having already run.
// Safe to call more than once (e.g. target + MTP draft); a no-op after the
// first successful call.
void llama_moe_pin_offloaded_experts(const llama_model & model);

// Build the cache for every host-resident expert layer of the model. Safe to
// call more than once (e.g. once per llama_context, target + MTP draft) --
// only the first call that actually finds host-resident layers does work;
// a model with none (like a fully GPU-resident draft head) is a no-op that
// leaves the door open for a later real call to still succeed.
void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts);

// nullptr when the cache is disabled or this tensor has no cached layer.
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps);

// Apply throttled LRU updates (publish completed uploads, evict + schedule
// new ones). Call once per decode(), between graph executions only -- never
// while a graph referencing the cache tensors/tables may still be running.
void llama_moe_cache_step();

// Stops the upload worker thread and frees cache resources. Call once at
// process/context teardown if a clean shutdown matters (tests, embedding
// this in a longer-lived host process); the OS reclaims everything on exit
// either way, so a normal llama-server run doesn't need to call this.
void llama_moe_cache_shutdown();
