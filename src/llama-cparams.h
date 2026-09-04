#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_moe_dynamic_k;

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    // override n_expert_used for prefill ubatches (n_seq_tokens > 1); see llama-graph.cpp.
    // 0 = disabled, use the model's own n_expert_used for everything. Used as the K for the
    // first prefill ubatch of a run even when n_expert_used_adaptive is on (see below), since
    // there's nothing observed yet to derive a K from at that point.
    int32_t  n_expert_used_prefill = 0;

    // override n_expert_used for every non-prefill ubatch (ordinary decode AND speculative-decode
    // verify batches); see llama-graph.cpp. 0 = disabled, use the model's own n_expert_used.
    int32_t  n_expert_used_decode = 0;

    // confidence-based dynamic K for prefill; see llama-moe-dynamic-k.h. Owned by the
    // llama_context that set this (non-null only when n_expert_used_adaptive or
    // n_expert_used_adaptive_log was requested).
    struct llama_moe_dynamic_k * moe_dynamic_k = nullptr;
    bool n_expert_used_adaptive = false; // apply moe_dynamic_k's suggested K to prefill ubatches

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool fused_hc_mix;       // use the fused hyper-connection mixer (qwen4exp decode)
    bool fused_hc_combine;   // use the fused hyper-connection combine (qwen4exp decode)
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    // 0 = disabled (default: --n-cpu-moe's static CPU/GPU split, unchanged).
    // >0 = number of experts kept resident in the device-side LRU cache pool
    // per cached MoE layer; see llama-moe-expert-cache.h.
    int32_t moe_expert_cache_size;
    // max expert uploads per cached layer per decode step (throttles the
    // async cache-fill worker so a cold cache can't saturate the host<->GPU
    // link); see llama-moe-expert-cache.h.
    int32_t moe_expert_cache_inserts;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
