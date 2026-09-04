#include "llama-moe-dynamic-k.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <vector>

llama_moe_dynamic_k::llama_moe_dynamic_k(
        uint32_t n_layer,
        uint32_t full_k,
        int32_t  tracked_layer,
        uint32_t k_min,
        float    conf_low,
        float    conf_high,
        bool     log_enabled) :
    full_k      (full_k),
    k_min       (std::min(k_min, full_k)),
    conf_low    (conf_low),
    conf_high   (conf_high),
    log_enabled (log_enabled) {
    const uint32_t layer = tracked_layer >= 0 ? (uint32_t) tracked_layer : n_layer / 2;
    name = "ffn_moe_probs-" + std::to_string(layer);
}

void llama_moe_dynamic_k::begin_ubatch(bool is_prefill) {
    is_prefill_ubatch = is_prefill;
}

void llama_moe_dynamic_k::set_chain(ggml_backend_sched_eval_callback cb, void * data) {
    user_cb      = cb;
    user_cb_data = data;
}

uint32_t llama_moe_dynamic_k::derive_k(float confidence) const {
    if (confidence <= conf_low) {
        return full_k;
    }
    if (confidence >= conf_high) {
        return k_min;
    }
    const float t = (confidence - conf_low) / (conf_high - conf_low);
    return (uint32_t) std::lround(full_k - t * (float) (full_k - k_min));
}

bool llama_moe_dynamic_k::observe(ggml_tensor * t, bool ask) {
    if (!is_prefill_ubatch || name != ggml_get_name(t)) {
        return false;
    }

    if (ask) {
        return true;
    }

    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return true; // handled (nothing further to chain), just can't read this shape
    }

    const int64_t n_expert = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    std::vector<float> host_buf;
    const float * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const float *) t->data;
    } else {
        host_buf.resize((size_t) n_expert * n_tokens);
        ggml_backend_tensor_get(t, host_buf.data(), 0, ggml_nbytes(t));
        data = host_buf.data();
    }

    double sum_top1 = 0.0;
    for (int64_t j = 0; j < n_tokens; ++j) {
        const float * row = data + j * n_expert;
        float mx = row[0];
        for (int64_t i = 1; i < n_expert; ++i) {
            mx = std::max(mx, row[i]);
        }
        sum_top1 += mx;
    }

    const float    confidence = n_tokens > 0 ? (float) (sum_top1 / n_tokens) : 0.0f;
    const uint32_t k          = derive_k(confidence);

    {
        std::lock_guard<std::mutex> lock(mtx);
        last_sample = sample{ confidence, k };
    }

    if (log_enabled) {
        // WARN, not INFO: common_log_get_verbosity() (common/log.cpp) maps a library-level
        // GGML_LOG_LEVEL_INFO to LOG_LEVEL_TRACE, so it'd need --verbosity 4 to show at all --
        // WARN stays at its own level and is visible with the more discoverable --verbosity 2.
        LLAMA_LOG_WARN("%s: prefill ubatch (%" PRId64 " tokens): router confidence (%s) = %.3f -> "
                        "suggested n_expert_used = %u for the next prefill ubatch\n",
                        __func__, n_tokens, name.c_str(), confidence, k);
    }

    return true;
}

bool llama_moe_dynamic_k::eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * self = (llama_moe_dynamic_k *) user_data;

    if (self->observe(t, ask)) {
        return true;
    }

    if (self->user_cb) {
        return self->user_cb(t, ask, self->user_cb_data);
    }

    return false;
}

uint32_t llama_moe_dynamic_k::suggest_k(uint32_t default_k) const {
    std::lock_guard<std::mutex> lock(mtx);
    return last_sample ? last_sample->k : default_k;
}

std::optional<llama_moe_dynamic_k::sample> llama_moe_dynamic_k::last() const {
    std::lock_guard<std::mutex> lock(mtx);
    return last_sample;
}
