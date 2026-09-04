#pragma once

// Confidence-based dynamic n_expert_used for prefill.
//
// --n-expert-used-prefill sets one fixed K for every prefill ubatch. This gives K a per-run
// signal instead: it reads back the real router softmax (ggml_soft_max output, tensor name
// "ffn_moe_probs-<il>") for one tracked layer via the standard ggml_backend_sched eval
// callback, averages each token's top-1 probability into a confidence score for that ubatch,
// and derives a suggested K from it (low confidence -> keep the model's full n_expert_used,
// high confidence -> --n-expert-used-adaptive-k-min).
//
// The observation always lags one ubatch behind: what's observed from ubatch N is applied to
// ubatch N+1's graph build, because K has to be fixed before that graph's tensor shapes are
// built and there's no way to know a ubatch's own routing confidence before running it. A
// prompt that fits in a single prefill ubatch (< --ubatch-size tokens) never adapts as a
// result -- it always runs at the --n-expert-used-prefill / model-default K.
//
// --n-expert-used-adaptive-log runs the observation and logs what K it would suggest on every
// prefill ubatch without changing behavior, independent of whether --n-expert-used-adaptive
// (which actually applies it) is set -- use it to sanity-check the confidence/K distribution
// on real prompts before trusting the applied version.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

struct llama_moe_dynamic_k {
    llama_moe_dynamic_k(
            uint32_t n_layer,
            uint32_t full_k,
            int32_t  tracked_layer, // < 0 = auto (n_layer / 2)
            uint32_t k_min,
            float    conf_low,
            float    conf_high,
            bool     log_enabled);

    // Call once per graph build, before the graph runs, with this ubatch's shape. Records
    // whether the upcoming compute is a prefill ubatch so the eval callback below knows
    // whether to bother inspecting the router tensor it's watching for.
    void begin_ubatch(bool is_prefill);

    // ggml_backend_sched_eval_callback. Chains to a caller-supplied callback (e.g. imatrix's)
    // for any tensor it doesn't care about, so installing this doesn't take over the slot.
    static bool eval_cb(ggml_tensor * t, bool ask, void * user_data);

    void set_chain(ggml_backend_sched_eval_callback cb, void * data);

    // n_expert_used to use for the *next* prefill ubatch's graph build; `default_k` if
    // nothing has been observed yet (first prefill ubatch of a run, or of a short prompt
    // that never produces a second one).
    uint32_t suggest_k(uint32_t default_k) const;

    struct sample {
        float    confidence;
        uint32_t k;
    };

    std::optional<sample> last() const;

private:
    bool observe(ggml_tensor * t, bool ask);
    uint32_t derive_k(float confidence) const;

    std::string name; // "ffn_moe_probs-<tracked_layer>"

    uint32_t full_k;
    uint32_t k_min;
    float    conf_low;
    float    conf_high;
    bool     log_enabled;

    ggml_backend_sched_eval_callback user_cb      = nullptr;
    void *                           user_cb_data = nullptr;

    bool is_prefill_ubatch = false;

    mutable std::mutex    mtx;
    std::optional<sample> last_sample;
};
