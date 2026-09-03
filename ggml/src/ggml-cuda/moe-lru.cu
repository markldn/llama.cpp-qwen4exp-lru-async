#include "moe-lru.cuh"
#include <cstdio>

// Device-side GPU-resident LRU expert cache. Ports the *algorithm* FreeToken's
// GPU kernels use (see ~/.claude/plans/indexed-zooming-dream.md) -- not its code,
// which is Python/Triton. The whole point of doing this on-device is to avoid the
// host round-trip that made the earlier ggml-backend.cpp-based approach regress
// (host readback of `ids` + host bitset computation, every decode step).
//
// Both kernels below are intentionally single-threaded/simple, not tuned like
// FreeToken's parallel Triton kernels: n_ids and cache_size here are always tiny
// (a handful of routed experts, a few dozen cache slots -- nowhere near the
// 40,000-slot regime FreeToken's streaming top-k eviction variant targets), so a
// sequential scan costs nanoseconds. Correctness first; revisit only if profiling
// says this kernel launch itself is a bottleneck.

// grid=(1,1,1), block=(1,1,1). Needs cache_size bytes of dynamic shared memory
// (one byte per slot, "claimed this call" flag).
static __global__ void moe_lru_ensure_kernel(
        const int32_t * __restrict__ ids,
        int32_t       * __restrict__ out,
        int32_t       * __restrict__ slot_of_id,
        int32_t       * __restrict__ id_of_slot,
        int64_t       * __restrict__ usage,
        int64_t       * __restrict__ step,
        int32_t       * __restrict__ src_idx,
        int32_t       * __restrict__ dst_idx,
        int64_t       * __restrict__ num_copy,
        int32_t n_ids, int32_t cache_size, int32_t num_experts) {

    extern __shared__ char claimed[]; // cache_size bytes

    for (int32_t s = 0; s < cache_size; s++) {
        claimed[s] = 0;
    }

    const int64_t cur_step = ++(*step);

    int32_t n_missing = 0;

    for (int32_t i = 0; i < n_ids; i++) {
        const int32_t id = ids[i];
        if (id < 0 || id >= num_experts) {
            // Defense in depth: should be unreachable now that the caller (apply())
            // forces `ids` contiguous, but a silent clamp beats a GPU fault if some
            // other path ever hands this op a malformed ids tensor.
            out[i] = 0;
            continue;
        }
        const int32_t slot = slot_of_id[id];

        if (slot >= 0) {
            usage[slot] = cur_step;
            claimed[slot] = 1;
            out[i] = slot;
            continue;
        }

        // dedup against earlier positions in this same call
        bool found = false;
        for (int32_t j = 0; j < i; j++) {
            if (ids[j] == id) {
                out[i] = out[j];
                found = true;
                break;
            }
        }
        if (found) {
            continue;
        }

        // evict: argmin usage over slots not already claimed this call
        int32_t victim = -1;
        int64_t best = INT64_MAX;
        for (int32_t s = 0; s < cache_size; s++) {
            if (claimed[s]) {
                continue;
            }
            if (usage[s] < best) {
                best = usage[s];
                victim = s;
            }
        }
        // victim always exists: caller guarantees cache_size >= n_ids

        const int32_t old_id = id_of_slot[victim];
        if (old_id >= 0) {
            slot_of_id[old_id] = -1;
        }
        id_of_slot[victim] = id;
        slot_of_id[id] = victim;
        usage[victim] = cur_step;
        claimed[victim] = 1;

        src_idx[n_missing] = id;
        dst_idx[n_missing] = victim;
        n_missing++;

        out[i] = victim;
    }

    *num_copy = n_missing;
}

void ggml_cuda_op_moe_lru_ensure(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids          = dst->src[0];
    const ggml_tensor * t_slot_of_id = dst->src[1];
    const ggml_tensor * t_id_of_slot = dst->src[2];
    const ggml_tensor * t_usage      = dst->src[3];
    const ggml_tensor * t_step       = dst->src[4];
    const ggml_tensor * t_src_idx    = dst->src[5];
    const ggml_tensor * t_dst_idx    = dst->src[6];
    const ggml_tensor * t_num_copy   = dst->src[7];

    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    // Derived from t_id_of_slot's shape rather than dst->op_params, for the same
    // robustness reason as ggml_cuda_op_moe_expert_copy below.
    const int32_t cache_size = (int32_t) ggml_nelements(t_id_of_slot);
    const int32_t n_ids = (int32_t) ggml_nelements(ids);
    GGML_ASSERT(n_ids <= cache_size);

    moe_lru_ensure_kernel<<<1, 1, cache_size, ctx.stream()>>>(
        (const int32_t *) ids->data,
        (int32_t *) dst->data,
        (int32_t *) t_slot_of_id->data,
        (int32_t *) t_id_of_slot->data,
        (int64_t *) t_usage->data,
        (int64_t *) t_step->data,
        (int32_t *) t_src_idx->data,
        (int32_t *) t_dst_idx->data,
        (int64_t *) t_num_copy->data,
        n_ids, cache_size, (int32_t) ggml_nelements(t_slot_of_id));
}

// One block per potential missing entry (fixed grid = max_missing, launch-time
// constant); each block no-ops if its index is >= num_copy (read once from
// device memory) -- self-terminating, no host sync, CUDA/HIP-graph-capturable.
// Threads within a block cooperate over the row's bytes.
static __global__ void moe_expert_copy_kernel(
        const int32_t * __restrict__ src_idx,
        const int32_t * __restrict__ dst_idx,
        const int64_t * __restrict__ num_copy,
        const char     * __restrict__ host_src,
        char           * __restrict__ pool,
        size_t row_bytes) {

    const int32_t i = blockIdx.x;
    if (i >= (int32_t) *num_copy) {
        return;
    }

    const int32_t sid = src_idx[i];
    const int32_t did = dst_idx[i];

    const char * src_row = host_src + (size_t) sid * row_bytes;
    char       * dst_row = pool     + (size_t) did * row_bytes;

    // vectorized 16-byte copy where alignment allows, byte tail otherwise
    const size_t n16 = row_bytes / 16;
    const uint4 * src16 = (const uint4 *) src_row;
    uint4       * dst16 = (uint4 *) dst_row;
    for (size_t k = threadIdx.x; k < n16; k += blockDim.x) {
        dst16[k] = src16[k];
    }
    for (size_t b = n16 * 16 + threadIdx.x; b < row_bytes; b += blockDim.x) {
        dst_row[b] = src_row[b];
    }
}

void ggml_cuda_op_moe_expert_copy(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * t_src_idx  = dst->src[0];
    const ggml_tensor * t_dst_idx  = dst->src[1];
    const ggml_tensor * t_num_copy = dst->src[2];
    const ggml_tensor * host_src   = dst->src[3];
    const ggml_tensor * pool       = dst->src[4];

    GGML_ASSERT(dst->data == pool->data); // result is a view of pool

    const size_t row_bytes = ggml_row_size(host_src->type, host_src->ne[0]);
    GGML_ASSERT(row_bytes == ggml_row_size(pool->type, pool->ne[0]));

    // Derived from t_src_idx's actual shape (a stable, persistent tensor -- see
    // llama_moe_expert_cache), not dst->op_params: op_params on a view-of-a-
    // persistent-tensor result is not reliably what the scheduler's graph-copy/
    // multi-copy machinery hands back to this dispatch function.
    const int32_t max_missing = (int32_t) ggml_nelements(t_src_idx);
    if (max_missing <= 0) {
        return;
    }

    const int threads = 256;
    moe_expert_copy_kernel<<<max_missing, threads, 0, ctx.stream()>>>(
        (const int32_t *) t_src_idx->data,
        (const int32_t *) t_dst_idx->data,
        (const int64_t *) t_num_copy->data,
        (const char *) host_src->data,
        (char *) dst->data,
        row_bytes);
}
