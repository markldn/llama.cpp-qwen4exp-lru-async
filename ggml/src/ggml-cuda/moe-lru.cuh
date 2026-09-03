#include "common.cuh"

void ggml_cuda_op_moe_lru_ensure(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_expert_copy(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
