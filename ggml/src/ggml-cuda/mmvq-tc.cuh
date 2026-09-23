#pragma once

#include "common.cuh"

// M <= 8 W4A16 tensor-core path for k-quants. Volta only.
bool ggml_cuda_should_use_mmvq_tc(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_vec_q_tc(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
