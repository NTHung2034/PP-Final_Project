#pragma once
#include "data/gpu_tensor_opt_v2.cuh"

// Standalone ReLU (for cases not fused with conv)
void relu_forward_opt_v2(GPUTensorOpt& tensor, cudaStream_t stream = nullptr);

void relu_backward_opt_v2(const GPUTensorOpt& grad_output, const GPUTensorOpt& forward_output,
                          GPUTensorOpt& grad_input, cudaStream_t stream = nullptr);
