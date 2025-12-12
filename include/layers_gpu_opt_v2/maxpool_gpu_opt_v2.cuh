#pragma once
#include "data/gpu_tensor_opt_v2.cuh"

void maxpool2d_forward_opt_v2(const GPUTensorOpt& input, GPUTensorOpt& output,
                               int* pool_indices, cudaStream_t stream = nullptr);

void maxpool2d_backward_opt_v2(const GPUTensorOpt& grad_output, const int* pool_indices,
                                GPUTensorOpt& grad_input, cudaStream_t stream = nullptr);
