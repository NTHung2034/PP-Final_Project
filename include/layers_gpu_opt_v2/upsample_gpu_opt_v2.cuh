// Upsample 2x GPU Optimized v2 - Header
#pragma once
#include "data/gpu_tensor_opt.cuh"

void upsample2d_forward_opt_v2(
    const GPUTensorOpt& input, 
    GPUTensorOpt& output,
    cudaStream_t stream = nullptr);

void upsample2d_backward_opt_v2(
    const GPUTensorOpt& grad_output, 
    GPUTensorOpt& grad_input,
    cudaStream_t stream = nullptr);
