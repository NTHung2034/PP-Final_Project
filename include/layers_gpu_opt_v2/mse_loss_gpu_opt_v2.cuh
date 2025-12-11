// MSE Loss GPU Optimized v2 - Header
#pragma once
#include "data/gpu_tensor_opt.cuh"

float mse_loss_forward_opt_v2(
    const GPUTensorOpt& output, 
    const GPUTensorOpt& target,
    cudaStream_t stream = nullptr);

void mse_loss_backward_opt_v2(
    const GPUTensorOpt& output, 
    const GPUTensorOpt& target, 
    GPUTensorOpt& grad_output,
    cudaStream_t stream = nullptr);
