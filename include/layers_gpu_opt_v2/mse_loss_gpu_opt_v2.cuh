#pragma once
#include "data/gpu_tensor_opt_v2.cuh"

void mse_loss_forward_opt_v2(const GPUTensorOpt& prediction, const GPUTensorOpt& target,
                              float* d_loss, cudaStream_t stream = nullptr);

void mse_loss_backward_opt_v2(const GPUTensorOpt& prediction, const GPUTensorOpt& target,
                               GPUTensorOpt& grad_input, cudaStream_t stream = nullptr);
