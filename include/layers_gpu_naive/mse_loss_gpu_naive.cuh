#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

float mse_loss_forward_gpu_naive(
    const GPUTensor& prediction,
    const GPUTensor& target);

void mse_loss_backward_gpu_naive(
    const GPUTensor& prediction,
    const GPUTensor& target,
    GPUTensor& grad_input);
