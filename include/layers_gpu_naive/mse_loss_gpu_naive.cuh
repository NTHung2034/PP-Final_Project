#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * Naive GPU MSE Loss - Forward Pass
 * 
 * Computes mean squared error: MSE = mean((pred - target)^2)
 * Returns scalar loss value
 */
float mse_loss_forward_gpu_naive(
    const GPUTensor& prediction,
    const GPUTensor& target);

/**
 * Naive GPU MSE Loss - Backward Pass
 * 
 * Gradient: d(MSE)/d(pred) = 2 * (pred - target) / N
 */
void mse_loss_backward_gpu_naive(
    const GPUTensor& prediction,
    const GPUTensor& target,
    GPUTensor& grad_input);
