#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * =============================================================================
 * MSE LOSS GPU - Header
 * =============================================================================
 * 
 * Mean Squared Error loss computation using hierarchical reduction
 * Efficient shared memory reduction for computing loss across all elements
 */

/**
 * MSE Loss Forward Pass
 * 
 * Computes Mean Squared Error: MSE = (1/N) * sum((predicted - target)^2)
 * Uses hierarchical reduction with shared memory for efficiency
 * 
 * @param predicted Predicted tensor [N, C, H, W]
 * @param target    Target/ground truth tensor [N, C, H, W]
 * @param stream    CUDA stream for async execution
 * @return          Mean squared error value (scalar)
 */
float mse_loss_forward_gpu(
    const GPUTensor& predicted,
    const GPUTensor& target,
    cudaStream_t stream = 0);

/**
 * MSE Loss Backward Pass
 * 
 * Computes gradient: grad = 2 * (predicted - target) / N
 * 
 * @param predicted Predicted tensor [N, C, H, W]
 * @param target    Target/ground truth tensor [N, C, H, W]
 * @param grad      Output gradient tensor [N, C, H, W]
 * @param stream    CUDA stream for async execution
 */
void mse_loss_backward_gpu(
    const GPUTensor& predicted,
    const GPUTensor& target,
    GPUTensor& grad,
    cudaStream_t stream = 0);

