#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * =============================================================================
 * NAIVE MSE LOSS GPU IMPLEMENTATION - PHASE 2
 * =============================================================================
 * 
 * Mean Squared Error loss for autoencoder reconstruction:
 * - Forward: Computes sum of squared differences using reduction
 * - Backward: Computes gradient = 2*(pred - target)/N
 */

/**
 * Naive MSE Loss Forward Pass
 * 
 * Computes MSE = (1/N) * Σ(predicted - target)²
 * Uses parallel reduction with shared memory and atomic operations
 * 
 * @param predicted  Predicted output tensor
 * @param target     Target tensor (ground truth)
 * @param stream     CUDA stream for async execution
 * @return           Mean squared error value
 */
float mse_loss_forward_gpu_naive(
    const GPUTensor& predicted,
    const GPUTensor& target,
    cudaStream_t stream = 0);

/**
 * Naive MSE Loss Backward Pass
 * 
 * Computes gradient: grad = 2 * (predicted - target) / N
 * Each thread handles one element
 * 
 * @param predicted  Predicted output tensor
 * @param target     Target tensor (ground truth)
 * @param grad       Gradient output
 * @param stream     CUDA stream for async execution
 */
void mse_loss_backward_gpu_naive(
    const GPUTensor& predicted,
    const GPUTensor& target,
    GPUTensor& grad,
    cudaStream_t stream = 0);

#pragma once