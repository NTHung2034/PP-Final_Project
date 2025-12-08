#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * =============================================================================
 * NAIVE RELU GPU IMPLEMENTATION - PHASE 2
 * =============================================================================
 * 
 * Simple element-wise operation:
 * - Each thread processes one element: out = max(0, in)
 * - Straightforward parallelization
 */

/**
 * Naive ReLU Forward Pass
 * 
 * Applies ReLU activation in-place: data[i] = max(0, data[i])
 * Each thread handles one element
 * 
 * @param data    Input/output tensor (modified in-place)
 * @param stream  CUDA stream for async execution
 */
void relu_forward_gpu_naive(GPUTensor& data, cudaStream_t stream = 0);

/**
 * Naive ReLU Backward Pass
 * 
 * Computes gradient: grad_in[i] = input[i] > 0 ? grad_out[i] : 0
 * Each thread handles one element
 * 
 * @param input       Original input tensor (before ReLU)
 * @param grad_output Gradient from next layer
 * @param grad_input  Gradient w.r.t. input (output of this function)
 * @param stream      CUDA stream for async execution
 */
void relu_backward_gpu_naive(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUTensor& grad_input,
    cudaStream_t stream = 0);