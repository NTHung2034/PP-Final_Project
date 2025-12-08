#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * =============================================================================
 * NAIVE MAXPOOL GPU IMPLEMENTATION - PHASE 2
 * =============================================================================
 * 
 * Simple 2x2 max pooling:
 * - Each thread computes one output element
 * - Finds maximum value in 2x2 window
 * - Stores index of maximum for backward pass
 */

/**
 * Naive MaxPool2D Forward Pass
 * 
 * Performs 2x2 max pooling with stride 2
 * Each thread handles one output element and finds max in 2x2 window
 * 
 * @param input         Input tensor [N, C, H_in, W_in]
 * @param output        Output tensor [N, C, H_out, W_out] where H_out = H_in/2
 * @param d_max_indices Device pointer to store max indices (needed for backward)
 * @param stream        CUDA stream for async execution
 */
void maxpool2d_forward_gpu_naive(
    const GPUTensor& input,
    GPUTensor& output,
    int* d_max_indices,
    cudaStream_t stream = 0);

/**
 * Naive MaxPool2D Backward Pass
 * 
 * Distributes gradient to position where max value was found
 * Uses stored indices from forward pass
 * 
 * @param grad_output   Gradient from next layer [N, C, H_out, W_out]
 * @param d_max_indices Indices stored during forward pass
 * @param grad_input    Gradient w.r.t. input [N, C, H_in, W_in]
 * @param stream        CUDA stream for async execution
 */
void maxpool2d_backward_gpu_naive(
    const GPUTensor& grad_output,
    const int* d_max_indices,
    GPUTensor& grad_input,
    cudaStream_t stream = 0);


