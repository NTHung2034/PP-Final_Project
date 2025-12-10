#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * =============================================================================
 * MAXPOOL2D GPU LAYER - Header
 * =============================================================================
 * 
 * 2x2 max pooling with stride 2 (downsamples by 2x)
 * Stores max indices for backward pass gradient routing
 */

/**
 * MaxPool2D Forward Pass
 * 
 * Performs 2x2 max pooling with stride 2
 * Each output element is the maximum of a 2x2 input window
 * 
 * @param input         Input tensor [N, C, H_in, W_in]
 * @param output        Output tensor [N, C, H_out, W_out] where H_out = H_in/2
 * @param d_max_indices Device pointer to store max indices for backward pass
 * @param stream        CUDA stream for async execution
 */
void maxpool2d_forward_gpu(
    const GPUTensor& input,
    GPUTensor& output,
    int* d_max_indices,
    cudaStream_t stream = 0);

/**
 * MaxPool2D Backward Pass
 * 
 * Routes gradient only through the maximum element position
 * Uses stored max indices from forward pass
 * 
 * @param grad_output   Gradient from next layer [N, C, H_out, W_out]
 * @param d_max_indices Device pointer with max indices from forward pass
 * @param grad_input    Gradient w.r.t. input [N, C, H_in, W_in]
 * @param stream        CUDA stream for async execution
 */
void maxpool2d_backward_gpu(
    const GPUTensor& grad_output,
    const int* d_max_indices,
    GPUTensor& grad_input,
    cudaStream_t stream = 0);
 
