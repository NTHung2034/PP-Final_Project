#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * =============================================================================
 * NAIVE UPSAMPLE GPU IMPLEMENTATION - PHASE 2
 * =============================================================================
 * 
 * Simple nearest neighbor upsampling:
 * - Each thread computes one output element
 * - Maps output coordinates back to input (integer division)
 * - Doubles spatial dimensions
 */

/**
 * Naive Upsample2D Forward Pass
 * 
 * Upsamples input by scale factor using nearest neighbor interpolation
 * Each thread handles one output element
 * 
 * @param input        Input tensor [N, C, H_in, W_in]
 * @param output       Output tensor [N, C, H_out, W_out] where H_out = H_in * scale
 * @param scale_factor Upsampling factor (typically 2)
 * @param stream       CUDA stream for async execution
 */
void upsample2d_forward_gpu_naive(
    const GPUTensor& input,
    GPUTensor& output,
    int scale_factor = 2,
    cudaStream_t stream = 0);

/**
 * Naive Upsample2D Backward Pass
 * 
 * Accumulates gradients from upsampled positions back to original positions
 * 
 * @param grad_output  Gradient from next layer [N, C, H_out, W_out]
 * @param grad_input   Gradient w.r.t. input [N, C, H_in, W_in]
 * @param scale_factor Upsampling factor (typically 2)
 * @param stream       CUDA stream for async execution
 */
void upsample2d_backward_gpu_naive(
    const GPUTensor& grad_output,
    GPUTensor& grad_input,
    int scale_factor = 2,
    cudaStream_t stream = 0);
