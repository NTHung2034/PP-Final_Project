#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * =============================================================================
 * NAIVE CONV2D GPU IMPLEMENTATION - PHASE 2
 * =============================================================================
 * 
 * Simple parallelization strategy:
 * - Each thread computes one output pixel
 * - Uses global memory for all reads/writes
 * - No shared memory optimization
 * - Sequential processing of input channels and kernel elements
 */

/**
 * Naive Conv2D Forward Pass
 * 
 * Each thread computes one output element by:
 * 1. Looping over all input channels
 * 2. For each channel, loop over kernel elements
 * 3. Accumulate weighted sum
 * 4. Add bias and optionally apply ReLU
 * 
 * @param input       Input tensor [N, C_in, H_in, W_in]
 * @param weights     Convolution weights and biases
 * @param output      Output tensor [N, C_out, H_out, W_out]
 * @param kernel_h    Kernel height
 * @param kernel_w    Kernel width
 * @param padding     Padding size
 * @param stride      Stride size
 * @param apply_relu  Whether to apply ReLU activation
 * @param stream      CUDA stream for async execution
 */
void conv2d_forward_gpu_naive(
    const GPUTensor& input,
    const GPUConvWeights& weights,
    GPUTensor& output,
    int kernel_h, int kernel_w,
    int padding, int stride,
    bool apply_relu,
    cudaStream_t stream = 0);

/**
 * Naive Conv2D Backward Pass
 * 
 * Computes gradients using simple parallelization:
 * - Gradient w.r.t. input: Each thread handles one input element
 * - Gradient w.r.t. weights: Uses atomic operations for accumulation
 * 
 * @param input        Original input tensor
 * @param grad_output  Gradient from next layer
 * @param weights      Convolution weights (gradients stored here)
 * @param grad_input   Gradient w.r.t. input
 * @param kernel_h     Kernel height
 * @param kernel_w     Kernel width
 * @param padding      Padding size
 * @param stride       Stride size
 * @param stream       CUDA stream for async execution
 */
void conv2d_backward_gpu_naive(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUConvWeights& weights,
    GPUTensor& grad_input,
    int kernel_h, int kernel_w,
    int padding, int stride,
    cudaStream_t stream = 0);

