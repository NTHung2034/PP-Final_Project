#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * Naive GPU Convolution2D - Forward Pass
 * 
 * Simple parallelization: 1 thread = 1 output element
 * Uses global memory only (no shared memory optimization)
 */
void conv2d_forward_gpu_naive(
    const GPUTensor& input,
    const GPUConvWeights& weights,
    GPUTensor& output,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    bool apply_relu = false);

/**
 * Naive GPU Convolution2D - Backward Pass
 * 
 * Computes gradients w.r.t. input, weights, and bias
 * Uses atomic operations for gradient accumulation
 */
void conv2d_backward_gpu_naive(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUConvWeights& weights,
    GPUTensor& grad_input,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding);
