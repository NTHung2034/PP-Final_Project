#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * Naive GPU Upsample2D - Forward Pass
 * 
 * Nearest neighbor upsampling
 * Duplicates each pixel by scale_factor in both dimensions
 */
void upsample2d_forward_gpu_naive(
    const GPUTensor& input,
    GPUTensor& output,
    int scale_factor);

/**
 * Naive GPU Upsample2D - Backward Pass
 * 
 * Sums gradients from duplicated positions back to source
 */
void upsample2d_backward_gpu_naive(
    const GPUTensor& grad_output,
    GPUTensor& grad_input,
    int scale_factor);
