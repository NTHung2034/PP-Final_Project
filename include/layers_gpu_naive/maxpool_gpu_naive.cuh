#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * Naive GPU MaxPool2D - Forward Pass
 * 
 * Reduces spatial dimensions by taking maximum in each window
 * Stores indices for backward pass
 */
void maxpool2d_forward_gpu_naive(
    const GPUTensor& input,
    GPUTensor& output,
    int* pool_indices);

/**
 * Naive GPU MaxPool2D - Backward Pass
 * 
 * Routes gradients back to max positions using stored indices
 */
void maxpool2d_backward_gpu_naive(
    const GPUTensor& grad_output,
    const int* pool_indices,
    GPUTensor& grad_input);
