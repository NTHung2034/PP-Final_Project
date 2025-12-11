#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * Naive GPU ReLU - Forward Pass
 * 
 * Element-wise activation: f(x) = max(0, x)
 * In-place operation
 */
void relu_forward_gpu_naive(
    GPUTensor& tensor);

/**
 * Naive GPU ReLU - Backward Pass
 * 
 * Gradient: d(ReLU)/dx = 1 if x > 0, else 0
 */
void relu_backward_gpu_naive(
    const GPUTensor& grad_output,
    const GPUTensor& forward_output,
    GPUTensor& grad_input);
