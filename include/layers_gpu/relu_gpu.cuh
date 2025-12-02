#ifndef RELU_GPU_CUH
#define RELU_GPU_CUH

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * =============================================================================
 * RELU ACTIVATION GPU LAYER - Header
 * =============================================================================
 * 
 * Element-wise ReLU activation: y = max(0, x)
 * Standalone version for decoder layers (can also be fused with Conv2D)
 */

/**
 * ReLU Forward Pass (In-place)
 * 
 * Applies ReLU activation in-place: data[i] = max(0, data[i])
 * 
 * @param data   Input/Output tensor (modified in-place)
 * @param stream CUDA stream for async execution
 */
void relu_forward_gpu(
    GPUTensor& data,
    cudaStream_t stream = 0);

/**
 * ReLU Backward Pass
 * 
 * Computes gradient: grad_input[i] = (input[i] > 0) ? grad_output[i] : 0
 * Gradient flows through only where input was positive
 * 
 * @param input       Original input tensor (before ReLU)
 * @param grad_output Gradient from next layer
 * @param grad_input  Gradient w.r.t. input
 * @param stream      CUDA stream for async execution
 */
void relu_backward_gpu(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUTensor& grad_input,
    cudaStream_t stream = 0);

#endif // RELU_GPU_CUH
