#ifndef CONV2D_GPU_CUH
#define CONV2D_GPU_CUH

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * =============================================================================
 * CONV2D GPU LAYER - Header
 * =============================================================================
 * 
 * Optimized Conv2D implementation with shared memory tiling
 * Includes fused Conv2D + Bias + ReLU kernel
 */

// Tile size for shared memory (must match block dimensions in implementation)
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

/**
 * Conv2D Forward Pass
 * 
 * Performs convolution with optional ReLU activation (kernel fusion)
 * Uses shared memory tiling for improved memory access efficiency
 * 
 * @param input       Input tensor [N, C_in, H_in, W_in]
 * @param weights     Convolution weights and biases
 * @param output      Output tensor [N, C_out, H_out, W_out]
 * @param kernel_h    Kernel height
 * @param kernel_w    Kernel width
 * @param padding     Padding size
 * @param stride      Stride size
 * @param apply_relu  Whether to apply ReLU activation (fused)
 * @param stream      CUDA stream for async execution
 */
void conv2d_forward_gpu(
    const GPUTensor& input,
    const GPUConvWeights& weights,
    GPUTensor& output,
    int kernel_h, int kernel_w,
    int padding, int stride,
    bool apply_relu,
    cudaStream_t stream = 0);

/**
 * Conv2D Backward Pass
 * 
 * Computes gradients for:
 * - Input (for backpropagation)
 * - Weights (for parameter update)
 * - Bias (for parameter update)
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
void conv2d_backward_gpu(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUConvWeights& weights,
    GPUTensor& grad_input,
    int kernel_h, int kernel_w,
    int padding, int stride,
    cudaStream_t stream = 0);

#endif // CONV2D_GPU_CUH
