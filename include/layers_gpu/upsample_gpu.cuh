#ifndef UPSAMPLE_GPU_CUH
#define UPSAMPLE_GPU_CUH

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

/**
 * =============================================================================
 * UPSAMPLE2D GPU LAYER (Nearest Neighbor) - Header
 * =============================================================================
 * 
 * Upsamples spatial dimensions using nearest neighbor interpolation
 * Default scale factor is 2x in both dimensions
 */

/**
 * Upsample2D Forward Pass (Nearest Neighbor)
 * 
 * Each output pixel copies from its nearest input pixel
 * Output dimensions: H_out = H_in * scale_factor, W_out = W_in * scale_factor
 * 
 * @param input        Input tensor [N, C, H_in, W_in]
 * @param output       Output tensor [N, C, H_out, W_out]
 * @param scale_factor Upsampling factor (default: 2)
 * @param stream       CUDA stream for async execution
 */
void upsample2d_forward_gpu(
    const GPUTensor& input,
    GPUTensor& output,
    int scale_factor = 2,
    cudaStream_t stream = 0);

/**
 * Upsample2D Backward Pass
 * 
 * Accumulates gradients from all output pixels that map to same input pixel
 * Each input gradient is the sum of scale_factor^2 output gradients
 * 
 * @param grad_output  Gradient from next layer [N, C, H_out, W_out]
 * @param grad_input   Gradient w.r.t. input [N, C, H_in, W_in]
 * @param scale_factor Upsampling factor (default: 2)
 * @param stream       CUDA stream for async execution
 */
void upsample2d_backward_gpu(
    const GPUTensor& grad_output,
    GPUTensor& grad_input,
    int scale_factor = 2,
    cudaStream_t stream = 0);

#endif // UPSAMPLE_GPU_CUH
