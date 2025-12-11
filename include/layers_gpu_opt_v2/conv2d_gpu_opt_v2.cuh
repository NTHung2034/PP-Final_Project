// Conv2D GPU Optimized v2 - Header
// Optimizations: Streams, Fused ReLU Backward, Vectorized Access, Full Unrolling
#pragma once
#include "data/gpu_tensor_opt.cuh"

#define TILE_H 16
#define TILE_W 16
#define KERNEL_SIZE 3
#define PAD 1

// Forward convolution (same as v1, already fused with ReLU)
void conv2d_forward_opt_v2(
    const GPUTensorOpt& input, 
    const GPUConvWeightsOpt& weights, 
    GPUTensorOpt& output, 
    bool apply_relu,
    cudaStream_t stream = nullptr);

// Backward with streams - grad_input, grad_weights, grad_bias computed in parallel streams
void conv2d_backward_opt_v2(
    const GPUTensorOpt& input,
    const GPUTensorOpt& grad_output,
    const GPUTensorOpt& forward_output,  // For fused ReLU backward
    GPUConvWeightsOpt& weights,
    GPUTensorOpt& grad_input,
    bool had_relu,                        // Whether forward had ReLU
    cudaStream_t stream1 = nullptr,       // For grad_input
    cudaStream_t stream2 = nullptr,       // For grad_weights
    cudaStream_t stream3 = nullptr);      // For grad_bias

// Synchronize all streams
void conv2d_sync_streams(cudaStream_t s1, cudaStream_t s2, cudaStream_t s3);
