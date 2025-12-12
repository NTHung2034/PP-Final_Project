#pragma once
#include "data/gpu_tensor_opt_v2.cuh"

#define TILE_W_V2 16
#define TILE_H_V2 16
#define KERNEL_SIZE_V2 3
#define PAD_V2 1

// Fused Conv2D + Bias + ReLU forward (stream-enabled)
void conv2d_forward_opt_v2(const GPUTensorOpt& input, const GPUConvWeightsOpt& weights,
                           GPUTensorOpt& output, bool apply_relu, cudaStream_t stream = nullptr);

// Conv2D backward (stream-enabled)
void conv2d_backward_opt_v2(const GPUTensorOpt& input, const GPUTensorOpt& grad_output,
                            GPUConvWeightsOpt& weights, GPUTensorOpt& grad_input,
                            const GPUTensorOpt* forward_output, cudaStream_t stream = nullptr);
