// Tiled Convolution with Shared Memory - Optimized v1
#pragma once
#include "data/gpu_tensor_opt.cuh"

// Tile dimensions for shared memory
#define TILE_W 16
#define TILE_H 16
#define KERNEL_SIZE 3
#define PAD 1

// Forward declarations
void conv2d_forward_opt_v1(const GPUTensorOpt& input, const GPUConvWeightsOpt& weights,
                           GPUTensorOpt& output, bool apply_relu = true);

void conv2d_backward_opt_v1(const GPUTensorOpt& input, const GPUTensorOpt& grad_output,
                            GPUConvWeightsOpt& weights, GPUTensorOpt& grad_input);
