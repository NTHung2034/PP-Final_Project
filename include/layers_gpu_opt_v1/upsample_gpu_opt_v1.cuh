// Upsample 2x (Nearest Neighbor) - Optimized v1
#pragma once
#include "data/gpu_tensor_opt.cuh"

void upsample2d_forward_opt_v1(const GPUTensorOpt& input, GPUTensorOpt& output);
void upsample2d_backward_opt_v1(const GPUTensorOpt& grad_output, GPUTensorOpt& grad_input);
