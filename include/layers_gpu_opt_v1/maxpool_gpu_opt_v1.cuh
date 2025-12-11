// MaxPool 2x2 - Optimized v1
#pragma once
#include "data/gpu_tensor_opt.cuh"

void maxpool2d_forward_opt_v1(const GPUTensorOpt& input, GPUTensorOpt& output, int* pool_indices);
void maxpool2d_backward_opt_v1(const GPUTensorOpt& grad_output, const int* pool_indices, GPUTensorOpt& grad_input);
