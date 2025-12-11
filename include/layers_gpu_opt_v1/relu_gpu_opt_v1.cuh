// ReLU - Optimized v1 (same as naive, already memory-efficient)
#pragma once
#include "data/gpu_tensor_opt.cuh"

void relu_forward_opt_v1(GPUTensorOpt& tensor);
void relu_backward_opt_v1(const GPUTensorOpt& grad_output, const GPUTensorOpt& forward_output, GPUTensorOpt& grad_input);
