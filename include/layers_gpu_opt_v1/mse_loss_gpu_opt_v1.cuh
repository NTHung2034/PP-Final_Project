#pragma once
#include "data/gpu_tensor_opt.cuh"

float mse_loss_forward_opt_v1(const GPUTensorOpt& prediction, const GPUTensorOpt& target);
void mse_loss_backward_opt_v1(const GPUTensorOpt& prediction, const GPUTensorOpt& target, GPUTensorOpt& grad_input);
