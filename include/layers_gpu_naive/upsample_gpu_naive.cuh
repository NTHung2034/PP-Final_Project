#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

void upsample2d_forward_gpu_naive(
    const GPUTensor& input,
    GPUTensor& output,
    int scale_factor);

void upsample2d_backward_gpu_naive(
    const GPUTensor& grad_output,
    GPUTensor& grad_input,
    int scale_factor);
