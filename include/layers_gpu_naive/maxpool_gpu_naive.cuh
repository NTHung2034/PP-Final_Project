#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>


void maxpool2d_forward_gpu_naive(
    const GPUTensor& input,
    GPUTensor& output,
    int* pool_indices);

void maxpool2d_backward_gpu_naive(
    const GPUTensor& grad_output,
    const int* pool_indices,
    GPUTensor& grad_input);
