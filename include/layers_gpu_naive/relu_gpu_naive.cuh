#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>

void relu_forward_gpu_naive(
    GPUTensor& tensor);

    
void relu_backward_gpu_naive(
    const GPUTensor& grad_output,
    const GPUTensor& forward_output,
    GPUTensor& grad_input);
