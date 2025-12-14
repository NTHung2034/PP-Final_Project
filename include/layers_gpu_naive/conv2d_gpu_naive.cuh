#pragma once

#include "data/gpu_data_types.cuh"
#include <cuda_runtime.h>


void conv2d_forward_gpu_naive(
    const GPUTensor& input,
    const GPUConvWeights& weights,
    GPUTensor& output,
    int kernel_h, int kernel_w,
    int stride, int padding,
    bool apply_relu = false);

void conv2d_backward_gpu_naive(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUConvWeights& weights,
    GPUTensor& grad_input,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding);
