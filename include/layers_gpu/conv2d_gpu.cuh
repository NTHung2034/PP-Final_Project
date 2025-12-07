#ifndef CONV2D_GPU_CUH
#define CONV2D_GPU_CUH

#include "gpu_data_types.cuh"

// Forward pass for 2D convolution with optional ReLU activation

// Performs: output = ReLU(Conv2D(input, weights) + bias)

// input Input tensor [N, C_in, H_in, W_in]
// weights Convolution weights [C_out, C_in, kH, kW]
// output Output tensor [N, C_out, H_out, W_out]
// kernel_h Kernel height
// kernel_w Kernel width
// padding Padding applied to input (same on all sides)
// stride Stride for convolution
// apply_relu Whether to apply ReLU activation
// stream CUDA stream for asynchronous execution

void conv2d_forward_gpu(
    const GPUTensor& input,
    const GPUConvWeights& weights,
    GPUTensor& output,
    int kernel_h, int kernel_w,
    int padding, int stride,
    bool apply_relu,
    cudaStream_t stream = 0
);


// Backward pass for 2D convolution
// 
// Computes:
// - grad_input: gradient w.r.t. input
// - grad_weights: gradient w.r.t. weights (accumulated)
// - grad_bias: gradient w.r.t. bias (accumulated)
// 
// input Original input tensor [N, C_in, H_in, W_in]
// grad_output Gradient w.r.t. output [N, C_out, H_out, W_out]
// weights Convolution weights (also stores gradients)
// grad_input Output: gradient w.r.t. input [N, C_in, H_in, W_in]
// kernel_h Kernel height
// kernel_w Kernel width
// padding Padding applied to input
// stride Stride for convolution
// stream CUDA stream for asynchronous execution

 void conv2d_backward_gpu(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUConvWeights& weights,
    GPUTensor& grad_input,
    int kernel_h, int kernel_w,
    int padding, int stride,
    cudaStream_t stream = 0
);

// Helper function to calculate output dimensions

inline void calc_conv2d_output_size(
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int padding, int stride,
    int& out_h, int& out_w)
{
    out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
}

#endif