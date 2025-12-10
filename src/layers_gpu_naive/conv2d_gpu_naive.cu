#include "layers_gpu_naive/conv2d_gpu_naive.cuh"

// ============================================================================
// CONVOLUTION FORWARD KERNEL (Naive)
// ============================================================================
__global__ void conv2d_forward_kernel_naive(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kH, int kW,
    int stride, int padding,
    bool apply_relu)
{
    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;
    
    int h_out = hw / W_out;
    int w_out = hw % W_out;
    
    if (n >= N || c_out >= C_out || h_out >= H_out || w_out >= W_out) return;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < C_in; c_in++) {       // iterate over input channels
        for (int kh = 0; kh < kH; kh++) {           // iterate over kernel height
            for (int kw = 0; kw < kW; kw++) {       // iterate over kernel width

                int w_in = w_out * stride - padding + kw;
                int h_in = h_out * stride - padding + kh;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in;
                    int weight_idx = c_out * (C_in * kH * kW) + c_in * (kH * kW) + kh * kW + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    sum += bias[c_out];
    
    if (apply_relu && sum < 0.0f) {
        sum = 0.0f;
    }
    
    int output_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
    output[output_idx] = sum;
}

void conv2d_forward_gpu_naive(
    const GPUTensor& input,
    const GPUConvWeights& weights,
    GPUTensor& output,
    int kernel_h, int kernel_w,
    int stride, int padding,
    bool apply_relu,
    cudaStream_t stream)
{
    int N = input.batch;
    int C_in = input.channels;
    int H_in = input.height;
    int W_in = input.width;
    int C_out = weights.out_channels;
    
    int H_out = (H_in + 2 * padding - kernel_h) / stride + 1;
    int W_out = (W_in + 2 * padding - kernel_w) / stride + 1;
    
    int threads = 256;
    int blocks_z = (H_out * W_out + threads - 1) / threads;
    dim3 grid(N, C_out, blocks_z);
    dim3 block(threads);
    
    conv2d_forward_kernel_naive<<<grid, block, 0, stream>>>(
        input.d_data, weights.d_weights, weights.d_bias, output.d_data,
        N, C_in, H_in, W_in, C_out, H_out, W_out,
        kernel_h, kernel_w, stride, padding, apply_relu
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// CONVOLUTION BACKWARD KERNELS (Naive)
// ============================================================================
__global__ void conv2d_backward_input_kernel_naive(
    const float* __restrict__ grad_output,
    const float* __restrict__ weights,
    float* __restrict__ grad_input,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kH, int kW,
    int stride, int padding)
{
    int n = blockIdx.x;
    int c_in = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;
    
    int h_in = hw / W_in;
    int w_in = hw % W_in;
    
    if (n >= N || c_in >= C_in || h_in >= H_in || w_in >= W_in) return;
    
    float grad = 0.0f;
    
    for (int c_out = 0; c_out < C_out; c_out++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int h_out = (h_in + padding - kh);
                int w_out = (w_in + padding - kw);
                
                if (h_out % stride == 0 && h_out >= 0 && w_out % stride == 0 && w_out >= 0) {
                    h_out /= stride;
                    w_out /= stride;
                    
                    if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                        int grad_out_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
                        int weight_idx = c_out * (C_in * kH * kW) + c_in * (kH * kW) + kh * kW + kw;
                        grad += grad_output[grad_out_idx] * weights[weight_idx];
                    }
                }
            }
        }
    }
    
    int grad_in_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in;
    grad_input[grad_in_idx] = grad; 
}

__global__ void conv2d_backward_weights_kernel_naive(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_weights,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kH, int kW,
    int stride, int padding)
{
    int c_out = blockIdx.x;
    int c_in = blockIdx.y;
    int khw = blockIdx.z * blockDim.x + threadIdx.x;
    
    int kh = khw / kW;
    int kw = khw % kW;
    
    if (c_out >= C_out || c_in >= C_in || kh >= kH || kw >= kW) return;
    
    float grad_w = 0.0f;
    
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int grad_out_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
                    int input_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in;
                    
                    grad_w += grad_output[grad_out_idx] * input[input_idx];
                }
            }
        }
    }
    
    int weight_idx = c_out * (C_in * kH * kW) + c_in * (kH * kW) + kh * kW + kw;
    grad_weights[weight_idx] = grad_w;
}

__global__ void conv2d_backward_bias_kernel_naive(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int N, int C_out, int H_out, int W_out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C_out) return;
    
    float grad_b = 0.0f;
    
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                int idx = n * (C_out * H_out * W_out) + c * (H_out * W_out) + h * W_out + w;
                grad_b += grad_output[idx];
            }
        }
    }
    
    grad_bias[c] = grad_b;
}

void conv2d_backward_gpu_naive(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUConvWeights& weights,
    GPUTensor& grad_input,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding)
{
    int N = input.batch;
    int C_in = input.channels;
    int H_in = input.height;
    int W_in = input.width;
    int C_out = weights.out_channels;
    int H_out = grad_output.height;
    int W_out = grad_output.width;
    
    // Gradient w.r.t. input
    {
        int threads = 256;
        int blocks_z = (H_in * W_in + threads - 1) / threads;
        dim3 grid(N, C_in, blocks_z);
        dim3 block(threads);
        
        conv2d_backward_input_kernel_naive<<<grid, block>>>(
            grad_output.d_data, weights.d_weights, grad_input.d_data,
            N, C_in, H_in, W_in, C_out, H_out, W_out,
            kernel_h, kernel_w, stride, padding
        );
    }
    
    // Gradient w.r.t. weights
    {
        int threads = 256;
        int blocks_z = (kernel_h * kernel_w + threads - 1) / threads;
        dim3 grid(C_out, C_in, blocks_z);
        dim3 block(threads);
        
        conv2d_backward_weights_kernel_naive<<<grid, block>>>(
            grad_output.d_data, input.d_data, weights.d_grad_w,
            N, C_in, H_in, W_in, C_out, H_out, W_out,
            kernel_h, kernel_w, stride, padding
        );
    }
    
    // Gradient w.r.t. bias
    {
        int threads = 256;
        int blocks = (C_out + threads - 1) / threads;
        
        conv2d_backward_bias_kernel_naive<<<blocks, threads>>>(
            grad_output.d_data, weights.d_grad_b,
            N, C_out, H_out, W_out
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}