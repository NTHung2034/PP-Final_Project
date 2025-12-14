#include "layers_gpu_naive/upsample_gpu_naive.cuh"

__global__ void upsample2d_forward_kernel_naive(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int scale_factor)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;
    
    int h_out = hw / W_out;
    int w_out = hw % W_out;
    
    if (n >= N || c >= C || h_out >= H_out || w_out >= W_out) return;
    
    int h_in = h_out / scale_factor;
    int w_in = w_out / scale_factor;
    
    int input_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;
    int output_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
    
    output[output_idx] = input[input_idx];
}

void upsample2d_forward_gpu_naive(
    const GPUTensor& input,
    GPUTensor& output,
    int scale_factor)
{
    int N = input.batch;
    int C = input.channels;
    int H_out = output.height;
    int W_out = output.width;
    
    int threads = 256;
    int blocks_z = (H_out * W_out + threads - 1) / threads;
    dim3 grid(N, C, blocks_z);
    dim3 block(threads);
    
    upsample2d_forward_kernel_naive<<<grid, block>>>(
        input.d_data, output.d_data,
        N, C, input.height, input.width, H_out, W_out,
        scale_factor
    );
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void upsample2d_backward_kernel_naive(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int scale_factor)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;
    
    int h_in = hw / W_in;
    int w_in = hw % W_in;
    
    if (n >= N || c >= C || h_in >= H_in || w_in >= W_in) return;
    
    float grad = 0.0f;
    
    for (int dh = 0; dh < scale_factor; dh++) {
        for (int dw = 0; dw < scale_factor; dw++) {
            int h_out = h_in * scale_factor + dh;
            int w_out = w_in * scale_factor + dw;
            
            if (h_out < H_out && w_out < W_out) {
                int output_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
                grad += grad_output[output_idx];
            }
        }
    }
    
    int input_idx = n * (C * H_in * W_in) + 
                   c * (H_in * W_in) + 
                   h_in * W_in + w_in;
    grad_input[input_idx] = grad;
}

void upsample2d_backward_gpu_naive(
    const GPUTensor& grad_output,
    GPUTensor& grad_input,
    int scale_factor)
{
    int N = grad_input.batch;
    int C = grad_input.channels;
    int H_in = grad_input.height;
    int W_in = grad_input.width;
    int H_out = grad_output.height;
    int W_out = grad_output.width;
    
    int threads = 256;
    int blocks_z = (H_in * W_in + threads - 1) / threads;
    dim3 grid(N, C, blocks_z);
    dim3 block(threads);
    
    upsample2d_backward_kernel_naive<<<grid, block>>>(
        grad_output.d_data, grad_input.d_data,
        N, C, H_in, W_in, H_out, W_out,
        scale_factor
    );
    
    CUDA_CHECK(cudaGetLastError());
}
