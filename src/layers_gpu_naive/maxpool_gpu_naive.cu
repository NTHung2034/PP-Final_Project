#include "layers_gpu_naive/maxpool_gpu_naive.cuh"

__global__ void maxpool2d_forward_kernel_naive(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ pool_indices,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int pool_size, int stride)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;
    
    int h_out = hw / W_out;
    int w_out = hw % W_out;
    
    if (n >= N || c >= C || h_out >= H_out || w_out >= W_out) return;
    
    float max_val = -INFINITY;
    int max_idx = 0;
    
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int h_in = h_out * stride + ph;
            int w_in = w_out * stride + pw;
            
            if (h_in < H_in && w_in < W_in) {
                int input_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in; 
                
                if (input[input_idx] > max_val) {
                    max_val = input[input_idx];
                    max_idx = input_idx;
                }
            }
        }
    }
    
    int output_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
    
    output[output_idx] = max_val;
    pool_indices[output_idx] = max_idx;
}

void maxpool2d_forward_gpu_naive(
    const GPUTensor& input,
    GPUTensor& output,
    int* pool_indices)
{
    int N = input.batch;
    int C = input.channels;
    int H_in = input.height;
    int W_in = input.width;
    int H_out = output.height;
    int W_out = output.width;
    
    int pool_size = 2;
    int stride = 2;
    
    int threads = 256;
    int blocks_z = (H_out * W_out + threads - 1) / threads;
    dim3 grid(N, C, blocks_z);
    dim3 block(threads);
    
    maxpool2d_forward_kernel_naive<<<grid, block>>>(
        input.d_data, output.d_data, pool_indices,
        N, C, H_in, W_in, H_out, W_out,
        pool_size, stride
    );
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void maxpool2d_backward_kernel_naive(
    const float* __restrict__ grad_output,
    const int* __restrict__ pool_indices,
    float* __restrict__ grad_input,
    int N, int C, int H_out, int W_out)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;
    
    int h_out = hw / W_out;
    int w_out = hw % W_out;
    
    if (n >= N || c >= C || h_out >= H_out || w_out >= W_out) return;
    
    int output_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
    
    int max_idx = pool_indices[output_idx];
    float grad = grad_output[output_idx];
    
    atomicAdd(&grad_input[max_idx], grad);
}

void maxpool2d_backward_gpu_naive(
    const GPUTensor& grad_output,
    const int* pool_indices,
    GPUTensor& grad_input)
{
    grad_input.zero();
    
    int N = grad_output.batch;
    int C = grad_output.channels;
    int H_out = grad_output.height;
    int W_out = grad_output.width;
    
    int threads = 256;
    int blocks_z = (H_out * W_out + threads - 1) / threads;
    dim3 grid(N, C, blocks_z);
    dim3 block(threads);
    
    maxpool2d_backward_kernel_naive<<<grid, block>>>(
        grad_output.d_data, pool_indices, grad_input.d_data,
        N, C, H_out, W_out
    );
    
    CUDA_CHECK(cudaGetLastError());
}
