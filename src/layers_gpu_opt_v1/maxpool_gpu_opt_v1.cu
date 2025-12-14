#include "layers_gpu_opt_v1/maxpool_gpu_opt_v1.cuh"

__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ input, float* __restrict__ output, int* __restrict__ indices,
    int N, int C, int H_in, int W_in, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;
    
    int w_out = idx % W_out; int tmp = idx / W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;
    
    float max_val = -INFINITY;
    int max_idx = 0;
    
    for (int ph = 0; ph < 2; ph++) {
        for (int pw = 0; pw < 2; pw++) {
            int h_in = h_out * 2 + ph;
            int w_in = w_out * 2 + pw;
            int in_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;
            if (input[in_idx] > max_val) {
                max_val = input[in_idx];
                max_idx = in_idx;
            }
        }
    }
    output[idx] = max_val;
    indices[idx] = max_idx;
}

__global__ void maxpool2d_backward_kernel(
    const float* __restrict__ grad_out, const int* __restrict__ indices,
    float* __restrict__ grad_in, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) atomicAdd(&grad_in[indices[idx]], grad_out[idx]);
}

void maxpool2d_forward_opt_v1(const GPUTensorOpt& input, GPUTensorOpt& output, int* pool_indices) {
    int total = output.size;
    maxpool2d_forward_kernel<<<(total + 255) / 256, 256>>>(
        input.d_data, output.d_data, pool_indices,
        input.batch, input.channels, input.height, input.width, output.height, output.width);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool2d_backward_opt_v1(const GPUTensorOpt& grad_output, const int* pool_indices, GPUTensorOpt& grad_input) {
    grad_input.zero();
    maxpool2d_backward_kernel<<<(grad_output.size + 255) / 256, 256>>>(
        grad_output.d_data, pool_indices, grad_input.d_data, grad_output.size);
    CUDA_CHECK(cudaGetLastError());
}
