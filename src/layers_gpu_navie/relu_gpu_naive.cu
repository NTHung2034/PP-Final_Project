#include "layers_gpu_naive/relu_gpu_naive.cuh"

__global__ void relu_forward_kernel_naive(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Simple ReLU: max(0, x)
        if (data[idx] < 0.0f) {
            data[idx] = 0.0f;
        }
    }
}

void relu_forward_gpu_naive(GPUTensor& data, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (data.size + threads - 1) / threads;
    
    relu_forward_kernel_naive<<<blocks, threads, 0, stream>>>(data.d_data, data.size);
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void relu_backward_kernel_naive(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Gradient passes through if input > 0, otherwise 0
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

void relu_backward_gpu_naive(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUTensor& grad_input,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (input.size + threads - 1) / threads;
    
    relu_backward_kernel_naive<<<blocks, threads, 0, stream>>>(
        input.d_data, grad_output.d_data, grad_input.d_data, input.size
    );
    
    CUDA_CHECK(cudaGetLastError());
}
