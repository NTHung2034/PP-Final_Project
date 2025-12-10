#include "layers_gpu_naive/relu_gpu_naive.cuh"

// ============================================================================
// ReLU FORWARD KERNEL (Naive)
// ============================================================================
__global__ void relu_forward_kernel_naive(
    float* __restrict__ data,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

void relu_forward_gpu_naive(
    GPUTensor& tensor,
    cudaStream_t stream)
{
    int size = tensor.size;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    relu_forward_kernel_naive<<<blocks, threads, 0, stream>>>(tensor.d_data, size);
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// ReLU BACKWARD KERNEL (Naive)
// ============================================================================
__global__ void relu_backward_kernel_naive(
    const float* __restrict__ grad_output,
    const float* __restrict__ forward_output,
    float* __restrict__ grad_input,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        grad_input[idx] = (forward_output[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

void relu_backward_gpu_naive(
    const GPUTensor& grad_output,
    const GPUTensor& forward_output,
    GPUTensor& grad_input)
{
    int size = grad_output.size;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    relu_backward_kernel_naive<<<blocks, threads>>>(
        grad_output.d_data, forward_output.d_data, grad_input.d_data, size
    );
    
    CUDA_CHECK(cudaGetLastError());
}
