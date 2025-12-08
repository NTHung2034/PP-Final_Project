#include "layers_gpu_naive/mse_loss_gpu_naive.cuh"

// Simple reduction approach:
// - Each thread computes partial sum of squared errors
// - Uses shared memory for block-level reduction
// - Final result accumulated with atomicAdd

__global__ void mse_loss_kernel_naive(
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ loss_sum,
    int size)
{
    // Shared memory for reduction
    __shared__ float partial_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute partial sum for this thread
    float sum = 0.0f;
    if (idx < size) {
        float diff = predicted[idx] - target[idx];
        sum = diff * diff;
    }
    
    partial_sum[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        __syncthreads();
    }
    
    // First thread writes block result
    if (tid == 0) {
        atomicAdd(loss_sum, partial_sum[0]);
    }
}

float mse_loss_forward_gpu_naive(
    const GPUTensor& predicted,
    const GPUTensor& target,
    cudaStream_t stream)
{
    // Allocate device memory for loss sum
    float* d_loss_sum;
    CUDA_CHECK(cudaMalloc(&d_loss_sum, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_loss_sum, 0, sizeof(float), stream));
    
    // Launch reduction kernel
    int threads = 256;
    int blocks = (predicted.size + threads - 1) / threads;
    
    mse_loss_kernel_naive<<<blocks, threads, 0, stream>>>(
        predicted.d_data, target.d_data, d_loss_sum, predicted.size
    );
    
    // Copy result back to host
    float h_loss_sum;
    CUDA_CHECK(cudaMemcpyAsync(&h_loss_sum, d_loss_sum, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(d_loss_sum));
    
    // Return mean squared error
    return h_loss_sum / predicted.size;
}

__global__ void mse_loss_backward_kernel_naive(
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ grad,
    float scale,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Gradient: 2 * (predicted - target) / N
        grad[idx] = scale * (predicted[idx] - target[idx]);
    }
}

void mse_loss_backward_gpu_naive(
    const GPUTensor& predicted,
    const GPUTensor& target,
    GPUTensor& grad,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (predicted.size + threads - 1) / threads;
    float scale = 2.0f / predicted.size;
    
    mse_loss_backward_kernel_naive<<<blocks, threads, 0, stream>>>(
        predicted.d_data, target.d_data, grad.d_data, scale, predicted.size
    );
    
    CUDA_CHECK(cudaGetLastError());
}
