#include "layers_gpu/mse_loss_gpu.cuh"

/**
 * =============================================================================
 * MSE LOSS COMPUTATION (Reduction Pattern)
 * =============================================================================
 * 
 * Computes Mean Squared Error loss using hierarchical reduction
 * Uses shared memory for efficient partial sums
 */

/**
 * MSE Loss Kernel with Hierarchical Reduction
 * 
 * Stage 1: Each thread block computes partial sum using shared memory
 * Stage 2: Atomically accumulate block results into global sum
 */
__global__ void mse_loss_kernel(
    const float* __restrict__ predicted,  // [N, C, H, W]
    const float* __restrict__ target,     // [N, C, H, W]
    float* __restrict__ loss_sum,         // Global loss accumulator
    int size)
{
    // Shared memory for block-level reduction
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data and compute squared error
    float diff = 0.0f;
    if (idx < size) {
        diff = predicted[idx] - target[idx];
        diff = diff * diff;  // Square the difference
    }
    sdata[tid] = diff;
    __syncthreads();
    
    // Hierarchical reduction in shared memory
    // Reduction tree: threads cooperatively sum values
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes block result to global memory
    if (tid == 0) {
        atomicAdd(loss_sum, sdata[0]);
    }
}

/**
 * MSE Loss Gradient Kernel
 * Computes gradient: 2 * (predicted - target) / N
 */
__global__ void mse_loss_backward_kernel(
    const float* __restrict__ predicted,
    const float* __restrict__ target,
    float* __restrict__ grad,
    int size,
    float scale)  // scale = 2.0 / N
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        grad[idx] = scale * (predicted[idx] - target[idx]);
    }
}

/**
 * MSE Loss Forward Pass
 * Returns mean squared error value
 */
float mse_loss_forward_gpu(
    const GPUTensor& predicted,
    const GPUTensor& target,
    cudaStream_t stream = 0)
{
    // Allocate device memory for loss sum
    float* d_loss_sum;
    CUDA_CHECK(cudaMalloc(&d_loss_sum, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_loss_sum, 0, sizeof(float), stream));
    
    // Launch reduction kernel
    int threads = 256;
    int blocks = (predicted.size + threads - 1) / threads;
    size_t smem_size = threads * sizeof(float);
    
    mse_loss_kernel<<<blocks, threads, smem_size, stream>>>(
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

/**
 * MSE Loss Backward Pass
 */
void mse_loss_backward_gpu(
    const GPUTensor& predicted,
    const GPUTensor& target,
    GPUTensor& grad,
    cudaStream_t stream = 0)
{
    int threads = 256;
    int blocks = (predicted.size + threads - 1) / threads;
    float scale = 2.0f / predicted.size;
    
    mse_loss_backward_kernel<<<blocks, threads, 0, stream>>>(
        predicted.d_data, target.d_data, grad.d_data,
        predicted.size, scale
    );
    
    CUDA_CHECK(cudaGetLastError());
}
