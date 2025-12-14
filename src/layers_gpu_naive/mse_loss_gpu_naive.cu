#include "layers_gpu_naive/mse_loss_gpu_naive.cuh"

__global__ void mse_loss_forward_kernel_naive(
    const float* __restrict__ prediction,
    const float* __restrict__ target,
    float* __restrict__ partial_loss,
    int size)
{
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    if (idx < size) {
        float diff = prediction[idx] - target[idx];
        local_sum = diff * diff;
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_loss[blockIdx.x] = shared_sum[0];
    }
}

float mse_loss_forward_gpu_naive(
    const GPUTensor& prediction,
    const GPUTensor& target)
{
    int size = prediction.size;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    float* d_partial_loss;
    CUDA_CHECK(cudaMalloc(&d_partial_loss, blocks * sizeof(float)));
    
    mse_loss_forward_kernel_naive<<<blocks, threads>>>(prediction.d_data, target.d_data, d_partial_loss, size);
    
    float* h_partial_loss = new float[blocks];
    CUDA_CHECK(cudaMemcpy(h_partial_loss, d_partial_loss, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float total_loss = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total_loss += h_partial_loss[i];
    }
    
    delete[] h_partial_loss;
    CUDA_CHECK(cudaFree(d_partial_loss));
    
    return total_loss / size;
}

__global__ void mse_loss_backward_kernel_naive(
    const float* __restrict__ prediction,
    const float* __restrict__ target,
    float* __restrict__ grad_input,
    int size,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        grad_input[idx] = scale * (prediction[idx] - target[idx]);
    }
}

void mse_loss_backward_gpu_naive(
    const GPUTensor& prediction,
    const GPUTensor& target,
    GPUTensor& grad_input)
{
    int size = prediction.size;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    float scale = 2.0f / size;
    
    mse_loss_backward_kernel_naive<<<blocks, threads>>>(
        prediction.d_data, target.d_data, grad_input.d_data, size, scale
    );
    
    CUDA_CHECK(cudaGetLastError());
}
