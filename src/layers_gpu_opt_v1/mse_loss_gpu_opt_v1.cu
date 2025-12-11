// MSE Loss - Optimized v1 (shared memory reduction + atomicAdd)
#include "layers_gpu_opt_v1/mse_loss_gpu_opt_v1.cuh"

__global__ void mse_loss_forward_kernel(
    const float* __restrict__ pred, const float* __restrict__ target,
    float* __restrict__ total_loss, int size)
{
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = 0.0f;
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        val = diff * diff;
    }
    shared[tid] = val;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(total_loss, shared[0]);
}

__global__ void mse_loss_backward_kernel(
    const float* __restrict__ pred, const float* __restrict__ target,
    float* __restrict__ grad, int size, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad[idx] = scale * (pred[idx] - target[idx]);
}

float mse_loss_forward_opt_v1(const GPUTensorOpt& prediction, const GPUTensorOpt& target) {
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    
    mse_loss_forward_kernel<<<(prediction.size + 255) / 256, 256>>>(
        prediction.d_data, target.d_data, d_loss, prediction.size);
    
    float loss;
    CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));
    return loss / prediction.size;
}

void mse_loss_backward_opt_v1(const GPUTensorOpt& prediction, const GPUTensorOpt& target, GPUTensorOpt& grad_input) {
    float scale = 2.0f / prediction.size;
    mse_loss_backward_kernel<<<(prediction.size + 255) / 256, 256>>>(
        prediction.d_data, target.d_data, grad_input.d_data, prediction.size, scale);
    CUDA_CHECK(cudaGetLastError());
}
