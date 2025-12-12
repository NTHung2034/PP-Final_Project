#include "layers_gpu_opt_v2/mse_loss_gpu_opt_v2.cuh"

__global__ void mse_loss_forward_kernel_v2(
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
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(total_loss, shared[0] / size);
}

__global__ void mse_loss_backward_kernel_v2(
    const float* __restrict__ pred, const float* __restrict__ target,
    float* __restrict__ grad, int size, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad[idx] = scale * (pred[idx] - target[idx]);
}

void mse_loss_forward_opt_v2(const GPUTensorOpt& prediction, const GPUTensorOpt& target,
                              float* d_loss, cudaStream_t stream) {
    mse_loss_forward_kernel_v2<<<(prediction.size + 255) / 256, 256, 0, stream>>>(
        prediction.d_data, target.d_data, d_loss, prediction.size);
    CUDA_CHECK(cudaGetLastError());
}

void mse_loss_backward_opt_v2(const GPUTensorOpt& prediction, const GPUTensorOpt& target,
                               GPUTensorOpt& grad_input, cudaStream_t stream) {
    float scale = 2.0f / prediction.size;
    mse_loss_backward_kernel_v2<<<(prediction.size + 255) / 256, 256, 0, stream>>>(
        prediction.d_data, target.d_data, grad_input.d_data, prediction.size, scale);
    CUDA_CHECK(cudaGetLastError());
}
