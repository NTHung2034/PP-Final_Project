#include "layers_gpu_opt_v2/mse_loss_gpu_opt_v2.cuh"
#include <cub/cub.cuh>  // For efficient reduction (if available)

// =============================================================================
// MSE LOSS FORWARD - Parallel reduction
// =============================================================================
__global__ void mse_loss_forward_kernel_v2(
    const float* __restrict__ output, 
    const float* __restrict__ target, 
    float* __restrict__ loss_buffer,
    int size)
{
    __shared__ float s_partial[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    
    // Grid-stride loop for handling large tensors
    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        float diff = output[i] - target[i];
        local_sum += diff * diff;
    }
    
    s_partial[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial[tid] += s_partial[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(loss_buffer, s_partial[0]);
    }
}

// =============================================================================
// MSE LOSS BACKWARD - Vectorized gradient computation
// =============================================================================
__global__ void mse_loss_backward_kernel_v2(
    const float* __restrict__ output, 
    const float* __restrict__ target, 
    float* __restrict__ grad_output,
    float scale,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_output[idx] = scale * (output[idx] - target[idx]);
    }
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

// Static buffer for reduction result (device memory)
static float* d_loss_buffer = nullptr;
static bool loss_buffer_allocated = false;

float mse_loss_forward_opt_v2(
    const GPUTensorOpt& output, 
    const GPUTensorOpt& target,
    cudaStream_t stream)
{
    // Allocate loss buffer if needed
    if (!loss_buffer_allocated) {
        CUDA_CHECK(cudaMalloc(&d_loss_buffer, sizeof(float)));
        loss_buffer_allocated = true;
    }
    
    // Zero the loss buffer
    CUDA_CHECK(cudaMemsetAsync(d_loss_buffer, 0, sizeof(float), stream));
    
    int size = output.size;
    int blocks = min((size + 255) / 256, 1024);  // Cap at 1024 blocks
    
    mse_loss_forward_kernel_v2<<<blocks, 256, 0, stream>>>(
        output.d_data, target.d_data, d_loss_buffer, size);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back
    float h_loss;
    CUDA_CHECK(cudaMemcpyAsync(&h_loss, d_loss_buffer, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    return h_loss / size;  // Mean
}

void mse_loss_backward_opt_v2(
    const GPUTensorOpt& output, 
    const GPUTensorOpt& target, 
    GPUTensorOpt& grad_output,
    cudaStream_t stream)
{
    float scale = 2.0f / output.size;  // Derivative of MSE
    
    mse_loss_backward_kernel_v2<<<(output.size + 255) / 256, 256, 0, stream>>>(
        output.d_data, target.d_data, grad_output.d_data, scale, output.size);
    CUDA_CHECK(cudaGetLastError());
}
