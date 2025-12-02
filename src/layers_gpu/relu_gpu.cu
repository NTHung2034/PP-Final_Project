#include "layers_gpu/relu_gpu.cuh"
/**
 * =============================================================================
 * RELU ACTIVATION LAYER
 * =============================================================================
 * 
 * Simple element-wise operation: y = max(0, x)
 * Can be fused with Conv2D (already done above)
 * Standalone version provided for decoder layers
 */

/**
 * ReLU Forward Kernel (In-place)
 * Each thread processes one element
 */
__global__ void relu_forward_kernel(
    float* __restrict__ data,  // Input/Output (in-place operation)
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // ReLU: max(0, x)
        // Predicated execution - no divergence overhead in practice
        if (data[idx] < 0.0f) {
            data[idx] = 0.0f;
        }
    }
}

/**
 * ReLU Backward Kernel
 * Gradient = grad_out if input > 0, else 0
 */
__global__ void relu_backward_kernel(
    const float* __restrict__ input,        // Original input
    const float* __restrict__ grad_output,  // Gradient from next layer
    float* __restrict__ grad_input,         // Gradient to previous layer
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Gradient flows through only if input was positive
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

/**
 * ReLU Forward Pass Launcher
 */
void relu_forward_gpu(GPUTensor& data, cudaStream_t stream) {
    int threads = 256;
    int blocks = (data.size + threads - 1) / threads;
    
    relu_forward_kernel<<<blocks, threads, 0, stream>>>(data.d_data, data.size);
    
    CUDA_CHECK(cudaGetLastError());
}

/**
 * ReLU Backward Pass Launcher
 */
void relu_backward_gpu(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUTensor& grad_input,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (input.size + threads - 1) / threads;
    
    relu_backward_kernel<<<blocks, threads, 0, stream>>>(
        input.d_data, grad_output.d_data, grad_input.d_data, input.size
    );
    
    CUDA_CHECK(cudaGetLastError());
}
