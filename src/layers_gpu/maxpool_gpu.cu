#include "layers_gpu/maxpool_gpu.cuh"
/**
 * =============================================================================
 * MAXPOOL2D LAYER
 * =============================================================================
 * 
 * Each thread computes one output element
 * Finds maximum value in a 2x2 window
 * Stores indices for backward pass
 */

/**
 * MaxPool2D Forward Kernel
 * Pool size: 2x2, stride: 2 (downsamples by 2x)
 */
__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ input,     // [N, C, H_in, W_in]
    float* __restrict__ output,          // [N, C, H_out, W_out]
    int* __restrict__ max_indices,       // [N, C, H_out, W_out] - store indices for backward
    int batch, int channels, int in_h, int in_w, int out_h, int out_w)
{
    // Each thread handles one output element
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;
    
    if (out_x >= out_w || out_y >= out_h || b >= batch)
        return;
    
    // Input position (top-left of 2x2 window)
    int in_y = out_y * 2;
    int in_x = out_x * 2;
    
    // Find maximum in 2x2 window
    float max_val = -1e38f;
    int max_idx = 0;
    
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int y = in_y + dy;
            int x = in_x + dx;
            
            if (y < in_h && x < in_w) {
                int idx = ((b * channels + c) * in_h + y) * in_w + x;
                float val = input[idx];
                
                if (val > max_val) {
                    max_val = val;
                    max_idx = dy * 2 + dx;  // Relative index in 2x2 window
                }
            }
        }
    }
    
    // Write output (coalesced memory access)
    int out_idx = ((b * channels + c) * out_h + out_y) * out_w + out_x;
    output[out_idx] = max_val;
    max_indices[out_idx] = max_idx;
}

/**
 * MaxPool2D Backward Kernel
 * Gradient flows only through the maximum element
 */
__global__ void maxpool2d_backward_kernel(
    const float* __restrict__ grad_output,   // [N, C, H_out, W_out]
    const int* __restrict__ max_indices,     // [N, C, H_out, W_out]
    float* __restrict__ grad_input,          // [N, C, H_in, W_in]
    int batch, int channels, int in_h, int in_w, int out_h, int out_w)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;
    
    if (out_x >= out_w || out_y >= out_h || b >= batch)
        return;
    
    int out_idx = ((b * channels + c) * out_h + out_y) * out_w + out_x;
    float grad = grad_output[out_idx];
    int max_idx = max_indices[out_idx];
    
    // Unpack max_idx to dy, dx
    int dy = max_idx / 2;
    int dx = max_idx % 2;
    
    // Input position
    int in_y = out_y * 2 + dy;
    int in_x = out_x * 2 + dx;
    
    if (in_y < in_h && in_x < in_w) {
        int in_idx = ((b * channels + c) * in_h + in_y) * in_w + in_x;
        
        // Use atomicAdd in case multiple pooling windows overlap (shouldn't happen with stride=2)
        atomicAdd(&grad_input[in_idx], grad);
    }
}

/**
 * MaxPool2D Forward Pass Launcher
 */
void maxpool2d_forward_gpu(
    const GPUTensor& input,
    GPUTensor& output,
    int* d_max_indices,  // Device pointer for max indices
    cudaStream_t stream = 0)
{
    dim3 blockDim(16, 16);
    int grid_x = (output.width + 15) / 16;
    int grid_y = (output.height + 15) / 16;
    int grid_z = output.batch * output.channels;
    dim3 gridDim(grid_x, grid_y, grid_z);
    
    maxpool2d_forward_kernel<<<gridDim, blockDim, 0, stream>>>(
        input.d_data, output.d_data, d_max_indices,
        input.batch, input.channels, input.height, input.width,
        output.height, output.width
    );
    
    CUDA_CHECK(cudaGetLastError());
}

/**
 * MaxPool2D Backward Pass Launcher
 */
void maxpool2d_backward_gpu(
    const GPUTensor& grad_output,
    const int* d_max_indices,
    GPUTensor& grad_input,
    cudaStream_t stream = 0)
{
    // Zero out grad_input first
    CUDA_CHECK(cudaMemsetAsync(grad_input.d_data, 0, grad_input.bytes, stream));
    
    dim3 blockDim(16, 16);
    int grid_x = (grad_output.width + 15) / 16;
    int grid_y = (grad_output.height + 15) / 16;
    int grid_z = grad_output.batch * grad_output.channels;
    dim3 gridDim(grid_x, grid_y, grid_z);
    
    maxpool2d_backward_kernel<<<gridDim, blockDim, 0, stream>>>(
        grad_output.d_data, d_max_indices, grad_input.d_data,
        grad_input.batch, grad_input.channels, grad_input.height, grad_input.width,
        grad_output.height, grad_output.width
    );
    
    CUDA_CHECK(cudaGetLastError());
}
