#include "layers_gpu/upsample_gpu.cuh"

/**
 * =============================================================================
 * UPSAMPLE2D LAYER (Nearest Neighbor)
 * =============================================================================
 * 
 * Each output pixel maps to its nearest input pixel
 * Upsamples by 2x in both dimensions
 */

/**
 * Upsample2D Forward Kernel (Nearest Neighbor)
 * Each thread computes one output element
 */
__global__ void upsample2d_forward_kernel(
    const float* __restrict__ input,     // [N, C, H_in, W_in]
    float* __restrict__ output,          // [N, C, H_out, W_out]
    int batch, int channels, int in_h, int in_w, int out_h, int out_w,
    int scale_factor)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;
    
    if (out_x >= out_w || out_y >= out_h || b >= batch)
        return;
    
    // Map output position to input position (nearest neighbor)
    int in_y = out_y / scale_factor;
    int in_x = out_x / scale_factor;
    
    // Clamp to valid input range (shouldn't be necessary with proper dimensions)
    in_y = min(in_y, in_h - 1);
    in_x = min(in_x, in_w - 1);
    
    // Read from input and write to output (coalesced access)
    int in_idx = ((b * channels + c) * in_h + in_y) * in_w + in_x;
    int out_idx = ((b * channels + c) * out_h + out_y) * out_w + out_x;
    
    output[out_idx] = input[in_idx];
}

/**
 * Upsample2D Backward Kernel
 * Gradient is accumulated from all output pixels that map to same input pixel
 */
__global__ void upsample2d_backward_kernel(
    const float* __restrict__ grad_output,   // [N, C, H_out, W_out]
    float* __restrict__ grad_input,          // [N, C, H_in, W_in]
    int batch, int channels, int in_h, int in_w, int out_h, int out_w,
    int scale_factor)
{
    int in_x = blockIdx.x * blockDim.x + threadIdx.x;
    int in_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;
    
    if (in_x >= in_w || in_y >= in_h || b >= batch)
        return;
    
    float grad_sum = 0.0f;
    
    // Sum gradients from all output pixels that map to this input pixel
    for (int dy = 0; dy < scale_factor; dy++) {
        for (int dx = 0; dx < scale_factor; dx++) {
            int out_y = in_y * scale_factor + dy;
            int out_x = in_x * scale_factor + dx;
            
            if (out_y < out_h && out_x < out_w) {
                int out_idx = ((b * channels + c) * out_h + out_y) * out_w + out_x;
                grad_sum += grad_output[out_idx];
            }
        }
    }
    
    int in_idx = ((b * channels + c) * in_h + in_y) * in_w + in_x;
    grad_input[in_idx] = grad_sum;
}

/**
 * Upsample2D Forward Pass Launcher
 */
void upsample2d_forward_gpu(
    const GPUTensor& input,
    GPUTensor& output,
    int scale_factor,
    cudaStream_t stream)
{
    dim3 blockDim(16, 16);
    int grid_x = (output.width + 15) / 16;
    int grid_y = (output.height + 15) / 16;
    int grid_z = output.batch * output.channels;
    dim3 gridDim(grid_x, grid_y, grid_z);
    
    upsample2d_forward_kernel<<<gridDim, blockDim, 0, stream>>>(
        input.d_data, output.d_data,
        input.batch, input.channels, input.height, input.width,
        output.height, output.width, scale_factor
    );
    
    CUDA_CHECK(cudaGetLastError());
}

/**
 * Upsample2D Backward Pass Launcher
 */
void upsample2d_backward_gpu(
    const GPUTensor& grad_output,
    GPUTensor& grad_input,
    int scale_factor,
    cudaStream_t stream)
{
    dim3 blockDim(16, 16);
    int grid_x = (grad_input.width + 15) / 16;
    int grid_y = (grad_input.height + 15) / 16;
    int grid_z = grad_input.batch * grad_input.channels;
    dim3 gridDim(grid_x, grid_y, grid_z);
    
    upsample2d_backward_kernel<<<gridDim, blockDim, 0, stream>>>(
        grad_output.d_data, grad_input.d_data,
        grad_input.batch, grad_input.channels, grad_input.height, grad_input.width,
        grad_output.height, grad_output.width, scale_factor
    );
    
    CUDA_CHECK(cudaGetLastError());
}
