#include "layers_gpu_naive/maxpool_gpu_naive.cuh"
#include <float.h>


__global__ void maxpool2d_forward_kernel_naive(
    const float* __restrict__ input,   // [N, C, H_in, W_in]
    float* __restrict__ output,         // [N, C, H_out, W_out]
    int* __restrict__ max_indices,      // Store indices for backward pass
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w)
{
    // Calculate output position
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z % channels;
    int batch_idx = blockIdx.z / channels;
    
    if (out_x >= out_w || out_y >= out_h || batch_idx >= batch) {
        return;
    }
    
    // Input position (top-left of 2x2 window)
    int in_y = out_y * 2;
    int in_x = out_x * 2;
    
    // Find maximum in 2x2 window
    float max_val = -FLT_MAX;
    int max_idx = 0;
    
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int y = in_y + dy;
            int x = in_x + dx;
            
            if (y < in_h && x < in_w) {
                int idx = ((batch_idx * channels + channel) * in_h + y) * in_w + x;
                float val = input[idx];
                
                if (val > max_val) {
                    max_val = val;
                    max_idx = dy * 2 + dx;  // Store relative index (0-3)
                }
            }
        }
    }
    
    // Write output
    int out_idx = ((batch_idx * channels + channel) * out_h + out_y) * out_w + out_x;
    output[out_idx] = max_val;
    max_indices[out_idx] = max_idx;
}


void maxpool2d_forward_gpu_naive(
    const GPUTensor& input,
    GPUTensor& output,
    int* d_max_indices,
    cudaStream_t stream)
{
    dim3 blockDim(16, 16);
    int grid_x = (output.width + 15) / 16;
    int grid_y = (output.height + 15) / 16;
    int grid_z = output.batch * output.channels;
    dim3 gridDim(grid_x, grid_y, grid_z);
    
    maxpool2d_forward_kernel_naive<<<gridDim, blockDim, 0, stream>>>(
        input.d_data, output.d_data, d_max_indices,
        input.batch, input.channels, input.height, input.width,
        output.height, output.width
    );
    
    CUDA_CHECK(cudaGetLastError());
}


__global__ void maxpool2d_backward_kernel_naive(
    const float* __restrict__ grad_output,  // [N, C, H_out, W_out]
    const int* __restrict__ max_indices,    // Indices from forward pass
    float* __restrict__ grad_input,         // [N, C, H_in, W_in]
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w)
{
    // Calculate output position
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z % channels;
    int batch_idx = blockIdx.z / channels;
    
    if (out_x >= out_w || out_y >= out_h || batch_idx >= batch) {
        return;
    }
    
    // Get gradient and max index
    int out_idx = ((batch_idx * channels + channel) * out_h + out_y) * out_w + out_x;
    float grad_val = grad_output[out_idx];
    int max_idx = max_indices[out_idx];
    
    // Convert relative index to dy, dx
    int dy = max_idx / 2;
    int dx = max_idx % 2;
    
    // Calculate input position
    int in_y = out_y * 2 + dy;
    int in_x = out_x * 2 + dx;
    
    // Write gradient to position where max was found
    int in_idx = ((batch_idx * channels + channel) * in_h + in_y) * in_w + in_x;
    grad_input[in_idx] = grad_val;
}


void maxpool2d_backward_gpu_naive(
    const GPUTensor& grad_output,
    const int* d_max_indices,
    GPUTensor& grad_input,
    cudaStream_t stream)
{
    // Zero out gradient input first (only max positions will be non-zero)
    CUDA_CHECK(cudaMemsetAsync(grad_input.d_data, 0, grad_input.bytes, stream));
    
    dim3 blockDim(16, 16);
    int grid_x = (grad_output.width + 15) / 16;
    int grid_y = (grad_output.height + 15) / 16;
    int grid_z = grad_output.batch * grad_output.channels;
    dim3 gridDim(grid_x, grid_y, grid_z);
    
    maxpool2d_backward_kernel_naive<<<gridDim, blockDim, 0, stream>>>(
        grad_output.d_data, d_max_indices, grad_input.d_data,
        grad_input.batch, grad_input.channels, grad_input.height, grad_input.width,
        grad_output.height, grad_output.width
    );
    
    CUDA_CHECK(cudaGetLastError());
}
