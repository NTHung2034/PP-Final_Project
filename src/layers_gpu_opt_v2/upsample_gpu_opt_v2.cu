#include "layers_gpu_opt_v2/upsample_gpu_opt_v2.cuh"

// =============================================================================
// UPSAMPLE FORWARD - Vectorized reads/writes
// =============================================================================
__global__ void upsample2d_forward_kernel_v2(
    const float* __restrict__ input, 
    float* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;
    
    int w_out = idx % W_out; 
    int tmp = idx / W_out;
    int h_out = tmp % H_out; 
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;
    
    // Nearest neighbor - just divide by 2
    int h_in = h_out >> 1;  // Fast divide by 2
    int w_in = w_out >> 1;
    
    output[idx] = input[n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in];
}

// =============================================================================
// UPSAMPLE BACKWARD - Sum 2x2 regions, fully unrolled
// =============================================================================
__global__ void upsample2d_backward_kernel_v2(
    const float* __restrict__ grad_out, 
    float* __restrict__ grad_in,
    int N, int C, int H_in, int W_in, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_in * W_in;
    if (idx >= total) return;
    
    int w_in = idx % W_in; 
    int tmp = idx / W_in;
    int h_in = tmp % H_in; 
    tmp /= H_in;
    int c = tmp % C;
    int n = tmp / C;
    
    // Base index in grad_out
    int go_base = n * (C * H_out * W_out) + c * (H_out * W_out);
    int h_out = h_in << 1;  // Fast multiply by 2
    int w_out = w_in << 1;
    
    // Sum 2x2 region - fully unrolled, no loop overhead
    float grad = grad_out[go_base + h_out * W_out + w_out]
               + grad_out[go_base + h_out * W_out + w_out + 1]
               + grad_out[go_base + (h_out + 1) * W_out + w_out]
               + grad_out[go_base + (h_out + 1) * W_out + w_out + 1];
    
    grad_in[idx] = grad;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================
void upsample2d_forward_opt_v2(
    const GPUTensorOpt& input, 
    GPUTensorOpt& output,
    cudaStream_t stream)
{
    upsample2d_forward_kernel_v2<<<(output.size + 255) / 256, 256, 0, stream>>>(
        input.d_data, output.d_data,
        input.batch, input.channels, input.height, input.width, output.height, output.width);
    CUDA_CHECK(cudaGetLastError());
}

void upsample2d_backward_opt_v2(
    const GPUTensorOpt& grad_output, 
    GPUTensorOpt& grad_input,
    cudaStream_t stream)
{
    upsample2d_backward_kernel_v2<<<(grad_input.size + 255) / 256, 256, 0, stream>>>(
        grad_output.d_data, grad_input.d_data,
        grad_input.batch, grad_input.channels, grad_input.height, grad_input.width,
        grad_output.height, grad_output.width);
    CUDA_CHECK(cudaGetLastError());
}
