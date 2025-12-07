#include "layers_gpu_naive/conv2d_gpu_naive.cuh"

// Simple approach:
// - Each thread computes one output pixel
// - Global memory access only (no shared memory)
// - Thread loops over input channels and kernel positions
__global__ void conv2d_forward_kernel_naive(
    const float* __restrict__ input,    // [N, C_in, H_in, W_in]
    const float* __restrict__ weights,  // [C_out, C_in, kH, kW]
    const float* __restrict__ bias,     // [C_out]
    float* __restrict__ output,         // [N, C_out, H_out, W_out]
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int kernel_h, int kernel_w, int padding, int stride,
    bool apply_relu)
{
    // Calculate output position this thread is responsible for
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_channel = blockIdx.z % out_c;
    int batch_idx = blockIdx.z / out_c;
    
    // Boundary check
    if (out_x >= out_w || out_y >= out_h || batch_idx >= batch) {
        return;
    }
    
    // Initialize accumulator with bias
    float sum = bias[out_channel];
    
    // Loop over input channels
    for (int ic = 0; ic < in_c; ic++) {
        // Loop over kernel height
        for (int kh = 0; kh < kernel_h; kh++) {
            // Loop over kernel width
            for (int kw = 0; kw < kernel_w; kw++) {
                // Calculate input position
                int in_y = out_y * stride + kh - padding;
                int in_x = out_x * stride + kw - padding;
                
                // Check if within input bounds (zero padding)
                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                    // Get input value
                    int in_idx = ((batch_idx * in_c + ic) * in_h + in_y) * in_w + in_x;
                    float in_val = input[in_idx];
                    
                    // Get weight value
                    int w_idx = ((out_channel * in_c + ic) * kernel_h + kh) * kernel_w + kw;
                    float w_val = weights[w_idx];
                    
                    // Accumulate
                    sum += in_val * w_val;
                }
            }
        }
    }
    
    // Apply ReLU if requested
    if (apply_relu && sum < 0.0f) {
        sum = 0.0f;
    }
    
    // Write output
    int out_idx = ((batch_idx * out_c + out_channel) * out_h + out_y) * out_w + out_x;
    output[out_idx] = sum;
}
 

void conv2d_forward_gpu_naive(
    const GPUTensor& input,
    const GPUConvWeights& weights,
    GPUTensor& output,
    int kernel_h, int kernel_w,
    int padding, int stride,
    bool apply_relu,
    cudaStream_t stream)
{
    // Simple thread block configuration
    dim3 blockDim(16, 16, 1);
    
    // Grid dimensions to cover all output positions
    int grid_x = (output.width + blockDim.x - 1) / blockDim.x;
    int grid_y = (output.height + blockDim.y - 1) / blockDim.y;
    int grid_z = output.batch * output.channels;  // Encode batch and channel in Z
    
    dim3 gridDim(grid_x, grid_y, grid_z);
    
    // Launch kernel
    conv2d_forward_kernel_naive<<<gridDim, blockDim, 0, stream>>>(
        input.d_data, weights.d_weights, weights.d_bias, output.d_data,
        input.batch, input.channels, input.height, input.width,
        output.channels, output.height, output.width,
        kernel_h, kernel_w, padding, stride, apply_relu
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Naive Backward Pass - Gradient w.r.t. Input
// Each thread computes gradient for one input element
__global__ void conv2d_backward_input_kernel_naive(
    const float* __restrict__ grad_output,  // [N, C_out, H_out, W_out]
    const float* __restrict__ weights,      // [C_out, C_in, kH, kW]
    float* __restrict__ grad_input,         // [N, C_in, H_in, W_in]
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int kernel_h, int kernel_w, int padding, int stride)
{
    // Calculate input position
    int in_x = blockIdx.x * blockDim.x + threadIdx.x;
    int in_y = blockIdx.y * blockDim.y + threadIdx.y;
    int in_channel = blockIdx.z % in_c;
    int batch_idx = blockIdx.z / in_c;
    
    if (in_x >= in_w || in_y >= in_h || batch_idx >= batch) {
        return;
    }
    
    float grad_sum = 0.0f;
    
    // Find which output positions this input affects
    for (int oc = 0; oc < out_c; oc++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                // Calculate output position
                int out_y_raw = in_y + padding - kh;
                int out_x_raw = in_x + padding - kw;
                
                // Check if this maps to valid output
                if (out_y_raw >= 0 && out_x_raw >= 0 && 
                    out_y_raw % stride == 0 && out_x_raw % stride == 0) {
                    
                    int out_y = out_y_raw / stride;
                    int out_x = out_x_raw / stride;
                    
                    if (out_y < out_h && out_x < out_w) {
                        int grad_out_idx = ((batch_idx * out_c + oc) * out_h + out_y) * out_w + out_x;
                        int w_idx = ((oc * in_c + in_channel) * kernel_h + kh) * kernel_w + kw;
                        
                        grad_sum += grad_output[grad_out_idx] * weights[w_idx];
                    }
                }
            }
        }
    }
    
    int in_idx = ((batch_idx * in_c + in_channel) * in_h + in_y) * in_w + in_x;
    grad_input[in_idx] = grad_sum;
}

// Naive Backward Pass - Gradient w.r.t. Weights
// Uses atomic operations for gradient accumulation

__global__ void conv2d_backward_weights_kernel_naive(
    const float* __restrict__ input,        // [N, C_in, H_in, W_in]
    const float* __restrict__ grad_output,  // [N, C_out, H_out, W_out]
    float* __restrict__ grad_weights,       // [C_out, C_in, kH, kW]
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int kernel_h, int kernel_w, int padding, int stride)
{
    // Each thread handles one weight element across all positions
    int kw = threadIdx.x;
    int kh = threadIdx.y;
    int in_channel = blockIdx.x;
    int out_channel = blockIdx.y;
    
    if (kw >= kernel_w || kh >= kernel_h || in_channel >= in_c || out_channel >= out_c) {
        return;
    }
    
    float grad_sum = 0.0f;
    
    // Accumulate over all batches and spatial positions
    for (int b = 0; b < batch; b++) {
        for (int out_y = 0; out_y < out_h; out_y++) {
            for (int out_x = 0; out_x < out_w; out_x++) {
                int in_y = out_y * stride + kh - padding;
                int in_x = out_x * stride + kw - padding;
                
                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                    int in_idx = ((b * in_c + in_channel) * in_h + in_y) * in_w + in_x;
                    int out_idx = ((b * out_c + out_channel) * out_h + out_y) * out_w + out_x;
                    
                    grad_sum += input[in_idx] * grad_output[out_idx];
                }
            }
        }
    }
    
    // Write gradient using atomic add (in case of thread conflicts)
    int w_idx = ((out_channel * in_c + in_channel) * kernel_h + kh) * kernel_w + kw;
    atomicAdd(&grad_weights[w_idx], grad_sum);
}

__global__ void conv2d_backward_bias_kernel_naive(
    const float* __restrict__ grad_output,  // [N, C_out, H_out, W_out]
    float* __restrict__ grad_bias,          // [C_out]
    int batch, int out_c, int out_h, int out_w)
{
    int out_channel = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_channel >= out_c) {
        return;
    }
    
    float grad_sum = 0.0f;
    
    // Sum over all batch samples and spatial positions
    for (int b = 0; b < batch; b++) {
        for (int out_y = 0; out_y < out_h; out_y++) {
            for (int out_x = 0; out_x < out_w; out_x++) {
                int out_idx = ((b * out_c + out_channel) * out_h + out_y) * out_w + out_x;
                grad_sum += grad_output[out_idx];
            }
        }
    }
    
    atomicAdd(&grad_bias[out_channel], grad_sum);
}


void conv2d_backward_gpu_naive(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUConvWeights& weights,
    GPUTensor& grad_input,
    int kernel_h, int kernel_w,
    int padding, int stride,
    cudaStream_t stream)
{
    // Zero out gradient buffers first
    CUDA_CHECK(cudaMemsetAsync(weights.d_grad_w, 0, 
                               weights.weight_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(weights.d_grad_b, 0, 
                               weights.bias_size * sizeof(float), stream));
    
    // 1. Gradient w.r.t. input
    {
        dim3 blockDim(16, 16);
        int grid_x = (input.width + 15) / 16;
        int grid_y = (input.height + 15) / 16;
        int grid_z = input.batch * input.channels;
        dim3 gridDim(grid_x, grid_y, grid_z);
        
        conv2d_backward_input_kernel_naive<<<gridDim, blockDim, 0, stream>>>(
            grad_output.d_data, weights.d_weights, grad_input.d_data,
            input.batch, input.channels, input.height, input.width,
            grad_output.channels, grad_output.height, grad_output.width,
            kernel_h, kernel_w, padding, stride
        );
    }
    
    // 2. Gradient w.r.t. weights
    {
        dim3 blockDim(kernel_w, kernel_h);
        dim3 gridDim(input.channels, grad_output.channels);
        
        conv2d_backward_weights_kernel_naive<<<gridDim, blockDim, 0, stream>>>(
            input.d_data, grad_output.d_data, weights.d_grad_w,
            input.batch, input.channels, input.height, input.width,
            grad_output.channels, grad_output.height, grad_output.width,
            kernel_h, kernel_w, padding, stride
        );
    }
    
    // 3. Gradient w.r.t. bias
    {
        int threads = 256;
        int blocks = (grad_output.channels + threads - 1) / threads;
        
        conv2d_backward_bias_kernel_naive<<<blocks, threads, 0, stream>>>(
            grad_output.d_data, weights.d_grad_b,
            input.batch, grad_output.channels, grad_output.height, grad_output.width
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}
