#include "layers_gpu/conv2d_gpu.cuh"
/**
 * OPTIMIZED CONV2D KERNEL WITH SHARED MEMORY TILING
 * 
 * Key Optimizations Applied:
 * 1. Shared Memory Tiling - Reduces global memory accesses
 * 2. Kernel Fusion - Combines Conv + Bias + ReLU in one kernel
 * 3. Memory Coalescing - Threads in warp access consecutive memory
 * 4. Efficient Thread Organization - 16x16 thread blocks
 */

// Tile size for shared memory (must match block dimensions)
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

/**
 * Fused Conv2D + Bias + ReLU Kernel
 * 
 * Each thread block processes a TILE_WIDTH x TILE_HEIGHT output tile
 * Uses shared memory to cache input tiles and reduce global memory reads
 * 
 * Thread Organization:
 * - Grid: (batch_size, output_channels, output_h/TILE_H, output_w/TILE_W)
 * - Block: (TILE_WIDTH, TILE_HEIGHT, 1)
 * 
 * Memory Access Pattern:
 * - Coalesced reads: threads in same warp read consecutive elements
 * - Shared memory: reduces global memory bandwidth by ~9x for 3x3 kernel
 */
__global__ void conv2d_relu_fused_kernel(
    const float* __restrict__ input,    // Input: [N, C_in, H_in, W_in]
    const float* __restrict__ weights,  // Weights: [C_out, C_in, kH, kW]
    const float* __restrict__ bias,     // Bias: [C_out]
    float* __restrict__ output,         // Output: [N, C_out, H_out, W_out]
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int kernel_h, int kernel_w, int padding, int stride,
    bool apply_relu)
{
    // Shared memory for input tile
    // Size: (TILE_H + kH - 1) x (TILE_W + kW - 1) to include padding
    extern __shared__ float tile[];
    
    // Thread indices within block
    int tx = threadIdx.x;  // 0 to TILE_WIDTH-1
    int ty = threadIdx.y;  // 0 to TILE_HEIGHT-1
    
    // Output position this thread is responsible for
    int out_x = blockIdx.x * TILE_WIDTH + tx;
    int out_y = blockIdx.y * TILE_HEIGHT + ty;
    int out_channel = blockIdx.z;
    int batch_idx = blockIdx.w;
    
    // Check if this thread is within valid output bounds
    if (out_x >= out_w || out_y >= out_h || batch_idx >= batch || out_channel >= out_c)
        return;
    
    // Initialize accumulator with bias value
    float sum = bias[out_channel];
    
    // Tile dimensions including halo region for kernel
    int tile_w = TILE_WIDTH + kernel_w - 1;
    int tile_h = TILE_HEIGHT + kernel_h - 1;
    
    // Loop over input channels
    for (int ic = 0; ic < in_c; ic++) {
        // Cooperatively load input tile into shared memory
        // Each thread may load multiple elements if tile is larger than block
        
        // Calculate how many elements each thread should load
        int elements_per_thread = (tile_h * tile_w + (TILE_WIDTH * TILE_HEIGHT - 1)) 
                                  / (TILE_WIDTH * TILE_HEIGHT);
        
        int thread_id = ty * TILE_WIDTH + tx;
        
        for (int i = 0; i < elements_per_thread; i++) {
            int tile_idx = thread_id + i * (TILE_WIDTH * TILE_HEIGHT);
            
            if (tile_idx < tile_h * tile_w) {
                int tile_y = tile_idx / tile_w;
                int tile_x = tile_idx % tile_w;
                
                // Map tile coordinates to input coordinates
                int in_y = blockIdx.y * TILE_HEIGHT * stride + tile_y - padding;
                int in_x = blockIdx.x * TILE_WIDTH * stride + tile_x - padding;
                
                // Load from global memory with boundary checking
                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                    int in_idx = ((batch_idx * in_c + ic) * in_h + in_y) * in_w + in_x;
                    tile[tile_idx] = input[in_idx];
                } else {
                    tile[tile_idx] = 0.0f;  // Padding
                }
            }
        }
        
        // Synchronize to ensure tile is fully loaded
        __syncthreads();
        
        // Perform convolution over this input channel
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                // Position in shared memory tile
                int tile_y = ty * stride + kh;
                int tile_x = tx * stride + kw;
                
                // Weight index
                int w_idx = ((out_channel * in_c + ic) * kernel_h + kh) * kernel_w + kw;
                
                // Multiply and accumulate
                sum += tile[tile_y * tile_w + tile_x] * weights[w_idx];
            }
        }
        
        // Synchronize before loading next input channel
        __syncthreads();
    }
    
    // Apply ReLU activation if requested (kernel fusion)
    if (apply_relu && sum < 0.0f) {
        sum = 0.0f;
    }
    
    // Write result to global memory (coalesced write)
    int out_idx = ((batch_idx * out_c + out_channel) * out_h + out_y) * out_w + out_x;
    output[out_idx] = sum;
}

/**
 * Conv2D Forward Pass Launcher
 * 
 * Configures optimal grid and block dimensions
 * Launches fused Conv2D + Bias + ReLU kernel
 */
void conv2d_forward_gpu(
    const GPUTensor& input,
    const GPUConvWeights& weights,
    GPUTensor& output,
    int kernel_h, int kernel_w,
    int padding, int stride,
    bool apply_relu,
    cudaStream_t stream = 0)
{
    // Configure launch parameters
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT, 1);
    
    // Grid dimensions to cover entire output
    int grid_x = (output.width + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (output.height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int grid_z = output.channels;
    int grid_w = output.batch;
    
    // Note: Using 4D grid (batch, channels, h_tiles, w_tiles)
    // This may exceed device limits, so we handle batch in a loop if needed
    
    // Check device capability
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (grid_w <= prop.maxGridSize[2]) {
        // Can fit everything in one launch
        dim3 gridDim(grid_x, grid_y, grid_z * grid_w);
        
        // Calculate shared memory size
        int tile_w = TILE_WIDTH + kernel_w - 1;
        int tile_h = TILE_HEIGHT + kernel_h - 1;
        size_t smem_size = tile_w * tile_h * sizeof(float);
        
        // Launch kernel
        conv2d_relu_fused_kernel<<<gridDim, blockDim, smem_size, stream>>>(
            input.d_data, weights.d_weights, weights.d_bias, output.d_data,
            input.batch, input.channels, input.height, input.width,
            output.channels, output.height, output.width,
            kernel_h, kernel_w, padding, stride, apply_relu
        );
    } else {
        // Launch multiple kernels for different batches
        for (int b = 0; b < grid_w; b++) {
            dim3 gridDim(grid_x, grid_y, grid_z);
            
            int tile_w = TILE_WIDTH + kernel_w - 1;
            int tile_h = TILE_HEIGHT + kernel_h - 1;
            size_t smem_size = tile_w * tile_h * sizeof(float);
            
            // Offset pointers for this batch
            const float* input_ptr = input.d_data + b * input.channels * input.height * input.width;
            float* output_ptr = output.d_data + b * output.channels * output.height * output.width;
            
            conv2d_relu_fused_kernel<<<gridDim, blockDim, smem_size, stream>>>(
                input_ptr, weights.d_weights, weights.d_bias, output_ptr,
                1, input.channels, input.height, input.width,
                output.channels, output.height, output.width,
                kernel_h, kernel_w, padding, stride, apply_relu
            );
        }
    }
    
    CUDA_CHECK(cudaGetLastError());
}

/**
 * BACKWARD PASS - Compute Gradients
 * 
 * Computes:
 * 1. Gradient w.r.t. input (for backprop)
 * 2. Gradient w.r.t. weights (for weight update)
 * 3. Gradient w.r.t. bias (for bias update)
 */

// Backward pass for input gradient (simplified version)
__global__ void conv2d_backward_input_kernel(
    const float* __restrict__ grad_output,  // [N, C_out, H_out, W_out]
    const float* __restrict__ weights,      // [C_out, C_in, kH, kW]
    float* __restrict__ grad_input,         // [N, C_in, H_in, W_in]
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int kernel_h, int kernel_w, int padding, int stride)
{
    int in_x = blockIdx.x * blockDim.x + threadIdx.x;
    int in_y = blockIdx.y * blockDim.y + threadIdx.y;
    int in_channel = blockIdx.z % in_c;
    int batch_idx = blockIdx.z / in_c;
    
    if (in_x >= in_w || in_y >= in_h || batch_idx >= batch)
        return;
    
    float grad_sum = 0.0f;
    
    // For each output position that this input affects
    for (int oc = 0; oc < out_c; oc++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                // Calculate which output position uses this input
                int out_y = (in_y + padding - kh) / stride;
                int out_x = (in_x + padding - kw) / stride;
                
                // Check if this is a valid output position
                if (out_y >= 0 && out_y < out_h && out_x >= 0 && out_x < out_w) {
                    // Check if convolution kernel exactly hits this input
                    if ((in_y + padding - kh) % stride == 0 && 
                        (in_x + padding - kw) % stride == 0) {
                        
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

// Backward pass for weight gradient
__global__ void conv2d_backward_weights_kernel(
    const float* __restrict__ input,        // [N, C_in, H_in, W_in]
    const float* __restrict__ grad_output,  // [N, C_out, H_out, W_out]
    float* __restrict__ grad_weights,       // [C_out, C_in, kH, kW]
    float* __restrict__ grad_bias,          // [C_out]
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int kernel_h, int kernel_w, int padding, int stride)
{
    int kw = threadIdx.x;
    int kh = threadIdx.y;
    int in_channel = blockIdx.x;
    int out_channel = blockIdx.y;
    
    if (kw >= kernel_w || kh >= kernel_h || in_channel >= in_c || out_channel >= out_c)
        return;
    
    float grad_w_sum = 0.0f;
    float grad_b_sum = 0.0f;
    
    // Accumulate gradient over all batch samples and spatial positions
    for (int b = 0; b < batch; b++) {
        for (int out_y = 0; out_y < out_h; out_y++) {
            for (int out_x = 0; out_x < out_w; out_x++) {
                // Input position
                int in_y = out_y * stride + kh - padding;
                int in_x = out_x * stride + kw - padding;
                
                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                    int in_idx = ((b * in_c + in_channel) * in_h + in_y) * in_w + in_x;
                    int out_idx = ((b * out_c + out_channel) * out_h + out_y) * out_w + out_x;
                    
                    grad_w_sum += input[in_idx] * grad_output[out_idx];
                }
                
                // Bias gradient (only need to do once per kernel position)
                if (in_channel == 0 && kh == 0 && kw == 0) {
                    int out_idx = ((b * out_c + out_channel) * out_h + out_y) * out_w + out_x;
                    grad_b_sum += grad_output[out_idx];
                }
            }
        }
    }
    
    // Write gradients using atomicAdd (multiple threads may update same weight)
    int w_idx = ((out_channel * in_c + in_channel) * kernel_h + kh) * kernel_w + kw;
    atomicAdd(&grad_weights[w_idx], grad_w_sum);
    
    if (in_channel == 0 && kh == 0 && kw == 0) {
        atomicAdd(&grad_bias[out_channel], grad_b_sum);
    }
}

/**
 * Conv2D Backward Pass Launcher
 */
void conv2d_backward_gpu(
    const GPUTensor& input,
    const GPUTensor& grad_output,
    GPUConvWeights& weights,
    GPUTensor& grad_input,
    int kernel_h, int kernel_w,
    int padding, int stride,
    cudaStream_t stream = 0)
{
    // Zero out gradient buffers
    CUDA_CHECK(cudaMemsetAsync(weights.d_grad_w, 0, 
                               weights.weight_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(weights.d_grad_b, 0, 
                               weights.bias_size * sizeof(float), stream));
    
    // Compute gradient w.r.t. input
    {
        dim3 blockDim(16, 16);
        int grid_x = (input.width + 15) / 16;
        int grid_y = (input.height + 15) / 16;
        int grid_z = input.batch * input.channels;
        dim3 gridDim(grid_x, grid_y, grid_z);
        
        conv2d_backward_input_kernel<<<gridDim, blockDim, 0, stream>>>(
            grad_output.d_data, weights.d_weights, grad_input.d_data,
            input.batch, input.channels, input.height, input.width,
            grad_output.channels, grad_output.height, grad_output.width,
            kernel_h, kernel_w, padding, stride
        );
    }
    
    // Compute gradient w.r.t. weights
    {
        dim3 blockDim(kernel_w, kernel_h);
        dim3 gridDim(input.channels, grad_output.channels);
        
        conv2d_backward_weights_kernel<<<gridDim, blockDim, 0, stream>>>(
            input.d_data, grad_output.d_data,
            weights.d_grad_w, weights.d_grad_b,
            input.batch, input.channels, input.height, input.width,
            grad_output.channels, grad_output.height, grad_output.width,
            kernel_h, kernel_w, padding, stride
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}