#include "layers_gpu_opt_v1/conv2d_gpu_opt_v1.cuh"

// Constant memory for biases (max 256 channels)
__constant__ float c_bias[256];

__global__ void conv2d_forward_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    bool apply_relu)
{
    // Shared memory for input tile (with halo for 3x3 kernel)
    __shared__ float s_input[TILE_H + 2][TILE_W + 2];
    
    int n = blockIdx.z;                              // Batch index
    int c_out = blockIdx.y;                          // Output channel
    int tile_row = blockIdx.x / ((W_out + TILE_W - 1) / TILE_W);
    int tile_col = blockIdx.x % ((W_out + TILE_W - 1) / TILE_W);
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int h_out_base = tile_row * TILE_H;
    int w_out_base = tile_col * TILE_W;
    int h_out = h_out_base + ty;
    int w_out = w_out_base + tx;
    
    float sum = 0.0f;
    
    // Iterate over input channels
    for (int c_in = 0; c_in < C_in; c_in++) {
        // Cooperative loading of input tile into shared memory
        // Each thread loads one element + halo elements
        int h_in_base = h_out_base - PAD;
        int w_in_base = w_out_base - PAD;
        
        // Load main tile element
        int h_load = h_in_base + ty;
        int w_load = w_in_base + tx;
        
        if (ty < TILE_H + 2 && tx < TILE_W + 2) {
            if (h_load >= 0 && h_load < H_in && w_load >= 0 && w_load < W_in) {
                s_input[ty][tx] = input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_load * W_in + w_load];
            } else {
                s_input[ty][tx] = 0.0f;  // Zero padding
            }
        }
        
        // Load extra halo elements (threads at edge of block load extra)
        if (tx < 2 && (w_out_base + TILE_W + tx) < W_out + PAD) {
            int h_extra = h_in_base + ty;
            int w_extra = w_in_base + TILE_W + tx;
            if (ty < TILE_H + 2) {
                if (h_extra >= 0 && h_extra < H_in && w_extra >= 0 && w_extra < W_in) {
                    s_input[ty][TILE_W + tx] = input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_extra * W_in + w_extra];
                } else {
                    s_input[ty][TILE_W + tx] = 0.0f;
                }
            }
        }
        if (ty < 2 && (h_out_base + TILE_H + ty) < H_out + PAD) {
            int h_extra = h_in_base + TILE_H + ty;
            int w_extra = w_in_base + tx;
            if (tx < TILE_W + 2) {
                if (h_extra >= 0 && h_extra < H_in && w_extra >= 0 && w_extra < W_in) {
                    s_input[TILE_H + ty][tx] = input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_extra * W_in + w_extra];
                } else {
                    s_input[TILE_H + ty][tx] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Convolution from shared memory
        if (h_out < H_out && w_out < W_out) {
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    int weight_idx = c_out * (C_in * KERNEL_SIZE * KERNEL_SIZE) + c_in * (KERNEL_SIZE * KERNEL_SIZE) + kh * KERNEL_SIZE + kw;
                    sum += s_input[ty + kh][tx + kw] * weights[weight_idx];
                }
            }
        }
        __syncthreads();
    }
    
    // Add bias from constant memory and write output
    if (h_out < H_out && w_out < W_out && n < N && c_out < C_out) {
        sum += c_bias[c_out];
        if (apply_relu && sum < 0.0f) sum = 0.0f;
        output[n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out] = sum;
    }
}

void conv2d_forward_opt_v1(const GPUTensorOpt& input, const GPUConvWeightsOpt& weights, GPUTensorOpt& output, bool apply_relu) {
    int N = input.batch, C_in = input.channels, H_in = input.height, W_in = input.width;
    int C_out = weights.out_c, H_out = output.height, W_out = output.width;
    
    // Copy bias to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_bias, weights.d_bias, weights.bias_size * sizeof(float)));
    
    // Grid: (num_tiles, C_out, N), Block: (TILE_W, TILE_H)
    int tiles_w = (W_out + TILE_W - 1) / TILE_W;
    int tiles_h = (H_out + TILE_H - 1) / TILE_H;
    dim3 grid(tiles_h * tiles_w, C_out, N);
    dim3 block(TILE_W + 2, TILE_H + 2);  // Extra threads for halo loading
    
    conv2d_forward_tiled_kernel<<<grid, block>>>(
        input.d_data, weights.d_weights, output.d_data,
        N, C_in, H_in, W_in, C_out, H_out, W_out, apply_relu);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// CONVOLUTION BACKWARD - Gradient computation
// =============================================================================
__global__ void conv2d_backward_input_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ weights,
    float* __restrict__ grad_input,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H_in * W_in;
    if (idx >= total) return;
    
    int w_in = idx % W_in; idx /= W_in;
    int h_in = idx % H_in; idx /= H_in;
    int c_in = idx % C_in;
    int n = idx / C_in;
    
    float grad = 0.0f;
    
    for (int c_out = 0; c_out < C_out; c_out++) {
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                int h_out = h_in + PAD - kh;
                int w_out = w_in + PAD - kw;
                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    int go_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
                    int w_idx = c_out * (C_in * KERNEL_SIZE * KERNEL_SIZE) + c_in * (KERNEL_SIZE * KERNEL_SIZE) + kh * KERNEL_SIZE + kw;
                    grad += grad_output[go_idx] * weights[w_idx];
                }
            }
        }
    }
    grad_input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in] = grad;
}

__global__ void conv2d_backward_weights_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * KERNEL_SIZE * KERNEL_SIZE;
    if (idx >= total) return;
    
    int kw = idx % KERNEL_SIZE; idx /= KERNEL_SIZE;
    int kh = idx % KERNEL_SIZE; idx /= KERNEL_SIZE;
    int c_in = idx % C_in;
    int c_out = idx / C_in;
    
    float grad = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                int h_in = h_out - PAD + kh;
                int w_in = w_out - PAD + kw;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    grad += grad_output[n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out] *
                            input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in];
                }
            }
        }
    }
    grad_weights[c_out * (C_in * KERNEL_SIZE * KERNEL_SIZE) + c_in * (KERNEL_SIZE * KERNEL_SIZE) + kh * KERNEL_SIZE + kw] = grad;
}

__global__ void conv2d_backward_bias_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int N, int C_out, int H_out, int W_out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C_out) return;
    
    float grad = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                grad += grad_output[n * (C_out * H_out * W_out) + c * (H_out * W_out) + h * W_out + w];
            }
        }
    }
    grad_bias[c] = grad;
}

void conv2d_backward_opt_v1(const GPUTensorOpt& input, const GPUTensorOpt& grad_output,
                            GPUConvWeightsOpt& weights, GPUTensorOpt& grad_input) {
    int N = input.batch, C_in = input.channels, H_in = input.height, W_in = input.width;
    int C_out = weights.out_c, H_out = grad_output.height, W_out = grad_output.width;
    
    // Gradient w.r.t input
    int total_in = N * C_in * H_in * W_in;
    conv2d_backward_input_kernel<<<(total_in + 255) / 256, 256>>>(
        grad_output.d_data, weights.d_weights, grad_input.d_data,
        N, C_in, H_in, W_in, C_out, H_out, W_out);
    
    // Gradient w.r.t weights
    int total_w = C_out * C_in * KERNEL_SIZE * KERNEL_SIZE;
    conv2d_backward_weights_kernel<<<(total_w + 255) / 256, 256>>>(
        input.d_data, grad_output.d_data, weights.d_grad_w,
        N, C_in, H_in, W_in, C_out, H_out, W_out);
    
    // Gradient w.r.t bias
    conv2d_backward_bias_kernel<<<(C_out + 255) / 256, 256>>>(
        grad_output.d_data, weights.d_grad_b, N, C_out, H_out, W_out);
    
    CUDA_CHECK(cudaGetLastError());
}
