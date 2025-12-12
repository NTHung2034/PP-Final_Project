#include "layers_gpu_opt_v2/conv2d_gpu_opt_v2.cuh"

// =============================================================================
// FUSED TILED CONVOLUTION FORWARD - Conv + Bias + ReLU in single kernel
// =============================================================================
__global__ void conv2d_forward_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    bool apply_relu)
{
    __shared__ float s_input[TILE_H_V2 + 2][TILE_W_V2 + 2];
    
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int tile_row = blockIdx.x / ((W_out + TILE_W_V2 - 1) / TILE_W_V2);
    int tile_col = blockIdx.x % ((W_out + TILE_W_V2 - 1) / TILE_W_V2);
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int h_out_base = tile_row * TILE_H_V2, w_out_base = tile_col * TILE_W_V2;
    int h_out = h_out_base + ty, w_out = w_out_base + tx;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < C_in; c_in++) {
        int h_in_base = h_out_base - PAD_V2, w_in_base = w_out_base - PAD_V2;
        
        // Cooperative loading with halo
        for (int load_y = ty; load_y < TILE_H_V2 + 2; load_y += blockDim.y) {
            for (int load_x = tx; load_x < TILE_W_V2 + 2; load_x += blockDim.x) {
                int h_load = h_in_base + load_y, w_load = w_in_base + load_x;
                s_input[load_y][load_x] = (h_load >= 0 && h_load < H_in && w_load >= 0 && w_load < W_in)
                    ? input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_load * W_in + w_load]
                    : 0.0f;
            }
        }
        __syncthreads();
        
        // Convolution from shared memory
        if (h_out < H_out && w_out < W_out) {
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE_V2; kh++) {
                #pragma unroll
                for (int kw = 0; kw < KERNEL_SIZE_V2; kw++) {
                    sum += s_input[ty + kh][tx + kw] * 
                           weights[c_out * (C_in * 9) + c_in * 9 + kh * 3 + kw];
                }
            }
        }
        __syncthreads();
    }
    
    // Fused: Add bias + optional ReLU + write output
    if (h_out < H_out && w_out < W_out && n < N && c_out < C_out) {
        sum += bias[c_out];
        output[n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out] =
            (apply_relu && sum < 0.0f) ? 0.0f : sum;
    }
}

void conv2d_forward_opt_v2(const GPUTensorOpt& input, const GPUConvWeightsOpt& weights,
                           GPUTensorOpt& output, bool apply_relu, cudaStream_t stream) {
    int N = input.batch, C_in = input.channels, H_in = input.height, W_in = input.width;
    int C_out = weights.out_c, H_out = output.height, W_out = output.width;
    
    int tiles_w = (W_out + TILE_W_V2 - 1) / TILE_W_V2;
    int tiles_h = (H_out + TILE_H_V2 - 1) / TILE_H_V2;
    dim3 grid(tiles_h * tiles_w, C_out, N);
    dim3 block(TILE_W_V2, TILE_H_V2);
    
    conv2d_forward_fused_kernel<<<grid, block, 0, stream>>>(
        input.d_data, weights.d_weights, weights.d_bias, output.d_data,
        N, C_in, H_in, W_in, C_out, H_out, W_out, apply_relu);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// FUSED BACKWARD - Gradient computation with ReLU mask applied
// =============================================================================
__global__ void conv2d_backward_input_fused_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ weights,
    const float* __restrict__ fwd_output,  // For ReLU mask
    float* __restrict__ grad_input,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    bool had_relu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H_in * W_in;
    if (idx >= total) return;
    
    int w_in = idx % W_in; int tmp = idx / W_in;
    int h_in = tmp % H_in; tmp /= H_in;
    int c_in = tmp % C_in;
    int n = tmp / C_in;
    
    float grad = 0.0f;
    
    for (int c_out = 0; c_out < C_out; c_out++) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int h_out = h_in + 1 - kh, w_out = w_in + 1 - kw;
                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    int go_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
                    float go = grad_output[go_idx];
                    // Fused ReLU backward: mask gradient if forward output was <= 0
                    if (had_relu && fwd_output && fwd_output[go_idx] <= 0.0f) go = 0.0f;
                    grad += go * weights[c_out * (C_in * 9) + c_in * 9 + kh * 3 + kw];
                }
            }
        }
    }
    grad_input[idx] = grad;
}

__global__ void conv2d_backward_weights_kernel_v2(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    const float* __restrict__ fwd_output,
    float* __restrict__ grad_weights,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    bool had_relu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * 9;
    if (idx >= total) return;
    
    int kw = idx % 3; int tmp = idx / 3;
    int kh = tmp % 3; tmp /= 3;
    int c_in = tmp % C_in;
    int c_out = tmp / C_in;
    
    float grad = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                int h_in = h_out - 1 + kh, w_in = w_out - 1 + kw;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int go_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
                    float go = grad_output[go_idx];
                    if (had_relu && fwd_output && fwd_output[go_idx] <= 0.0f) go = 0.0f;
                    grad += go * input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in];
                }
            }
        }
    }
    grad_weights[idx] = grad;
}

__global__ void conv2d_backward_bias_kernel_v2(
    const float* __restrict__ grad_output,
    const float* __restrict__ fwd_output,
    float* __restrict__ grad_bias,
    int N, int C_out, int H_out, int W_out,
    bool had_relu)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C_out) return;
    
    float grad = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                int idx = n * (C_out * H_out * W_out) + c * (H_out * W_out) + h * W_out + w;
                float go = grad_output[idx];
                if (had_relu && fwd_output && fwd_output[idx] <= 0.0f) go = 0.0f;
                grad += go;
            }
        }
    }
    grad_bias[c] = grad;
}

void conv2d_backward_opt_v2(const GPUTensorOpt& input, const GPUTensorOpt& grad_output,
                            GPUConvWeightsOpt& weights, GPUTensorOpt& grad_input,
                            const GPUTensorOpt* forward_output, cudaStream_t stream) {
    int N = input.batch, C_in = input.channels, H_in = input.height, W_in = input.width;
    int C_out = weights.out_c, H_out = grad_output.height, W_out = grad_output.width;
    bool had_relu = (forward_output != nullptr);
    const float* fwd_ptr = had_relu ? forward_output->d_data : nullptr;
    
    int total_in = N * C_in * H_in * W_in;
    conv2d_backward_input_fused_kernel<<<(total_in + 255) / 256, 256, 0, stream>>>(
        grad_output.d_data, weights.d_weights, fwd_ptr, grad_input.d_data,
        N, C_in, H_in, W_in, C_out, H_out, W_out, had_relu);
    
    int total_w = C_out * C_in * 9;
    conv2d_backward_weights_kernel_v2<<<(total_w + 255) / 256, 256, 0, stream>>>(
        input.d_data, grad_output.d_data, fwd_ptr, weights.d_grad_w,
        N, C_in, H_in, W_in, C_out, H_out, W_out, had_relu);
    
    conv2d_backward_bias_kernel_v2<<<(C_out + 255) / 256, 256, 0, stream>>>(
        grad_output.d_data, fwd_ptr, weights.d_grad_b, N, C_out, H_out, W_out, had_relu);
    
    CUDA_CHECK(cudaGetLastError());
}
