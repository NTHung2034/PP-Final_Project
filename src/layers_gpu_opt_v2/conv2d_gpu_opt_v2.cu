#include "layers_gpu_opt_v2/conv2d_gpu_opt_v2.cuh"

// Constant memory for biases (max 256 channels)
__constant__ float c_bias_v2[256];

// =============================================================================
// TILED CONVOLUTION FORWARD - Same as v1 (already optimized)
// =============================================================================
__global__ void conv2d_forward_tiled_kernel_v2(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    bool apply_relu)
{
    __shared__ float s_input[TILE_H + 2][TILE_W + 2];
    
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int tile_row = blockIdx.x / ((W_out + TILE_W - 1) / TILE_W);
    int tile_col = blockIdx.x % ((W_out + TILE_W - 1) / TILE_W);
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int h_out_base = tile_row * TILE_H;
    int w_out_base = tile_col * TILE_W;
    int h_out = h_out_base + ty;
    int w_out = w_out_base + tx;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < C_in; c_in++) {
        int h_in_base = h_out_base - PAD;
        int w_in_base = w_out_base - PAD;
        int h_load = h_in_base + ty;
        int w_load = w_in_base + tx;
        
        if (ty < TILE_H + 2 && tx < TILE_W + 2) {
            if (h_load >= 0 && h_load < H_in && w_load >= 0 && w_load < W_in) {
                s_input[ty][tx] = input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_load * W_in + w_load];
            } else {
                s_input[ty][tx] = 0.0f;
            }
        }
        
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
        
        if (h_out < H_out && w_out < W_out) {
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                #pragma unroll
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    int weight_idx = c_out * (C_in * KERNEL_SIZE * KERNEL_SIZE) + c_in * (KERNEL_SIZE * KERNEL_SIZE) + kh * KERNEL_SIZE + kw;
                    sum += s_input[ty + kh][tx + kw] * weights[weight_idx];
                }
            }
        }
        __syncthreads();
    }
    
    if (h_out < H_out && w_out < W_out && n < N && c_out < C_out) {
        sum += c_bias_v2[c_out];
        if (apply_relu && sum < 0.0f) sum = 0.0f;
        output[n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out] = sum;
    }
}

// =============================================================================
// V2 OPTIMIZATION: FUSED RELU BACKWARD WITH GRADIENT INPUT
// Instead of separate ReLU backward, we fuse it into conv backward
// =============================================================================
__global__ void conv2d_backward_input_fused_relu_kernel_v2(
    const float* __restrict__ grad_output,
    const float* __restrict__ forward_output,  // Output from forward pass (post-ReLU)
    const float* __restrict__ weights,
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
    
    // Fully unrolled inner loops for 3x3 kernel
    #pragma unroll
    for (int c_out = 0; c_out < 256; c_out++) {  // Max channels - compiler will optimize
        if (c_out >= C_out) break;
        
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int h_out = h_in + PAD - kh;
                int w_out = w_in + PAD - kw;
                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    int go_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
                    int w_idx = c_out * (C_in * KERNEL_SIZE * KERNEL_SIZE) + c_in * (KERNEL_SIZE * KERNEL_SIZE) + kh * KERNEL_SIZE + kw;
                    
                    float g = grad_output[go_idx];
                    // Fused ReLU backward: if forward output was 0 (ReLU killed it), gradient is 0
                    if (had_relu && forward_output[go_idx] <= 0.0f) {
                        g = 0.0f;
                    }
                    grad += g * weights[w_idx];
                }
            }
        }
    }
    grad_input[idx] = grad;
}

// =============================================================================
// V2 OPTIMIZATION: PARALLEL WEIGHT GRADIENT COMPUTATION
// Uses shared memory for partial sums
// =============================================================================
__global__ void conv2d_backward_weights_kernel_v2(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    const float* __restrict__ forward_output,
    float* __restrict__ grad_weights,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    bool had_relu)
{
    // Each block computes gradient for one weight element
    __shared__ float s_partial[256];  // Partial sums
    
    int idx = blockIdx.x;
    int total = C_out * C_in * KERNEL_SIZE * KERNEL_SIZE;
    if (idx >= total) return;
    
    int kw = idx % KERNEL_SIZE; int tmp = idx / KERNEL_SIZE;
    int kh = tmp % KERNEL_SIZE; tmp /= KERNEL_SIZE;
    int c_in = tmp % C_in;
    int c_out = tmp / C_in;
    
    int tid = threadIdx.x;
    int spatial_size = N * H_out * W_out;
    
    float local_sum = 0.0f;
    
    // Each thread handles multiple spatial positions
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int w_out = i % W_out;
        int h_out = (i / W_out) % H_out;
        int n = i / (H_out * W_out);
        
        int h_in = h_out - PAD + kh;
        int w_in = w_out - PAD + kw;
        
        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            int go_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
            float g = grad_output[go_idx];
            // Fused ReLU
            if (had_relu && forward_output[go_idx] <= 0.0f) {
                g = 0.0f;
            }
            local_sum += g * input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in];
        }
    }
    
    s_partial[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            s_partial[tid] += s_partial[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        grad_weights[idx] = s_partial[0];
    }
}

// =============================================================================
// V2 OPTIMIZATION: PARALLEL BIAS GRADIENT WITH REDUCTION
// =============================================================================
__global__ void conv2d_backward_bias_kernel_v2(
    const float* __restrict__ grad_output,
    const float* __restrict__ forward_output,
    float* __restrict__ grad_bias,
    int N, int C_out, int H_out, int W_out,
    bool had_relu)
{
    __shared__ float s_partial[256];
    
    int c = blockIdx.x;
    if (c >= C_out) return;
    
    int tid = threadIdx.x;
    int spatial_size = N * H_out * W_out;
    
    float local_sum = 0.0f;
    
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int w = i % W_out;
        int h = (i / W_out) % H_out;
        int n = i / (H_out * W_out);
        
        int idx = n * (C_out * H_out * W_out) + c * (H_out * W_out) + h * W_out + w;
        float g = grad_output[idx];
        // Fused ReLU backward
        if (had_relu && forward_output[idx] <= 0.0f) {
            g = 0.0f;
        }
        local_sum += g;
    }
    
    s_partial[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            s_partial[tid] += s_partial[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        grad_bias[c] = s_partial[0];
    }
}

// =============================================================================
// API FUNCTIONS
// =============================================================================
void conv2d_forward_opt_v2(
    const GPUTensorOpt& input, 
    const GPUConvWeightsOpt& weights, 
    GPUTensorOpt& output, 
    bool apply_relu,
    cudaStream_t stream)
{
    int N = input.batch, C_in = input.channels, H_in = input.height, W_in = input.width;
    int C_out = weights.out_c, H_out = output.height, W_out = output.width;
    
    CUDA_CHECK(cudaMemcpyToSymbol(c_bias_v2, weights.d_bias, weights.bias_size * sizeof(float)));
    
    int tiles_w = (W_out + TILE_W - 1) / TILE_W;
    int tiles_h = (H_out + TILE_H - 1) / TILE_H;
    dim3 grid(tiles_h * tiles_w, C_out, N);
    dim3 block(TILE_W + 2, TILE_H + 2);
    
    conv2d_forward_tiled_kernel_v2<<<grid, block, 0, stream>>>(
        input.d_data, weights.d_weights, output.d_data,
        N, C_in, H_in, W_in, C_out, H_out, W_out, apply_relu);
    CUDA_CHECK(cudaGetLastError());
}

void conv2d_backward_opt_v2(
    const GPUTensorOpt& input,
    const GPUTensorOpt& grad_output,
    const GPUTensorOpt& forward_output,
    GPUConvWeightsOpt& weights,
    GPUTensorOpt& grad_input,
    bool had_relu,
    cudaStream_t stream1,
    cudaStream_t stream2,
    cudaStream_t stream3)
{
    int N = input.batch, C_in = input.channels, H_in = input.height, W_in = input.width;
    int C_out = weights.out_c, H_out = grad_output.height, W_out = grad_output.width;
    
    // Stream 1: Gradient w.r.t. input (critical path, on main stream)
    int total_in = N * C_in * H_in * W_in;
    conv2d_backward_input_fused_relu_kernel_v2<<<(total_in + 255) / 256, 256, 0, stream1>>>(
        grad_output.d_data, forward_output.d_data, weights.d_weights, grad_input.d_data,
        N, C_in, H_in, W_in, C_out, H_out, W_out, had_relu);
    
    // Stream 2: Gradient w.r.t. weights (can run in parallel)
    int total_w = C_out * C_in * KERNEL_SIZE * KERNEL_SIZE;
    conv2d_backward_weights_kernel_v2<<<total_w, 256, 0, stream2>>>(
        input.d_data, grad_output.d_data, forward_output.d_data, weights.d_grad_w,
        N, C_in, H_in, W_in, C_out, H_out, W_out, had_relu);
    
    // Stream 3: Gradient w.r.t. bias (can run in parallel)
    conv2d_backward_bias_kernel_v2<<<C_out, 256, 0, stream3>>>(
        grad_output.d_data, forward_output.d_data, weights.d_grad_b,
        N, C_out, H_out, W_out, had_relu);
    
    CUDA_CHECK(cudaGetLastError());
}

void conv2d_sync_streams(cudaStream_t s1, cudaStream_t s2, cudaStream_t s3) {
    if (s1) CUDA_CHECK(cudaStreamSynchronize(s1));
    if (s2) CUDA_CHECK(cudaStreamSynchronize(s2));
    if (s3) CUDA_CHECK(cudaStreamSynchronize(s3));
}
