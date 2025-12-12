// Upsample 2x (Nearest Neighbor) - Optimized v1
#include "layers_gpu_opt_v1/upsample_gpu_opt_v1.cuh"

__global__ void upsample2d_forward_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;
    
    int w_out = idx % W_out; int tmp = idx / W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;
    
    int h_in = h_out / 2;
    int w_in = w_out / 2;
    output[idx] = input[n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in];
}

__global__ void upsample2d_backward_kernel(
    const float* __restrict__ grad_out, float* __restrict__ grad_in,
    int N, int C, int H_in, int W_in, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_in * W_in;
    if (idx >= total) return;
    
    int w_in = idx % W_in; int tmp = idx / W_in;
    int h_in = tmp % H_in; tmp /= H_in;
    int c = tmp % C;
    int n = tmp / C;
    
    float grad = 0.0f;
    for (int dh = 0; dh < 2; dh++) {
        for (int dw = 0; dw < 2; dw++) {
            int h_out = h_in * 2 + dh;
            int w_out = w_in * 2 + dw;
            grad += grad_out[n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out];
        }
    }
    grad_in[idx] = grad;
}

void upsample2d_forward_opt_v1(const GPUTensorOpt& input, GPUTensorOpt& output) {
    upsample2d_forward_kernel<<<(output.size + 255) / 256, 256>>>(
        input.d_data, output.d_data,
        input.batch, input.channels, input.height, input.width, output.height, output.width);
    CUDA_CHECK(cudaGetLastError());
}

void upsample2d_backward_opt_v1(const GPUTensorOpt& grad_output, GPUTensorOpt& grad_input) {
    upsample2d_backward_kernel<<<(grad_input.size + 255) / 256, 256>>>(
        grad_output.d_data, grad_input.d_data,
        grad_input.batch, grad_input.channels, grad_input.height, grad_input.width,
        grad_output.height, grad_output.width);
    CUDA_CHECK(cudaGetLastError());
}
