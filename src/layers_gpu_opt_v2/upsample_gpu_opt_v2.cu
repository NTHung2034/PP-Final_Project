#include "layers_gpu_opt_v2/upsample_gpu_opt_v2.cuh"

__global__ void upsample2d_forward_kernel_v2(
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
    
    output[idx] = input[n * (C * H_in * W_in) + c * (H_in * W_in) + (h_out / 2) * W_in + (w_out / 2)];
}

__global__ void upsample2d_backward_kernel_v2(
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
    int base = n * (C * H_out * W_out) + c * (H_out * W_out);
    #pragma unroll
    for (int dh = 0; dh < 2; dh++) {
        #pragma unroll
        for (int dw = 0; dw < 2; dw++) {
            grad += grad_out[base + (h_in * 2 + dh) * W_out + (w_in * 2 + dw)];
        }
    }
    grad_in[idx] = grad;
}

void upsample2d_forward_opt_v2(const GPUTensorOpt& input, GPUTensorOpt& output, cudaStream_t stream) {
    upsample2d_forward_kernel_v2<<<(output.size + 255) / 256, 256, 0, stream>>>(
        input.d_data, output.d_data,
        input.batch, input.channels, input.height, input.width, output.height, output.width);
    CUDA_CHECK(cudaGetLastError());
}

void upsample2d_backward_opt_v2(const GPUTensorOpt& grad_output, GPUTensorOpt& grad_input, cudaStream_t stream) {
    upsample2d_backward_kernel_v2<<<(grad_input.size + 255) / 256, 256, 0, stream>>>(
        grad_output.d_data, grad_input.d_data,
        grad_input.batch, grad_input.channels, grad_input.height, grad_input.width,
        grad_output.height, grad_output.width);
    CUDA_CHECK(cudaGetLastError());
}
