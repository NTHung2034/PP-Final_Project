#include "layers_gpu_opt_v2/relu_gpu_opt_v2.cuh"

__global__ void relu_forward_kernel_v2(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

__global__ void relu_backward_kernel_v2(const float* grad_out, const float* fwd_out, float* grad_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad_in[idx] = (fwd_out[idx] > 0.0f) ? grad_out[idx] : 0.0f;
}

void relu_forward_opt_v2(GPUTensorOpt& tensor, cudaStream_t stream) {
    relu_forward_kernel_v2<<<(tensor.size + 255) / 256, 256, 0, stream>>>(tensor.d_data, tensor.size);
    CUDA_CHECK(cudaGetLastError());
}

void relu_backward_opt_v2(const GPUTensorOpt& grad_output, const GPUTensorOpt& forward_output,
                          GPUTensorOpt& grad_input, cudaStream_t stream) {
    relu_backward_kernel_v2<<<(grad_output.size + 255) / 256, 256, 0, stream>>>(
        grad_output.d_data, forward_output.d_data, grad_input.d_data, grad_output.size);
    CUDA_CHECK(cudaGetLastError());
}
