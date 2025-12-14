#include "layers_gpu_opt_v1/relu_gpu_opt_v1.cuh"

__global__ void relu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

__global__ void relu_backward_kernel(const float* grad_out, const float* fwd_out, float* grad_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad_in[idx] = (fwd_out[idx] > 0.0f) ? grad_out[idx] : 0.0f;
}

void relu_forward_opt_v1(GPUTensorOpt& tensor) {
    relu_forward_kernel<<<(tensor.size + 255) / 256, 256>>>(tensor.d_data, tensor.size);
    CUDA_CHECK(cudaGetLastError());
}

void relu_backward_opt_v1(const GPUTensorOpt& grad_output, const GPUTensorOpt& forward_output, GPUTensorOpt& grad_input) {
    relu_backward_kernel<<<(grad_output.size + 255) / 256, 256>>>(
        grad_output.d_data, forward_output.d_data, grad_input.d_data, grad_output.size);
    CUDA_CHECK(cudaGetLastError());
}
