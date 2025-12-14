#include "models/autoencoder_gpu_naive.cuh"

#include <vector>
#include <cmath>

__global__ void sgd_update_kernel_naive(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float learning_rate,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

void sgd_update_gpu_naive(
    float* d_weights,
    const float* d_gradients,
    float learning_rate,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    sgd_update_kernel_naive<<<blocks, threads>>>(d_weights, d_gradients, learning_rate, size);
    
    CUDA_CHECK(cudaGetLastError());
}

GPUAutoencoderNaive::GPUAutoencoderNaive() {
    // Initialize weights (3x3 kernels, padding=1, stride=1)
    conv1 = new GPUConvWeights(256, 3, 3, 3);
    conv2 = new GPUConvWeights(128, 256, 3, 3);
    conv3 = new GPUConvWeights(128, 128, 3, 3);
    conv4 = new GPUConvWeights(256, 128, 3, 3);
    conv5 = new GPUConvWeights(3, 256, 3, 3);
    
    // Initialize with Xavier
    conv1->initializeXavier();
    conv2->initializeXavier();
    conv3->initializeXavier();
    conv4->initializeXavier();
    conv5->initializeXavier();
    
    // Pooling indices (allocated later)
    d_pool1_indices = nullptr;
    d_pool2_indices = nullptr;
}

GPUAutoencoderNaive::~GPUAutoencoderNaive() {
    delete conv1;
    delete conv2;
    delete conv3;
    delete conv4;
    delete conv5;
    
    if (d_pool1_indices) cudaFree(d_pool1_indices);
    if (d_pool2_indices) cudaFree(d_pool2_indices);
}

void GPUAutoencoderNaive::allocatePoolingIndices(int batch_size) {
    if (!d_pool1_indices) {
        size_t pool1_size = batch_size * 256 * 16 * 16;
        CUDA_CHECK(cudaMalloc(&d_pool1_indices, pool1_size * sizeof(int)));
    }
    if (!d_pool2_indices) {
        size_t pool2_size = batch_size * 128 * 8 * 8;
        CUDA_CHECK(cudaMalloc(&d_pool2_indices, pool2_size * sizeof(int)));
    }
}

void GPUAutoencoderNaive::forward_inference(const GPUTensor& input, GPUTensor& output) {
    int batch = input.batch;
    
    // Temporary activation buffers (device only)
    GPUTensor act1(batch, 256, 32, 32, true);  // After Conv1
    GPUTensor act2(batch, 256, 16, 16, true);  // After Pool1
    GPUTensor act3(batch, 128, 16, 16, true);  // After Conv2
    GPUTensor act4(batch, 128, 8, 8, true);    // After Pool2 (LATENT)
    GPUTensor act5(batch, 128, 8, 8, true);    // After Conv3
    GPUTensor act6(batch, 128, 16, 16, true);  // After Upsample1
    GPUTensor act7(batch, 256, 16, 16, true);  // After Conv4
    GPUTensor act8(batch, 256, 32, 32, true);  // After Upsample2
    
    // Temporary pooling indices
    int* temp_pool1_idx;
    int* temp_pool2_idx;
    CUDA_CHECK(cudaMalloc(&temp_pool1_idx, batch * 256 * 16 * 16 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&temp_pool2_idx, batch * 128 * 8 * 8 * sizeof(int)));
    
    // ENCODER - Using naive GPU kernels
    conv2d_forward_gpu_naive(input, *conv1, act1, 3, 3, 1, 1, true);
    maxpool2d_forward_gpu_naive(act1, act2, temp_pool1_idx);
    conv2d_forward_gpu_naive(act2, *conv2, act3, 3, 3, 1, 1, true);
    maxpool2d_forward_gpu_naive(act3, act4, temp_pool2_idx);
    
    // DECODER - Using naive GPU kernels
    conv2d_forward_gpu_naive(act4, *conv3, act5, 3, 3, 1, 1, true);
    upsample2d_forward_gpu_naive(act5, act6, 2);
    conv2d_forward_gpu_naive(act6, *conv4, act7, 3, 3, 1, 1, true);
    upsample2d_forward_gpu_naive(act7, act8, 2);
    conv2d_forward_gpu_naive(act8, *conv5, output, 3, 3, 1, 1, false);
    
    // Wait for completion
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Free temporary indices
    CUDA_CHECK(cudaFree(temp_pool1_idx));
    CUDA_CHECK(cudaFree(temp_pool2_idx));
}

void GPUAutoencoderNaive::extract_features(const GPUTensor& input, GPUTensor& features) {
    int batch = input.batch;
    
    // Run encoder only
    GPUTensor act1(batch, 256, 32, 32, true);
    GPUTensor act2(batch, 256, 16, 16, true);
    GPUTensor act3(batch, 128, 16, 16, true);
    
    int* temp_pool_idx;
    
    // Encoder forward pass with naive kernels
    conv2d_forward_gpu_naive(input, *conv1, act1, 3, 3, 1, 1, true);
    
    CUDA_CHECK(cudaMalloc(&temp_pool_idx, batch * 256 * 16 * 16 * sizeof(int)));
    maxpool2d_forward_gpu_naive(act1, act2, temp_pool_idx);
    CUDA_CHECK(cudaFree(temp_pool_idx));
    
    conv2d_forward_gpu_naive(act2, *conv2, act3, 3, 3, 1, 1, true);
    
    CUDA_CHECK(cudaMalloc(&temp_pool_idx, batch * 128 * 8 * 8 * sizeof(int)));
    maxpool2d_forward_gpu_naive(act3, features, temp_pool_idx);
    CUDA_CHECK(cudaFree(temp_pool_idx));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

float GPUAutoencoderNaive::forward_backward_update(
    const GPUTensor& input,
    const GPUTensor& target,
    float learning_rate,
    std::vector<GPUTensor*>& activations)
{
    int batch = input.batch;
    
    // Activations (reuse preallocated buffers)
    GPUTensor& act1 = *activations[0];   // Conv1 output
    GPUTensor& act2 = *activations[1];   // Pool1 output
    GPUTensor& act3 = *activations[2];   // Conv2 output
    GPUTensor& act4 = *activations[3];   // Pool2 output (latent)
    GPUTensor& act5 = *activations[4];   // Conv3 output
    GPUTensor& act6 = *activations[5];   // Upsample1 output
    GPUTensor& act7 = *activations[6];   // Conv4 output
    GPUTensor& act8 = *activations[7];   // Upsample2 output
    GPUTensor& output = *activations[8]; // Final output
    
    // forward pass  
    conv2d_forward_gpu_naive(input, *conv1, act1, 3, 3, 1, 1, true);
    maxpool2d_forward_gpu_naive(act1, act2, d_pool1_indices);
    conv2d_forward_gpu_naive(act2, *conv2, act3, 3, 3, 1, 1, true);
    maxpool2d_forward_gpu_naive(act3, act4, d_pool2_indices);
    
    conv2d_forward_gpu_naive(act4, *conv3, act5, 3, 3, 1, 1, true);
    upsample2d_forward_gpu_naive(act5, act6, 2);
    conv2d_forward_gpu_naive(act6, *conv4, act7, 3, 3, 1, 1, true);
    upsample2d_forward_gpu_naive(act7, act8, 2);
    conv2d_forward_gpu_naive(act8, *conv5, output, 3, 3, 1, 1, false);
    
    // compute loss
    float loss = mse_loss_forward_gpu_naive(output, target);
    
    // backward pass
    GPUTensor grad_output(batch, 3, 32, 32, true);
    mse_loss_backward_gpu_naive(output, target, grad_output);
    
    // Conv5 backward
    GPUTensor grad_act8(batch, 256, 32, 32, true);
    conv2d_backward_gpu_naive(act8, grad_output, *conv5, grad_act8, 3, 3, 1, 1);
    
    // Upsample2 backward
    GPUTensor grad_act7(batch, 256, 16, 16, true);
    upsample2d_backward_gpu_naive(grad_act8, grad_act7, 2);
    
    // Conv4 backward
    GPUTensor grad_act6(batch, 128, 16, 16, true);
    conv2d_backward_gpu_naive(act6, grad_act7, *conv4, grad_act6, 3, 3, 1, 1);
    
    // Upsample1 backward
    GPUTensor grad_act5(batch, 128, 8, 8, true);
    upsample2d_backward_gpu_naive(grad_act6, grad_act5, 2);
    
    // Conv3 backward
    GPUTensor grad_act4(batch, 128, 8, 8, true);
    conv2d_backward_gpu_naive(act4, grad_act5, *conv3, grad_act4, 3, 3, 1, 1);
    
    // Pool2 backward
    GPUTensor grad_act3(batch, 128, 16, 16, true);
    maxpool2d_backward_gpu_naive(grad_act4, d_pool2_indices, grad_act3);
    
    // Conv2 backward
    GPUTensor grad_act2(batch, 256, 16, 16, true);
    conv2d_backward_gpu_naive(act2, grad_act3, *conv2, grad_act2, 3, 3, 1, 1);
    
    // Pool1 backward
    GPUTensor grad_act1(batch, 256, 32, 32, true);
    maxpool2d_backward_gpu_naive(grad_act2, d_pool1_indices, grad_act1);
    
    // Conv1 backward
    GPUTensor grad_input(batch, 3, 32, 32, true);
    conv2d_backward_gpu_naive(input, grad_act1, *conv1, grad_input, 3, 3, 1, 1);
    
    // update weights (SGD) 
    sgd_update_gpu_naive(conv1->d_weights, conv1->d_grad_w, learning_rate, conv1->weight_size);
    sgd_update_gpu_naive(conv1->d_bias, conv1->d_grad_b, learning_rate, conv1->bias_size);
    
    sgd_update_gpu_naive(conv2->d_weights, conv2->d_grad_w, learning_rate, conv2->weight_size);
    sgd_update_gpu_naive(conv2->d_bias, conv2->d_grad_b, learning_rate, conv2->bias_size);
    
    sgd_update_gpu_naive(conv3->d_weights, conv3->d_grad_w, learning_rate, conv3->weight_size);
    sgd_update_gpu_naive(conv3->d_bias, conv3->d_grad_b, learning_rate, conv3->bias_size);
    
    sgd_update_gpu_naive(conv4->d_weights, conv4->d_grad_w, learning_rate, conv4->weight_size);
    sgd_update_gpu_naive(conv4->d_bias, conv4->d_grad_b, learning_rate, conv4->bias_size);
    
    sgd_update_gpu_naive(conv5->d_weights, conv5->d_grad_w, learning_rate, conv5->weight_size);
    sgd_update_gpu_naive(conv5->d_bias, conv5->d_grad_b, learning_rate, conv5->bias_size);
    
    // Synchronize before returning
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return loss;
}