#include "models/autoencoder_gpu.cuh"
 
#include <vector>
#include <cmath>

__global__ void sgd_update_kernel(
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

void sgd_update_gpu(
    float* d_weights,
    const float* d_gradients,
    float learning_rate,
    int size,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    sgd_update_kernel<<<blocks, threads, 0, stream>>>(
        d_weights, d_gradients, learning_rate, size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

GPUAutoencoder::GPUAutoencoder() {
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
    
    // Allocate pooling indices (only needed during training)
    d_pool1_indices = nullptr;
    d_pool2_indices = nullptr;
    
    // Create non-blocking streams for concurrency
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_transfer, cudaStreamNonBlocking));
}

GPUAutoencoder::~GPUAutoencoder() {
    delete conv1;
    delete conv2;
    delete conv3;
    delete conv4;
    delete conv5;
    
    if (d_pool1_indices) cudaFree(d_pool1_indices);
    if (d_pool2_indices) cudaFree(d_pool2_indices);
    
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_transfer);
}

void GPUAutoencoder::allocatePoolingIndices(int batch_size) {
    if (!d_pool1_indices) {
        size_t pool1_size = batch_size * 256 * 16 * 16;
        CUDA_CHECK(cudaMalloc(&d_pool1_indices, pool1_size * sizeof(int)));
    }
    if (!d_pool2_indices) {
        size_t pool2_size = batch_size * 128 * 8 * 8;
        CUDA_CHECK(cudaMalloc(&d_pool2_indices, pool2_size * sizeof(int)));
    }
}

void GPUAutoencoder::forward_inference(const GPUTensor& input, GPUTensor& output) {
    int batch = input.batch;
    
    // Temporary activation buffers (device only - no host mirror)
    GPUTensor act1(batch, 256, 32, 32, true);  // After Conv1
    GPUTensor act2(batch, 256, 16, 16, true);  // After Pool1
    GPUTensor act3(batch, 128, 16, 16, true);  // After Conv2
    GPUTensor act4(batch, 128, 8, 8, true);    // After Pool2 (LATENT)
    GPUTensor act5(batch, 128, 8, 8, true);    // After Conv3
    GPUTensor act6(batch, 128, 16, 16, true);  // After Upsample1
    GPUTensor act7(batch, 256, 16, 16, true);  // After Conv4
    GPUTensor act8(batch, 256, 32, 32, true);  // After Upsample2
    
    // ENCODER
    conv2d_forward_gpu(input, *conv1, act1, 3, 3, 1, 1, true, stream_compute);
    
    // Need temporary storage for pooling in inference
    int* temp_pool1_idx;
    CUDA_CHECK(cudaMalloc(&temp_pool1_idx, batch * 256 * 16 * 16 * sizeof(int)));
    maxpool2d_forward_gpu(act1, act2, temp_pool1_idx, stream_compute);
    
    conv2d_forward_gpu(act2, *conv2, act3, 3, 3, 1, 1, true, stream_compute);
    
    int* temp_pool2_idx;
    CUDA_CHECK(cudaMalloc(&temp_pool2_idx, batch * 128 * 8 * 8 * sizeof(int)));
    maxpool2d_forward_gpu(act3, act4, temp_pool2_idx, stream_compute);
    
    // DECODER
    conv2d_forward_gpu(act4, *conv3, act5, 3, 3, 1, 1, true, stream_compute);
    upsample2d_forward_gpu(act5, act6, 2, stream_compute);
    conv2d_forward_gpu(act6, *conv4, act7, 3, 3, 1, 1, true, stream_compute);
    upsample2d_forward_gpu(act7, act8, 2, stream_compute);
    conv2d_forward_gpu(act8, *conv5, output, 3, 3, 1, 1, false, stream_compute);
    
    // Wait for completion
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    
    // Free temporary indices
    CUDA_CHECK(cudaFree(temp_pool1_idx));
    CUDA_CHECK(cudaFree(temp_pool2_idx));
}

void GPUAutoencoder::extract_features(const GPUTensor& input, GPUTensor& features) {
    int batch = input.batch;
    
    // Run encoder only
    GPUTensor act1(batch, 256, 32, 32, true);
    GPUTensor act2(batch, 256, 16, 16, true);
    GPUTensor act3(batch, 128, 16, 16, true);
    
    conv2d_forward_gpu(input, *conv1, act1, 3, 3, 1, 1, true, stream_compute);
    
    int* temp_pool_idx;
    CUDA_CHECK(cudaMalloc(&temp_pool_idx, batch * 256 * 16 * 16 * sizeof(int)));
    maxpool2d_forward_gpu(act1, act2, temp_pool_idx, stream_compute);
    CUDA_CHECK(cudaFree(temp_pool_idx));
    
    conv2d_forward_gpu(act2, *conv2, act3, 3, 3, 1, 1, true, stream_compute);
    
    CUDA_CHECK(cudaMalloc(&temp_pool_idx, batch * 128 * 8 * 8 * sizeof(int)));
    maxpool2d_forward_gpu(act3, features, temp_pool_idx, stream_compute);
    CUDA_CHECK(cudaFree(temp_pool_idx));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
}

float GPUAutoencoder::forward_backward_update(
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
    
    // FORWARD PASS
    conv2d_forward_gpu(input, *conv1, act1, 3, 3, 1, 1, true, stream_compute);
    maxpool2d_forward_gpu(act1, act2, d_pool1_indices, stream_compute);
    conv2d_forward_gpu(act2, *conv2, act3, 3, 3, 1, 1, true, stream_compute);
    maxpool2d_forward_gpu(act3, act4, d_pool2_indices, stream_compute);
    
    conv2d_forward_gpu(act4, *conv3, act5, 3, 3, 1, 1, true, stream_compute);
    upsample2d_forward_gpu(act5, act6, 2, stream_compute);
    conv2d_forward_gpu(act6, *conv4, act7, 3, 3, 1, 1, true, stream_compute);
    upsample2d_forward_gpu(act7, act8, 2, stream_compute);
    conv2d_forward_gpu(act8, *conv5, output, 3, 3, 1, 1, false, stream_compute);
    
    // Compute loss
    float loss = mse_loss_forward_gpu(output, target, stream_compute);
    
    // BACKWARD PASS
    GPUTensor grad_output(batch, 3, 32, 32, true);
    mse_loss_backward_gpu(output, target, grad_output, stream_compute);
    
    // Conv5 backward
    GPUTensor grad_act8(batch, 256, 32, 32, true);
    conv2d_backward_gpu(act8, grad_output, *conv5, grad_act8, 3, 3, 1, 1, stream_compute);
    
    // Upsample2 backward
    GPUTensor grad_act7(batch, 256, 16, 16, true);
    upsample2d_backward_gpu(grad_act8, grad_act7, 2, stream_compute);
    
    // Conv4 backward
    GPUTensor grad_act6(batch, 128, 16, 16, true);
    conv2d_backward_gpu(act6, grad_act7, *conv4, grad_act6, 3, 3, 1, 1, stream_compute);
    
    // Upsample1 backward
    GPUTensor grad_act5(batch, 128, 8, 8, true);
    upsample2d_backward_gpu(grad_act6, grad_act5, 2, stream_compute);
    
    // Conv3 backward
    GPUTensor grad_act4(batch, 128, 8, 8, true);
    conv2d_backward_gpu(act4, grad_act5, *conv3, grad_act4, 3, 3, 1, 1, stream_compute);
    
    // Pool2 backward
    GPUTensor grad_act3(batch, 128, 16, 16, true);
    maxpool2d_backward_gpu(grad_act4, d_pool2_indices, grad_act3, stream_compute);
    
    // Conv2 backward
    GPUTensor grad_act2(batch, 256, 16, 16, true);
    conv2d_backward_gpu(act2, grad_act3, *conv2, grad_act2, 3, 3, 1, 1, stream_compute);
    
    // Pool1 backward
    GPUTensor grad_act1(batch, 256, 32, 32, true);
    maxpool2d_backward_gpu(grad_act2, d_pool1_indices, grad_act1, stream_compute);
    
    // Conv1 backward
    GPUTensor grad_input(batch, 3, 32, 32, true);
    conv2d_backward_gpu(input, grad_act1, *conv1, grad_input, 3, 3, 1, 1, stream_compute);
    
    // UPDATE WEIGHTS (SGD)
    sgd_update_gpu(conv1->d_weights, conv1->d_grad_w, learning_rate, conv1->weight_size, stream_compute);
    sgd_update_gpu(conv1->d_bias, conv1->d_grad_b, learning_rate, conv1->bias_size, stream_compute);
    
    sgd_update_gpu(conv2->d_weights, conv2->d_grad_w, learning_rate, conv2->weight_size, stream_compute);
    sgd_update_gpu(conv2->d_bias, conv2->d_grad_b, learning_rate, conv2->bias_size, stream_compute);
    
    sgd_update_gpu(conv3->d_weights, conv3->d_grad_w, learning_rate, conv3->weight_size, stream_compute);
    sgd_update_gpu(conv3->d_bias, conv3->d_grad_b, learning_rate, conv3->bias_size, stream_compute);
    
    sgd_update_gpu(conv4->d_weights, conv4->d_grad_w, learning_rate, conv4->weight_size, stream_compute);
    sgd_update_gpu(conv4->d_bias, conv4->d_grad_b, learning_rate, conv4->bias_size, stream_compute);
    
    sgd_update_gpu(conv5->d_weights, conv5->d_grad_w, learning_rate, conv5->weight_size, stream_compute);
    sgd_update_gpu(conv5->d_bias, conv5->d_grad_b, learning_rate, conv5->bias_size, stream_compute);
    
    // Synchronize before returning
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    
    return loss;
}