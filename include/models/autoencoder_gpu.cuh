#pragma once

#include "layers_gpu/conv2d_gpu.cuh"
#include "layers_gpu/relu_gpu.cuh"  
#include "layers_gpu/maxpool_gpu.cuh"
#include "layers_gpu/upsample_gpu.cuh"
#include "layers_gpu/mse_loss_gpu.cuh"

#include <cuda_runtime.h>
#include <vector>

// SGD Weight Update
// Updates weights using gradient descent: w = w - lr * grad_w

void sgd_update_gpu(
    float* d_weights,
    const float* d_gradients,
    float learning_rate,
    int size,
    cudaStream_t stream = 0);


class GPUAutoencoder {
public:
    // Network dimensions
    static constexpr int INPUT_H = 32;
    static constexpr int INPUT_W = 32;
    static constexpr int INPUT_C = 3;
    
    // Layer weights
    GPUConvWeights* conv1;  // 3 -> 256
    GPUConvWeights* conv2;  // 256 -> 128
    GPUConvWeights* conv3;  // 128 -> 128 (decoder)
    GPUConvWeights* conv4;  // 128 -> 256
    GPUConvWeights* conv5;  // 256 -> 3
    
    // Max pooling indices (needed for backward pass)
    int* d_pool1_indices;
    int* d_pool2_indices;
    
    // CUDA streams for pipeline parallelism
    cudaStream_t stream_compute;
    cudaStream_t stream_transfer;
    
    GPUAutoencoder();
    
    ~GPUAutoencoder();
    
    // Allocate memory for pooling indices
    // Only needed during training
    void allocatePoolingIndices(int batch_size);
    
    // Forward Pass - Inference Only
    // Runs full encoder-decoder pipeline
    void forward_inference(const GPUTensor& input, GPUTensor& output);
    

    // Extract Features - Encoder Only

    // Returns latent representation (8x8x128 = 8192 dimensions)
    void extract_features(const GPUTensor& input, GPUTensor& features);
    
    
    // Forward-Backward-Update - Training Mode
    // @return             MSE loss value
    float forward_backward_update(
        const GPUTensor& input,
        const GPUTensor& target,
        float learning_rate,
        std::vector<GPUTensor*>& activations);
};

