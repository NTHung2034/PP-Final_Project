#pragma once

#include "data/gpu_data_types.cuh"
#include "layers_gpu_naive/conv2d_gpu_naive.cuh"
#include "layers_gpu_naive/relu_gpu_naive.cuh"  
#include "layers_gpu_naive/maxpool_gpu_naive.cuh"
#include "layers_gpu_naive/upsample_gpu_naive.cuh"
#include "layers_gpu_naive/mse_loss_gpu_naive.cuh"

#include <cuda_runtime.h>
#include <vector>

//  SGD Weight Update
void sgd_update_gpu_naive(
    float* d_weights,
    const float* d_gradients,
    float learning_rate,
    int size);

// GPU Autoencoder Class
class GPUAutoencoderNaive {
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
    
    GPUAutoencoderNaive();
    ~GPUAutoencoderNaive();
    
    // Allocate memory for pooling indices
    void allocatePoolingIndices(int batch_size);
    
    //  Forward Pass - Inference Only
    //  Runs encoder-decoder pipeline without storing activations
    //  Input tensor [N, 3, 32, 32]
    //  Output tensor [N, 3, 32, 32]
    void forward_inference(const GPUTensor& input, GPUTensor& output);
    
    //  Extract Features - Encoder Only
    //  Returns latent representation (8x8x128 = 8192 dimensions)
    //  Input tensor [N, 3, 32, 32]
    //  Output features [N, 128, 8, 8]
    void extract_features(const GPUTensor& input, GPUTensor& features);
    
    //  Forward-Backward-Update - Training Mode
    //  Complete training step:
    //  1. Forward pass with activation caching
    //  2. Compute MSE loss
    //  3. Backward pass
    //  4. SGD weight update

    //  Input tensor [N, 3, 32, 32]
    //  Target tensor [N, 3, 32, 32]
    //  learning_rate Learning rate for SGD
    //  activations  Preallocated activation buffers
    //  return             MSE loss value
    float forward_backward_update(
        const GPUTensor& input,
        const GPUTensor& target,
        float learning_rate,
        std::vector<GPUTensor*>& activations);
};
