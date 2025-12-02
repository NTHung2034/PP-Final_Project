#ifndef AUTOENCODER_GPU_H
#define AUTOENCODER_GPU_H

#include "layers_gpu/conv2d_gpu.cuh"
#include "layers_gpu/relu_gpu.cuh"  
#include "layers_gpu/maxpool_gpu.cuh"
#include "layers_gpu/upsample_gpu.cuh"
#include "layers_gpu/mse_loss_gpu.cuh"

#include <cuda_runtime.h>
#include <vector>

/**
 * =============================================================================
 * GPU AUTOENCODER MODEL - Header
 * =============================================================================
 * 
 * Convolutional Autoencoder for CIFAR-10 image reconstruction
 * 
 * Architecture:
 * INPUT (32x32x3) 
 * -> Conv1 (32x32x256) + ReLU 
 * -> MaxPool1 (16x16x256)
 * -> Conv2 (16x16x128) + ReLU 
 * -> MaxPool2 (8x8x128) [LATENT]
 * -> Conv3 (8x8x128) + ReLU
 * -> Upsample1 (16x16x128)
 * -> Conv4 (16x16x256) + ReLU
 * -> Upsample2 (32x32x256)
 * -> Conv5 (32x32x3) [OUTPUT]
 * 
 * Latent space: 8x8x128 = 8192 dimensions
 */

/**
 * SGD Weight Update
 * 
 * Updates weights using gradient descent: w = w - lr * grad_w
 * 
 * @param d_weights      Device pointer to weights
 * @param d_gradients    Device pointer to gradients
 * @param learning_rate  Learning rate
 * @param size           Number of elements
 * @param stream         CUDA stream for async execution
 */
void sgd_update_gpu(
    float* d_weights,
    const float* d_gradients,
    float learning_rate,
    int size,
    cudaStream_t stream = 0);

/**
 * GPU Autoencoder Class
 * 
 * Implements a convolutional autoencoder for image reconstruction
 * Supports both inference and training modes
 */
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
    
    /**
     * Constructor - Initialize network with Xavier initialization
     */
    GPUAutoencoder();
    
    /**
     * Destructor - Free all GPU resources
     */
    ~GPUAutoencoder();
    
    /**
     * Allocate memory for pooling indices
     * Only needed during training
     * 
     * @param batch_size Batch size for training
     */
    void allocatePoolingIndices(int batch_size);
    
    /**
     * Forward Pass - Inference Only
     * 
     * Runs full encoder-decoder pipeline
     * Does not store intermediate activations
     * 
     * @param input  Input tensor [N, 3, 32, 32]
     * @param output Output tensor [N, 3, 32, 32] (reconstructed image)
     */
    void forward_inference(const GPUTensor& input, GPUTensor& output);
    
    /**
     * Extract Features - Encoder Only
     * 
     * Returns latent representation (8x8x128 = 8192 dimensions)
     * Used for downstream tasks like SVM classification
     * 
     * @param input    Input tensor [N, 3, 32, 32]
     * @param features Output features [N, 128, 8, 8]
     */
    void extract_features(const GPUTensor& input, GPUTensor& features);
    
    /**
     * Forward-Backward-Update - Training Mode
     * 
     * Performs complete training step:
     * 1. Forward pass with activation caching
     * 2. Compute MSE loss
     * 3. Backward pass (gradient computation)
     * 4. SGD weight update
     * 
     * @param input        Input tensor [N, 3, 32, 32]
     * @param target       Target tensor [N, 3, 32, 32] (same as input for autoencoder)
     * @param learning_rate Learning rate for SGD
     * @param activations  Preallocated activation buffers (9 tensors)
     * @return             MSE loss value
     */
    float forward_backward_update(
        const GPUTensor& input,
        const GPUTensor& target,
        float learning_rate,
        std::vector<GPUTensor*>& activations);
};

#endif // AUTOENCODER_GPU_H
