// Autoencoder GPU Optimized v1 - Memory Pool Strategy
#pragma once

#include "data/gpu_tensor_opt.cuh"
#include "layers_gpu_opt_v1/conv2d_gpu_opt_v1.cuh"
#include "layers_gpu_opt_v1/relu_gpu_opt_v1.cuh"
#include "layers_gpu_opt_v1/maxpool_gpu_opt_v1.cuh"
#include "layers_gpu_opt_v1/upsample_gpu_opt_v1.cuh"
#include "layers_gpu_opt_v1/mse_loss_gpu_opt_v1.cuh"

class AutoencoderGPUOptV1 {
public:
    // Convolution weights
    GPUConvWeightsOpt *conv1, *conv2, *conv3, *conv4, *conv5;
    
    // Pre-allocated memory pool
    GPUMemoryPool pool;
    
    // Input buffer (persistent)
    GPUTensorOpt input_buffer;
    
    AutoencoderGPUOptV1(int batch_size);
    ~AutoencoderGPUOptV1();
    
    // Forward pass - returns loss
    float forward(const float* h_input, int batch_size);
    
    // Backward pass + weight update
    void backward(float learning_rate);
    
    // Combined forward-backward-update (most efficient)
    float train_step(const float* h_input, int batch_size, float learning_rate);
    
    // SGD weight update
    void update_weights(float learning_rate);
    
    // Save/load weights
    void save_weights(const std::string& dir);
    void load_weights(const std::string& dir);
};
