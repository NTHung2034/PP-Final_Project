#pragma once

#include <string>
#include "data/gpu_tensor_opt_v2.cuh"
#include "layers_gpu_opt_v2/conv2d_gpu_opt_v2.cuh"
#include "layers_gpu_opt_v2/relu_gpu_opt_v2.cuh"
#include "layers_gpu_opt_v2/maxpool_gpu_opt_v2.cuh"
#include "layers_gpu_opt_v2/upsample_gpu_opt_v2.cuh"
#include "layers_gpu_opt_v2/mse_loss_gpu_opt_v2.cuh"

class AutoencoderGPUOptV2 {
public:
    GPUConvWeightsOpt *conv1, *conv2, *conv3, *conv4, *conv5;
    GPUMemoryPoolV2 pool;
    GPUTensorOpt input_buffer;
    
    AutoencoderGPUOptV2(int batch_size);
    ~AutoencoderGPUOptV2();
    
    // Forward pass with optional stream
    float forward(const float* h_input, int batch_size);
    
    // Backward pass + weight update
    void backward(float learning_rate);
    
    // Combined training step
    float train_step(const float* h_input, int batch_size, float learning_rate);
    
    // Async input transfer (for pipeline)
    void async_load_input(const float* h_input, int batch_size);
    
    // Forward/backward using streams (for pipeline)
    float forward_stream();
    void backward_stream(float learning_rate);
    
    void update_weights(float learning_rate);
    void save_weights(const std::string& dir);
    void load_weights(const std::string& dir);
    
    cudaStream_t get_compute_stream() { return pool.compute_stream; }
};
