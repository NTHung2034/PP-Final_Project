#pragma once

#include <string>
#include <vector>
#include "data/gpu_tensor_opt.cuh"
#include "layers_gpu_opt_v2/conv2d_gpu_opt_v2.cuh"
#include "layers_gpu_opt_v2/maxpool_gpu_opt_v2.cuh"
#include "layers_gpu_opt_v2/upsample_gpu_opt_v2.cuh"
#include "layers_gpu_opt_v2/mse_loss_gpu_opt_v2.cuh"

class AutoencoderGPUOptV2 {
public:
    // Convolution weights
    GPUConvWeightsOpt *conv1, *conv2, *conv3, *conv4, *conv5;
    
    // Pre-allocated memory pool (with streams)
    GPUMemoryPool pool;
    
    // Double buffering: two input buffers for pipeline overlap
    GPUTensorOpt input_buffer[2];
    int current_buffer;  // 0 or 1, ping-pong between buffers
    
    // Stream for async data transfer
    cudaStream_t transfer_stream;
    
    AutoencoderGPUOptV2(int batch_size);
    ~AutoencoderGPUOptV2();
    
    // Forward pass - returns loss (uses current_buffer)
    float forward(int batch_size);
    
    // Backward pass + weight update (uses streams for overlap)
    void backward(float learning_rate);
    
    // Combined forward-backward-update (most efficient)
    float train_step(float learning_rate);
    
    // Async copy next batch to alternate buffer (for pipelining)
    void copy_input_async(const float* h_input, int batch_size);
    
    // Wait for async copy to complete and swap buffers
    void swap_buffers();
    
    // SGD weight update (streams for overlap)
    void update_weights(float learning_rate);
    
    // Save/load weights
    void save_weights(const std::string& dir);
    void load_weights(const std::string& dir);
};
