#include "models/autoencoder_gpu_opt_v2.cuh"
#include <fstream>
#include <cstring>

// SGD update kernel (stream-enabled)
__global__ void sgd_update_kernel_v2(float* weights, const float* grads, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) weights[idx] -= lr * grads[idx];
}

AutoencoderGPUOptV2::AutoencoderGPUOptV2(int batch_size) : current_buffer(0) {
    // Initialize convolution weights
    conv1 = new GPUConvWeightsOpt(256, 3, 3, 3);    // 3->256
    conv2 = new GPUConvWeightsOpt(128, 256, 3, 3);  // 256->128
    conv3 = new GPUConvWeightsOpt(128, 128, 3, 3);  // 128->128
    conv4 = new GPUConvWeightsOpt(256, 128, 3, 3);  // 128->256
    conv5 = new GPUConvWeightsOpt(3, 256, 3, 3);    // 256->3
    
    conv1->initXavier(); conv2->initXavier(); conv3->initXavier();
    conv4->initXavier(); conv5->initXavier();
    
    // Allocate memory pool (all buffers at once)
    pool.allocate(batch_size);
    
    // V2: Create CUDA streams for parallel execution
    pool.create_streams();
    
    // Create dedicated transfer stream for double buffering
    CUDA_CHECK(cudaStreamCreate(&transfer_stream));
    
    // Double buffering: allocate TWO input buffers
    input_buffer[0].allocate(batch_size, 3, 32, 32);
    input_buffer[1].allocate(batch_size, 3, 32, 32);
}

AutoencoderGPUOptV2::~AutoencoderGPUOptV2() {
    delete conv1; delete conv2; delete conv3; delete conv4; delete conv5;
    input_buffer[0].free();
    input_buffer[1].free();
    cudaStreamDestroy(transfer_stream);
    // pool destructor handles stream cleanup
}

// Async copy next batch to the alternate buffer (overlaps with computation)
void AutoencoderGPUOptV2::copy_input_async(const float* h_input, int batch_size) {
    int next_buffer = 1 - current_buffer;  // Alternate buffer
    CUDA_CHECK(cudaMemcpyAsync(input_buffer[next_buffer].d_data, h_input, 
                               batch_size * 3 * 32 * 32 * sizeof(float), 
                               cudaMemcpyHostToDevice, transfer_stream));
}

// Wait for async copy to complete and swap buffers
void AutoencoderGPUOptV2::swap_buffers() {
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream));
    current_buffer = 1 - current_buffer;  // Swap
}

float AutoencoderGPUOptV2::forward(int batch_size) {
    GPUTensorOpt& input = input_buffer[current_buffer];
    
    // Forward pass on stream1 (sequential - each layer depends on previous)
    // === ENCODER ===
    conv2d_forward_opt_v2(input, *conv1, pool.act1, true, pool.stream1);  // Conv1+ReLU
    maxpool2d_forward_opt_v2(pool.act1, pool.act2, pool.pool1_idx, pool.stream1); // Pool1
    conv2d_forward_opt_v2(pool.act2, *conv2, pool.act3, true, pool.stream1);      // Conv2+ReLU
    maxpool2d_forward_opt_v2(pool.act3, pool.act4, pool.pool2_idx, pool.stream1); // Pool2 (latent)
    
    // === DECODER ===
    conv2d_forward_opt_v2(pool.act4, *conv3, pool.act5, true, pool.stream1);      // Conv3+ReLU
    upsample2d_forward_opt_v2(pool.act5, pool.act6, pool.stream1);                 // Upsample1
    conv2d_forward_opt_v2(pool.act6, *conv4, pool.act7, true, pool.stream1);      // Conv4+ReLU
    upsample2d_forward_opt_v2(pool.act7, pool.act8, pool.stream1);                 // Upsample2
    conv2d_forward_opt_v2(pool.act8, *conv5, pool.output, false, pool.stream1);   // Conv5 (no ReLU)
    
    // Compute MSE loss
    return mse_loss_forward_opt_v2(pool.output, input, pool.stream1);
}

void AutoencoderGPUOptV2::backward(float learning_rate) {
    GPUTensorOpt& input = input_buffer[current_buffer];
    
    // Compute output gradient on stream1
    mse_loss_backward_opt_v2(pool.output, input, pool.grad_out, pool.stream1);
    
    // === DECODER BACKWARD ===
    // Conv5 backward - uses all 3 streams for parallel gradient computation
    // grad_input on stream1, grad_weights on stream2, grad_bias on stream3
    conv2d_backward_opt_v2(pool.act8, pool.grad_out, pool.output, *conv5, pool.grad8, 
                           false, pool.stream1, pool.stream2, pool.stream3);
    
    // Wait for grad_input (stream1) before upsample backward
    CUDA_CHECK(cudaStreamSynchronize(pool.stream1));
    upsample2d_backward_opt_v2(pool.grad8, pool.grad7, pool.stream1);
    
    // Conv4 backward (had ReLU)
    conv2d_backward_opt_v2(pool.act6, pool.grad7, pool.act7, *conv4, pool.grad6,
                           true, pool.stream1, pool.stream2, pool.stream3);
    
    CUDA_CHECK(cudaStreamSynchronize(pool.stream1));
    upsample2d_backward_opt_v2(pool.grad6, pool.grad5, pool.stream1);
    
    // Conv3 backward (had ReLU)
    conv2d_backward_opt_v2(pool.act4, pool.grad5, pool.act5, *conv3, pool.grad4,
                           true, pool.stream1, pool.stream2, pool.stream3);
    
    // === ENCODER BACKWARD ===
    CUDA_CHECK(cudaStreamSynchronize(pool.stream1));
    maxpool2d_backward_opt_v2(pool.grad4, pool.pool2_idx, pool.grad3, pool.stream1);
    
    // Conv2 backward (had ReLU)
    conv2d_backward_opt_v2(pool.act2, pool.grad3, pool.act3, *conv2, pool.grad2,
                           true, pool.stream1, pool.stream2, pool.stream3);
    
    CUDA_CHECK(cudaStreamSynchronize(pool.stream1));
    maxpool2d_backward_opt_v2(pool.grad2, pool.pool1_idx, pool.grad1, pool.stream1);
    
    // Conv1 backward (had ReLU)
    conv2d_backward_opt_v2(input, pool.grad1, pool.act1, *conv1, pool.grad_in,
                           true, pool.stream1, pool.stream2, pool.stream3);
    
    // Sync all streams before weight update
    conv2d_sync_streams(pool.stream1, pool.stream2, pool.stream3);
    
    // Update weights
    update_weights(learning_rate);
}

float AutoencoderGPUOptV2::train_step(float learning_rate) {
    float loss = forward(input_buffer[current_buffer].batch);
    backward(learning_rate);
    return loss;
}

void AutoencoderGPUOptV2::update_weights(float learning_rate) {
    auto update = [learning_rate, this](GPUConvWeightsOpt* w, cudaStream_t stream) {
        sgd_update_kernel_v2<<<(w->weight_size + 255) / 256, 256, 0, stream>>>(
            w->d_weights, w->d_grad_w, learning_rate, w->weight_size);
        sgd_update_kernel_v2<<<(w->bias_size + 255) / 256, 256, 0, stream>>>(
            w->d_bias, w->d_grad_b, learning_rate, w->bias_size);
    };
    
    // Update weights on stream1 (fast, sequential is fine)
    update(conv1, pool.stream1);
    update(conv2, pool.stream1);
    update(conv3, pool.stream1);
    update(conv4, pool.stream1);
    update(conv5, pool.stream1);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(pool.stream1));
}

void AutoencoderGPUOptV2::save_weights(const std::string& dir) {
    auto save_conv = [&dir](GPUConvWeightsOpt* w, const std::string& name) {
        std::vector<float> h_w(w->weight_size), h_b(w->bias_size);
        cudaMemcpy(h_w.data(), w->d_weights, w->weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b.data(), w->d_bias, w->bias_size * sizeof(float), cudaMemcpyDeviceToHost);
        std::ofstream(dir + "/" + name + "_w.bin", std::ios::binary).write((char*)h_w.data(), h_w.size() * sizeof(float));
        std::ofstream(dir + "/" + name + "_b.bin", std::ios::binary).write((char*)h_b.data(), h_b.size() * sizeof(float));
    };
    save_conv(conv1, "conv1"); save_conv(conv2, "conv2"); save_conv(conv3, "conv3");
    save_conv(conv4, "conv4"); save_conv(conv5, "conv5");
}

void AutoencoderGPUOptV2::load_weights(const std::string& dir) {
    auto load_conv = [&dir](GPUConvWeightsOpt* w, const std::string& name) {
        std::vector<float> h_w(w->weight_size), h_b(w->bias_size);
        std::ifstream(dir + "/" + name + "_w.bin", std::ios::binary).read((char*)h_w.data(), h_w.size() * sizeof(float));
        std::ifstream(dir + "/" + name + "_b.bin", std::ios::binary).read((char*)h_b.data(), h_b.size() * sizeof(float));
        cudaMemcpy(w->d_weights, h_w.data(), w->weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(w->d_bias, h_b.data(), w->bias_size * sizeof(float), cudaMemcpyHostToDevice);
    };
    load_conv(conv1, "conv1"); load_conv(conv2, "conv2"); load_conv(conv3, "conv3");
    load_conv(conv4, "conv4"); load_conv(conv5, "conv5");
}
