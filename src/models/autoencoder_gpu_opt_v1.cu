// Autoencoder GPU Optimized v1 - Implementation
#include "models/autoencoder_gpu_opt_v1.cuh"
#include <fstream>
#include <cstring>

// SGD update kernel
__global__ void sgd_update_kernel(float* weights, const float* grads, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) weights[idx] -= lr * grads[idx];
}

AutoencoderGPUOptV1::AutoencoderGPUOptV1(int batch_size) {
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
    
    // Allocate persistent input buffer
    input_buffer.allocate(batch_size, 3, 32, 32);
}

AutoencoderGPUOptV1::~AutoencoderGPUOptV1() {
    delete conv1; delete conv2; delete conv3; delete conv4; delete conv5;
    input_buffer.free();
}

float AutoencoderGPUOptV1::forward(const float* h_input, int batch_size) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(input_buffer.d_data, h_input, 
                          batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    
    // === ENCODER ===
    conv2d_forward_opt_v1(input_buffer, *conv1, pool.act1, true);  // Conv1+ReLU
    maxpool2d_forward_opt_v1(pool.act1, pool.act2, pool.pool1_idx); // Pool1
    conv2d_forward_opt_v1(pool.act2, *conv2, pool.act3, true);      // Conv2+ReLU
    maxpool2d_forward_opt_v1(pool.act3, pool.act4, pool.pool2_idx); // Pool2 (latent)
    
    // === DECODER ===
    conv2d_forward_opt_v1(pool.act4, *conv3, pool.act5, true);      // Conv3+ReLU
    upsample2d_forward_opt_v1(pool.act5, pool.act6);                 // Upsample1
    conv2d_forward_opt_v1(pool.act6, *conv4, pool.act7, true);      // Conv4+ReLU
    upsample2d_forward_opt_v1(pool.act7, pool.act8);                 // Upsample2
    conv2d_forward_opt_v1(pool.act8, *conv5, pool.output, false);   // Conv5 (no ReLU)
    
    // Compute MSE loss
    return mse_loss_forward_opt_v1(pool.output, input_buffer);
}

void AutoencoderGPUOptV1::backward(float learning_rate) {
    // Compute output gradient
    mse_loss_backward_opt_v1(pool.output, input_buffer, pool.grad_out);
    
    // === DECODER BACKWARD ===
    conv2d_backward_opt_v1(pool.act8, pool.grad_out, *conv5, pool.grad8);
    upsample2d_backward_opt_v1(pool.grad8, pool.grad7);
    conv2d_backward_opt_v1(pool.act6, pool.grad7, *conv4, pool.grad6);
    upsample2d_backward_opt_v1(pool.grad6, pool.grad5);
    conv2d_backward_opt_v1(pool.act4, pool.grad5, *conv3, pool.grad4);
    
    // === ENCODER BACKWARD ===
    maxpool2d_backward_opt_v1(pool.grad4, pool.pool2_idx, pool.grad3);
    conv2d_backward_opt_v1(pool.act2, pool.grad3, *conv2, pool.grad2);
    maxpool2d_backward_opt_v1(pool.grad2, pool.pool1_idx, pool.grad1);
    conv2d_backward_opt_v1(input_buffer, pool.grad1, *conv1, pool.grad_in);
    
    // Update weights
    update_weights(learning_rate);
}

float AutoencoderGPUOptV1::train_step(const float* h_input, int batch_size, float learning_rate) {
    float loss = forward(h_input, batch_size);
    backward(learning_rate);
    return loss;
}

void AutoencoderGPUOptV1::update_weights(float learning_rate) {
    auto update = [learning_rate](GPUConvWeightsOpt* w) {
        sgd_update_kernel<<<(w->weight_size + 255) / 256, 256>>>(w->d_weights, w->d_grad_w, learning_rate, w->weight_size);
        sgd_update_kernel<<<(w->bias_size + 255) / 256, 256>>>(w->d_bias, w->d_grad_b, learning_rate, w->bias_size);
    };
    update(conv1); update(conv2); update(conv3); update(conv4); update(conv5);
    CUDA_CHECK(cudaGetLastError());
}

void AutoencoderGPUOptV1::save_weights(const std::string& dir) {
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

void AutoencoderGPUOptV1::load_weights(const std::string& dir) {
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
