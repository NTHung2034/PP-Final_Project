#include "models/autoencoder_gpu_opt_v2.cuh"
#include <fstream>

// SGD update kernel
__global__ void sgd_update_kernel_v2(float* weights, const float* grads, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) weights[idx] -= lr * grads[idx];
}

AutoencoderGPUOptV2::AutoencoderGPUOptV2(int batch_size) {
    conv1 = new GPUConvWeightsOpt(256, 3, 3, 3);
    conv2 = new GPUConvWeightsOpt(128, 256, 3, 3);
    conv3 = new GPUConvWeightsOpt(128, 128, 3, 3);
    conv4 = new GPUConvWeightsOpt(256, 128, 3, 3);
    conv5 = new GPUConvWeightsOpt(3, 256, 3, 3);
    
    conv1->initXavier(); conv2->initXavier(); conv3->initXavier();
    conv4->initXavier(); conv5->initXavier();
    
    pool.allocate(batch_size);
    input_buffer.allocate(batch_size, 3, 32, 32);
}

AutoencoderGPUOptV2::~AutoencoderGPUOptV2() {
    delete conv1; delete conv2; delete conv3; delete conv4; delete conv5;
    input_buffer.free();
}

void AutoencoderGPUOptV2::async_load_input(const float* h_input, int batch_size) {
    pool.async_input_transfer(input_buffer.d_data, h_input, batch_size * 3 * 32 * 32 * sizeof(float));
}

void AutoencoderGPUOptV2::forward_stream() {
    cudaStream_t s = pool.compute_stream;
    pool.sync_before_compute();
    
    // ENCODER (fused conv+relu+bias)
    conv2d_forward_opt_v2(input_buffer, *conv1, pool.act1, true, s);
    maxpool2d_forward_opt_v2(pool.act1, pool.act2, pool.pool1_idx, s);
    conv2d_forward_opt_v2(pool.act2, *conv2, pool.act3, true, s);
    maxpool2d_forward_opt_v2(pool.act3, pool.act4, pool.pool2_idx, s);
    
    // DECODER
    conv2d_forward_opt_v2(pool.act4, *conv3, pool.act5, true, s);
    upsample2d_forward_opt_v2(pool.act5, pool.act6, s);
    conv2d_forward_opt_v2(pool.act6, *conv4, pool.act7, true, s);
    upsample2d_forward_opt_v2(pool.act7, pool.act8, s);
    conv2d_forward_opt_v2(pool.act8, *conv5, pool.output, false, s);
    
    // Accumulate loss (no sync)
    mse_loss_forward_opt_v2(pool.output, input_buffer, pool.d_loss, s);
}

void AutoencoderGPUOptV2::backward_stream(float learning_rate) {
    cudaStream_t s = pool.compute_stream;
    
    mse_loss_backward_opt_v2(pool.output, input_buffer, pool.grad_out, s);
    
    // DECODER BACKWARD (fused with ReLU gradient)
    conv2d_backward_opt_v2(pool.act8, pool.grad_out, *conv5, pool.grad8, nullptr, s);
    upsample2d_backward_opt_v2(pool.grad8, pool.grad7, s);
    conv2d_backward_opt_v2(pool.act6, pool.grad7, *conv4, pool.grad6, &pool.act7, s);
    upsample2d_backward_opt_v2(pool.grad6, pool.grad5, s);
    conv2d_backward_opt_v2(pool.act4, pool.grad5, *conv3, pool.grad4, &pool.act5, s);
    
    // ENCODER BACKWARD
    maxpool2d_backward_opt_v2(pool.grad4, pool.pool2_idx, pool.grad3, s);
    conv2d_backward_opt_v2(pool.act2, pool.grad3, *conv2, pool.grad2, &pool.act3, s);
    maxpool2d_backward_opt_v2(pool.grad2, pool.pool1_idx, pool.grad1, s);
    conv2d_backward_opt_v2(input_buffer, pool.grad1, *conv1, pool.grad_in, &pool.act1, s);
    
    update_weights(learning_rate);
}

float AutoencoderGPUOptV2::forward(const float* h_input, int batch_size) {
    CUDA_CHECK(cudaMemcpy(input_buffer.d_data, h_input, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    
    conv2d_forward_opt_v2(input_buffer, *conv1, pool.act1, true);
    maxpool2d_forward_opt_v2(pool.act1, pool.act2, pool.pool1_idx);
    conv2d_forward_opt_v2(pool.act2, *conv2, pool.act3, true);
    maxpool2d_forward_opt_v2(pool.act3, pool.act4, pool.pool2_idx);
    conv2d_forward_opt_v2(pool.act4, *conv3, pool.act5, true);
    upsample2d_forward_opt_v2(pool.act5, pool.act6);
    conv2d_forward_opt_v2(pool.act6, *conv4, pool.act7, true);
    upsample2d_forward_opt_v2(pool.act7, pool.act8);
    conv2d_forward_opt_v2(pool.act8, *conv5, pool.output, false);
    
    // For non-stream version, sync and get loss immediately
    pool.reset_loss();
    mse_loss_forward_opt_v2(pool.output, input_buffer, pool.d_loss);
    return pool.get_loss(1);
}

void AutoencoderGPUOptV2::backward(float learning_rate) {
    mse_loss_backward_opt_v2(pool.output, input_buffer, pool.grad_out);
    
    conv2d_backward_opt_v2(pool.act8, pool.grad_out, *conv5, pool.grad8, nullptr);
    upsample2d_backward_opt_v2(pool.grad8, pool.grad7);
    conv2d_backward_opt_v2(pool.act6, pool.grad7, *conv4, pool.grad6, &pool.act7);
    upsample2d_backward_opt_v2(pool.grad6, pool.grad5);
    conv2d_backward_opt_v2(pool.act4, pool.grad5, *conv3, pool.grad4, &pool.act5);
    maxpool2d_backward_opt_v2(pool.grad4, pool.pool2_idx, pool.grad3);
    conv2d_backward_opt_v2(pool.act2, pool.grad3, *conv2, pool.grad2, &pool.act3);
    maxpool2d_backward_opt_v2(pool.grad2, pool.pool1_idx, pool.grad1);
    conv2d_backward_opt_v2(input_buffer, pool.grad1, *conv1, pool.grad_in, &pool.act1);
    
    update_weights(learning_rate);
}

float AutoencoderGPUOptV2::train_step(const float* h_input, int batch_size, float learning_rate) {
    float loss = forward(h_input, batch_size);
    backward(learning_rate);
    return loss;
}

void AutoencoderGPUOptV2::update_weights(float learning_rate) {
    cudaStream_t s = pool.compute_stream;
    auto update = [learning_rate, s](GPUConvWeightsOpt* w) {
        sgd_update_kernel_v2<<<(w->weight_size + 255) / 256, 256, 0, s>>>(w->d_weights, w->d_grad_w, learning_rate, w->weight_size);
        sgd_update_kernel_v2<<<(w->bias_size + 255) / 256, 256, 0, s>>>(w->d_bias, w->d_grad_b, learning_rate, w->bias_size);
    };
    update(conv1); update(conv2); update(conv3); update(conv4); update(conv5);
    CUDA_CHECK(cudaGetLastError());
}

void AutoencoderGPUOptV2::save_weights(const std::string& dir) {
    auto save = [&dir](GPUConvWeightsOpt* w, const std::string& name) {
        std::vector<float> hw(w->weight_size), hb(w->bias_size);
        cudaMemcpy(hw.data(), w->d_weights, w->weight_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.data(), w->d_bias, w->bias_size * sizeof(float), cudaMemcpyDeviceToHost);
        std::ofstream(dir + "/" + name + "_w.bin", std::ios::binary).write((char*)hw.data(), hw.size() * sizeof(float));
        std::ofstream(dir + "/" + name + "_b.bin", std::ios::binary).write((char*)hb.data(), hb.size() * sizeof(float));
    };
    save(conv1, "conv1"); save(conv2, "conv2"); save(conv3, "conv3"); save(conv4, "conv4"); save(conv5, "conv5");
}

void AutoencoderGPUOptV2::load_weights(const std::string& dir) {
    auto load = [&dir](GPUConvWeightsOpt* w, const std::string& name) {
        std::vector<float> hw(w->weight_size), hb(w->bias_size);
        std::ifstream(dir + "/" + name + "_w.bin", std::ios::binary).read((char*)hw.data(), hw.size() * sizeof(float));
        std::ifstream(dir + "/" + name + "_b.bin", std::ios::binary).read((char*)hb.data(), hb.size() * sizeof(float));
        cudaMemcpy(w->d_weights, hw.data(), w->weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(w->d_bias, hb.data(), w->bias_size * sizeof(float), cudaMemcpyHostToDevice);
    };
    load(conv1, "conv1"); load(conv2, "conv2"); load(conv3, "conv3"); load(conv4, "conv4"); load(conv5, "conv5");
}
