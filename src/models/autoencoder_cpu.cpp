/**
 * @file autoencoder_cpu.cpp
 * @brief CPU implementation of the Convolutional Autoencoder
 * 
 * This file implements the complete autoencoder with:
 * - Forward pass through encoder and decoder
 * - Feature extraction using encoder only
 * - Backward pass with gradient computation
 * - Weight save/load functionality
 * 
 * Architecture follows the project specification with ~751,875 parameters.
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#include "models/autoencoder_cpu.h"
#include "utils/logger.h"
#include <fstream>
#include <stdexcept>
#include <cstring>

AutoencoderCPU::AutoencoderCPU()
    : latent_({1, 128, 8, 8})  // Placeholder, will be replaced
    , output_({1, 3, 32, 32})   // Placeholder, will be replaced
{
    LOG_INFO("Initializing AutoencoderCPU...");
    
    // =========================================
    // ENCODER LAYERS
    // =========================================
    
    // Block 1: Input (32,32,3) → (16,16,256)
    // Conv: 3 input channels → 256 output channels, 3x3 kernel, padding=1
    enc_conv1_ = std::make_unique<Conv2DCPU>(3, 256, 3, 1, 1);
    enc_relu1_ = std::make_unique<ReLUCPU>();
    enc_pool1_ = std::make_unique<MaxPoolCPU>(2, 2);
    
    // Block 2: (16,16,256) → (8,8,128)
    // Conv: 256 → 128, 3x3 kernel, padding=1
    enc_conv2_ = std::make_unique<Conv2DCPU>(256, 128, 3, 1, 1);
    enc_relu2_ = std::make_unique<ReLUCPU>();
    enc_pool2_ = std::make_unique<MaxPoolCPU>(2, 2);
    
    // =========================================
    // DECODER LAYERS
    // =========================================
    
    // Block 3: (8,8,128) → (16,16,128)
    // Conv: 128 → 128, 3x3 kernel, padding=1
    dec_conv1_ = std::make_unique<Conv2DCPU>(128, 128, 3, 1, 1);
    dec_relu1_ = std::make_unique<ReLUCPU>();
    dec_up1_ = std::make_unique<UpsampleCPU>(2);
    
    // Block 4: (16,16,128) → (32,32,256)
    // Conv: 128 → 256, 3x3 kernel, padding=1
    dec_conv2_ = std::make_unique<Conv2DCPU>(128, 256, 3, 1, 1);
    dec_relu2_ = std::make_unique<ReLUCPU>();
    dec_up2_ = std::make_unique<UpsampleCPU>(2);
    
    // Block 5: (32,32,256) → (32,32,3)
    // Conv: 256 → 3, 3x3 kernel, padding=1 (no activation - linear output)
    dec_conv3_ = std::make_unique<Conv2DCPU>(256, 3, 3, 1, 1);
    
    LOG_INFO("AutoencoderCPU initialized with %zu parameters", get_num_parameters());
}

Tensor AutoencoderCPU::forward(const Tensor& input) {
    /**
     * Forward pass through the complete autoencoder:
     * 
     * ENCODER:
     *   Input (32,32,3) → Conv1 → ReLU → Pool → (16,16,256)
     *                   → Conv2 → ReLU → Pool → (8,8,128) [LATENT]
     * 
     * DECODER:
     *   Latent (8,8,128) → Conv3 → ReLU → Upsample → (16,16,128)
     *                    → Conv4 → ReLU → Upsample → (32,32,256)
     *                    → Conv5 → (32,32,3) [OUTPUT]
     */
    
    // =========================================
    // ENCODER PATH
    // =========================================
    
    // Block 1: (32,32,3) → (16,16,256)
    auto x = enc_conv1_->forward(input);     // (32,32,3) → (32,32,256)
    x = enc_relu1_->forward(x);               // ReLU activation
    x = enc_pool1_->forward(x);               // (32,32,256) → (16,16,256)
    
    // Block 2: (16,16,256) → (8,8,128)
    x = enc_conv2_->forward(x);              // (16,16,256) → (16,16,128)
    x = enc_relu2_->forward(x);               // ReLU activation
    latent_ = enc_pool2_->forward(x);         // (16,16,128) → (8,8,128)
    
    // =========================================
    // DECODER PATH
    // =========================================
    
    // Block 3: (8,8,128) → (16,16,128)
    x = dec_conv1_->forward(latent_);        // (8,8,128) → (8,8,128)
    x = dec_relu1_->forward(x);               // ReLU activation
    x = dec_up1_->forward(x);                 // (8,8,128) → (16,16,128)
    
    // Block 4: (16,16,128) → (32,32,256)
    x = dec_conv2_->forward(x);              // (16,16,128) → (16,16,256)
    x = dec_relu2_->forward(x);               // ReLU activation
    x = dec_up2_->forward(x);                 // (16,16,256) → (32,32,256)
    
    // Block 5: (32,32,256) → (32,32,3)
    output_ = dec_conv3_->forward(x);        // Final reconstruction (no activation)
    
    return output_;
}

Tensor AutoencoderCPU::extract_features(const Tensor& input) {
    /**
     * Extract features using only the encoder.
     * 
     * This is used after training to get the 8,192-dimensional
     * feature vector for each input image.
     * 
     * The output shape is [N, 128, 8, 8] which can be flattened
     * to [N, 8192] for SVM training.
     */
    
    // Block 1: (32,32,3) → (16,16,256)
    auto x = enc_conv1_->forward(input);
    x = enc_relu1_->forward(x);
    x = enc_pool1_->forward(x);
    
    // Block 2: (16,16,256) → (8,8,128)
    x = enc_conv2_->forward(x);
    x = enc_relu2_->forward(x);
    x = enc_pool2_->forward(x);
    
    return x;  // Latent representation [N, 128, 8, 8]
}

float AutoencoderCPU::backward(const Tensor& target, float learning_rate) {
    /**
     * Backward pass through the autoencoder:
     * 
     * 1. Compute loss gradient at output
     * 2. Backpropagate through decoder
     * 3. Backpropagate through encoder
     * 4. Update all weights
     * 
     * Returns the loss value for monitoring.
     */
    
    // Compute loss and gradient
    Tensor grad;
    float loss = LossFunctions::mse_loss_with_gradient(output_, target, grad);
    
    // =========================================
    // DECODER BACKWARD
    // =========================================
    
    // Block 5: (32,32,3) backward
    grad = dec_conv3_->backward(grad);
    
    // Block 4: (32,32,256) → (16,16,128) backward
    grad = dec_up2_->backward(grad);
    grad = dec_relu2_->backward(grad);
    grad = dec_conv2_->backward(grad);
    
    // Block 3: (16,16,128) → (8,8,128) backward
    grad = dec_up1_->backward(grad);
    grad = dec_relu1_->backward(grad);
    grad = dec_conv1_->backward(grad);
    
    // =========================================
    // ENCODER BACKWARD
    // =========================================
    
    // Block 2: (8,8,128) → (16,16,256) backward
    grad = enc_pool2_->backward(grad);
    grad = enc_relu2_->backward(grad);
    grad = enc_conv2_->backward(grad);
    
    // Block 1: (16,16,256) → (32,32,3) backward
    grad = enc_pool1_->backward(grad);
    grad = enc_relu1_->backward(grad);
    grad = enc_conv1_->backward(grad);
    
    // =========================================
    // WEIGHT UPDATES
    // =========================================
    
    // Update encoder weights
    enc_conv1_->update_weights(learning_rate);
    enc_conv2_->update_weights(learning_rate);
    
    // Update decoder weights
    dec_conv1_->update_weights(learning_rate);
    dec_conv2_->update_weights(learning_rate);
    dec_conv3_->update_weights(learning_rate);
    
    return loss;
}

float AutoencoderCPU::compute_loss(const Tensor& output, const Tensor& target) {
    return LossFunctions::mse_loss(output, target);
}

size_t AutoencoderCPU::get_num_parameters() const {
    /**
     * Calculate total number of trainable parameters:
     * 
     * Encoder:
     *   Conv1: 3 * 256 * 3 * 3 + 256 = 6,912 + 256 = 7,168
     *   Conv2: 256 * 128 * 3 * 3 + 128 = 294,912 + 128 = 295,040
     * 
     * Decoder:
     *   Conv3: 128 * 128 * 3 * 3 + 128 = 147,456 + 128 = 147,584
     *   Conv4: 128 * 256 * 3 * 3 + 256 = 294,912 + 256 = 295,168
     *   Conv5: 256 * 3 * 3 * 3 + 3 = 6,912 + 3 = 6,915
     * 
     * Total: 7,168 + 295,040 + 147,584 + 295,168 + 6,915 = 751,875
     */
    
    size_t total = 0;
    
    // Encoder
    total += enc_conv1_->get_weights().size() + enc_conv1_->get_bias().size();
    total += enc_conv2_->get_weights().size() + enc_conv2_->get_bias().size();
    
    // Decoder
    total += dec_conv1_->get_weights().size() + dec_conv1_->get_bias().size();
    total += dec_conv2_->get_weights().size() + dec_conv2_->get_bias().size();
    total += dec_conv3_->get_weights().size() + dec_conv3_->get_bias().size();
    
    return total;
}

void AutoencoderCPU::save_weights(const std::string& filepath) {
    /**
     * Save model weights to binary file.
     * 
     * Format:
     *   - Magic number (4 bytes)
     *   - Number of layers (4 bytes)
     *   - For each layer:
     *     - Weight size (4 bytes)
     *     - Weights (weight_size * 4 bytes)
     *     - Bias size (4 bytes)
     *     - Biases (bias_size * 4 bytes)
     */
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    // Magic number for file validation
    const int magic = 0x41455743;  // "CWEA" (CUDA Weights for Encoder/Autoencoder)
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    
    // Number of convolutional layers
    const int num_layers = 5;
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    // Helper lambda to write layer weights
    auto write_layer = [&file](const Conv2DCPU* layer) {
        const auto& weights = layer->get_weights();
        const auto& bias = layer->get_bias();
        
        int w_size = weights.size();
        int b_size = bias.size();
        
        file.write(reinterpret_cast<const char*>(&w_size), sizeof(w_size));
        file.write(reinterpret_cast<const char*>(weights.data()), w_size * sizeof(float));
        
        file.write(reinterpret_cast<const char*>(&b_size), sizeof(b_size));
        file.write(reinterpret_cast<const char*>(bias.data()), b_size * sizeof(float));
    };
    
    // Write all layer weights
    write_layer(enc_conv1_.get());
    write_layer(enc_conv2_.get());
    write_layer(dec_conv1_.get());
    write_layer(dec_conv2_.get());
    write_layer(dec_conv3_.get());
    
    file.close();
    LOG_INFO("Model weights saved to %s", filepath.c_str());
}

void AutoencoderCPU::load_weights(const std::string& filepath) {
    /**
     * Load model weights from binary file.
     */
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    
    // Verify magic number
    int magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x41455743) {
        throw std::runtime_error("Invalid weight file format");
    }
    
    // Read number of layers
    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    if (num_layers != 5) {
        throw std::runtime_error("Layer count mismatch in weight file");
    }
    
    // Helper lambda to read layer weights
    auto read_layer = [&file](Conv2DCPU* layer) {
        int w_size, b_size;
        
        file.read(reinterpret_cast<char*>(&w_size), sizeof(w_size));
        std::vector<float> weights(w_size);
        file.read(reinterpret_cast<char*>(weights.data()), w_size * sizeof(float));
        
        file.read(reinterpret_cast<char*>(&b_size), sizeof(b_size));
        std::vector<float> bias(b_size);
        file.read(reinterpret_cast<char*>(bias.data()), b_size * sizeof(float));
        
        layer->set_weights(weights, bias);
    };
    
    // Read all layer weights
    read_layer(enc_conv1_.get());
    read_layer(enc_conv2_.get());
    read_layer(dec_conv1_.get());
    read_layer(dec_conv2_.get());
    read_layer(dec_conv3_.get());
    
    file.close();
    LOG_INFO("Model weights loaded from %s", filepath.c_str());
}
