/**
 * @file autoencoder_cpu.h
 * @brief CPU implementation of the Convolutional Autoencoder
 * 
 * This file implements the complete autoencoder architecture as specified
 * in the project requirements:
 * 
 * ENCODER (Downsampling Path):
 *   Input: (32, 32, 3)
 *   Conv2D(256, 3×3, pad=1) + ReLU → (32, 32, 256)
 *   MaxPool(2×2)                   → (16, 16, 256)
 *   Conv2D(128, 3×3, pad=1) + ReLU → (16, 16, 128)
 *   MaxPool(2×2)                   → (8, 8, 128)
 *   LATENT: 8×8×128 = 8,192 dimensions
 * 
 * DECODER (Upsampling Path - Mirror of Encoder):
 *   Latent: (8, 8, 128)
 *   Conv2D(128, 3×3, pad=1) + ReLU → (8, 8, 128)
 *   UpSample(2×2)                  → (16, 16, 128)
 *   Conv2D(256, 3×3, pad=1) + ReLU → (16, 16, 256)
 *   UpSample(2×2)                  → (32, 32, 256)
 *   Conv2D(3, 3×3, pad=1)          → (32, 32, 3)
 *   Output: (32, 32, 3)
 * 
 * Total Parameters: ~751,875 (as per Keras reference)
 * 
 * Reference:
 * - Project requirements: docs/CSC14120_2025_Final_Project.md
 * - turkdogan/autoencoder: https://github.com/turkdogan/autoencoder
 * - Hinton & Salakhutdinov (2006): https://www.cs.toronto.edu/~hinton/science.pdf
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#pragma once

#include "layers/conv2d_cpu.h"
#include "layers/relu_cpu.h"
#include "layers/maxpool_cpu.h"
#include "layers/upsample_cpu.h"
#include "layers/loss_functions.h"
#include "data/data_types.h"
#include <memory>
#include <string>

/**
 * @class AutoencoderCPU
 * @brief Complete autoencoder model with encoder and decoder
 * 
 * The autoencoder is trained to reconstruct input images. During training,
 * it learns a compressed representation (latent space) that captures
 * the essential features of the input.
 * 
 * After training, the encoder can be used to extract features for
 * downstream tasks like classification.
 */
class AutoencoderCPU {
public:
    /**
     * @brief Construct the autoencoder with default architecture
     */
    AutoencoderCPU();
    
    /**
     * @brief Destructor
     */
    ~AutoencoderCPU() = default;
    
    // Disable copy (layers have non-copyable members)
    AutoencoderCPU(const AutoencoderCPU&) = delete;
    AutoencoderCPU& operator=(const AutoencoderCPU&) = delete;
    
    // Enable move
    AutoencoderCPU(AutoencoderCPU&&) = default;
    AutoencoderCPU& operator=(AutoencoderCPU&&) = default;
    
    /**
     * @brief Forward pass through the entire autoencoder (encoder + decoder)
     * 
     * @param input Input images [N, 3, 32, 32]
     * @return Tensor Reconstructed images [N, 3, 32, 32]
     */
    Tensor forward(const Tensor& input);
    
    /**
     * @brief Extract features using only the encoder
     * 
     * This is used after training to get the latent representation
     * for SVM classification.
     * 
     * @param input Input images [N, 3, 32, 32]
     * @return Tensor Latent features [N, 128, 8, 8] (can be flattened to [N, 8192])
     */
    Tensor extract_features(const Tensor& input);
    
    /**
     * @brief Backward pass and weight update
     * 
     * Computes gradients for all layers and updates weights using SGD.
     * 
     * @param target Original input images (for computing reconstruction loss)
     * @param learning_rate Learning rate for SGD
     * @return float The loss value
     */
    float backward(const Tensor& target, float learning_rate);
    
    /**
     * @brief Compute reconstruction loss (without backward pass)
     * 
     * @param output Reconstructed images
     * @param target Original images
     * @return float MSE loss
     */
    float compute_loss(const Tensor& output, const Tensor& target);
    
    /**
     * @brief Save model weights to file
     * 
     * @param filepath Path to save weights
     */
    void save_weights(const std::string& filepath);
    
    /**
     * @brief Load model weights from file
     * 
     * @param filepath Path to load weights from
     */
    void load_weights(const std::string& filepath);
    
    /**
     * @brief Get the latent dimension (8192 for this architecture)
     */
    int get_latent_dim() const { return 8 * 8 * 128; }
    
    /**
     * @brief Get total number of parameters
     */
    size_t get_num_parameters() const;
    
private:
    // =========================================
    // ENCODER LAYERS
    // =========================================
    
    // Block 1: Input → (32,32,256)
    std::unique_ptr<Conv2DCPU> enc_conv1_;   // 3 → 256, 3x3, pad=1
    std::unique_ptr<ReLUCPU> enc_relu1_;
    std::unique_ptr<MaxPoolCPU> enc_pool1_;   // → (16,16,256)
    
    // Block 2: (16,16,256) → (8,8,128)
    std::unique_ptr<Conv2DCPU> enc_conv2_;   // 256 → 128, 3x3, pad=1
    std::unique_ptr<ReLUCPU> enc_relu2_;
    std::unique_ptr<MaxPoolCPU> enc_pool2_;   // → (8,8,128) = LATENT
    
    // =========================================
    // DECODER LAYERS
    // =========================================
    
    // Block 3: (8,8,128) → (16,16,128)
    std::unique_ptr<Conv2DCPU> dec_conv1_;   // 128 → 128, 3x3, pad=1
    std::unique_ptr<ReLUCPU> dec_relu1_;
    std::unique_ptr<UpsampleCPU> dec_up1_;    // → (16,16,128)
    
    // Block 4: (16,16,128) → (32,32,256)
    std::unique_ptr<Conv2DCPU> dec_conv2_;   // 128 → 256, 3x3, pad=1
    std::unique_ptr<ReLUCPU> dec_relu2_;
    std::unique_ptr<UpsampleCPU> dec_up2_;    // → (32,32,256)
    
    // Block 5: (32,32,256) → (32,32,3)
    std::unique_ptr<Conv2DCPU> dec_conv3_;   // 256 → 3, 3x3, pad=1 (no activation)
    
    // =========================================
    // CACHED TENSORS
    // =========================================
    
    Tensor latent_;   // Cached latent representation [N, 128, 8, 8]
    Tensor output_;   // Cached output for backward pass
};
