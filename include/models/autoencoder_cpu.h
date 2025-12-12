#pragma once
#include "layers/conv2d_cpu.h"
#include "layers/relu_cpu.h"
#include "layers/maxpool_cpu.h"
#include "layers/upsample_cpu.h"
#include <memory>
#include <string>

/**
 * Convolutional Autoencoder (CPU)
 *
 * Architecture:
 *   ENCODER:
 *     Input  [batch, 3, 32, 32]
 *     Conv1  [batch, 256, 32, 32]  (3→256, k=3, s=1, p=1)
 *     ReLU1
 *     Pool1  [batch, 256, 16, 16]  (2x2 max pool)
 *     Conv2  [batch, 128, 16, 16]  (256→128, k=3, s=1, p=1)
 *     ReLU2
 *     Pool2  [batch, 128, 8, 8]    (2x2 max pool) = LATENT (8192 features)
 *
 *   DECODER:
 *     Conv3  [batch, 128, 8, 8]    (128→128, k=3, s=1, p=1)
 *     ReLU3
 *     Up1    [batch, 128, 16, 16]  (2x upsample)
 *     Conv4  [batch, 256, 16, 16]  (128→256, k=3, s=1, p=1)
 *     ReLU4
 *     Up2    [batch, 256, 32, 32]  (2x upsample)
 *     Conv5  [batch, 3, 32, 32]    (256→3, k=3, s=1, p=1) = OUTPUT
 *
 * Memory Layout: NCHW (batch, channels, height, width)
 * All data normalized to [0, 1]
 *
 * Performance Optimization:
 *   Layers write directly to pre-allocated activation buffers (passed as output
 *   parameters) to eliminate memcpy operations. This removes ~17 copy operations
 *   per forward pass, saving ~150MB of memory bandwidth per batch of 32 images.
 */
class AutoencoderCPU
{
private:
    // ========================
    // ENCODER LAYERS
    // ========================
    std::unique_ptr<Conv2DCPU> conv1_; // 3 → 256, k=3, p=1
    std::unique_ptr<ReLUCPU> relu1_;
    std::unique_ptr<MaxPoolCPU> pool1_; // 2x2

    std::unique_ptr<Conv2DCPU> conv2_; // 256 → 128, k=3, p=1
    std::unique_ptr<ReLUCPU> relu2_;
    std::unique_ptr<MaxPoolCPU> pool2_; // 2x2

    // ========================
    // DECODER LAYERS
    // ========================
    std::unique_ptr<Conv2DCPU> conv3_; // 128 → 128, k=3, p=1
    std::unique_ptr<ReLUCPU> relu3_;
    std::unique_ptr<UpsampleCPU> up1_; // 2x

    std::unique_ptr<Conv2DCPU> conv4_; // 128 → 256, k=3, p=1
    std::unique_ptr<ReLUCPU> relu4_;
    std::unique_ptr<UpsampleCPU> up2_; // 2x

    std::unique_ptr<Conv2DCPU> conv5_; // 256 → 3, k=3, p=1

    // ========================
    // INTERMEDIATE ACTIVATIONS
    // ========================
    // These store outputs of each layer for backward pass
    float *act_conv1_ = nullptr; // [batch, 256, 32, 32]
    float *act_relu1_ = nullptr; // [batch, 256, 32, 32]
    float *act_pool1_ = nullptr; // [batch, 256, 16, 16]
    float *act_conv2_ = nullptr; // [batch, 128, 16, 16]
    float *act_relu2_ = nullptr; // [batch, 128, 16, 16]
    float *act_pool2_ = nullptr; // [batch, 128, 8, 8] = LATENT

    float *act_conv3_ = nullptr; // [batch, 128, 8, 8]
    float *act_relu3_ = nullptr; // [batch, 128, 8, 8]
    float *act_up1_ = nullptr;   // [batch, 128, 16, 16]
    float *act_conv4_ = nullptr; // [batch, 256, 16, 16]
    float *act_relu4_ = nullptr; // [batch, 256, 16, 16]
    float *act_up2_ = nullptr;   // [batch, 256, 32, 32]
    float *act_conv5_ = nullptr; // [batch, 3, 32, 32] = OUTPUT

    // Gradient buffer for loss computation
    float *grad_buffer_ = nullptr;

    // Cached batch size for backward pass
    int cached_batch_ = 0;

    // Activation buffer sizes (for reallocation check)
    size_t act_buffer_size_ = 0;

    // Allocate activation buffers for given batch size
    void allocate_activations(int batch);
    void free_activations();

public:
    AutoencoderCPU();
    ~AutoencoderCPU();

    // Disable copy
    AutoencoderCPU(const AutoencoderCPU &) = delete;
    AutoencoderCPU &operator=(const AutoencoderCPU &) = delete;

    /**
     * Forward pass - full autoencoder (encoder + decoder)
     * @param input     Input images [batch, 3, 32, 32]
     * @param batch     Batch size
     * @return          Reconstructed images [batch, 3, 32, 32]
     */
    float *forward(const float *input, int batch);

    /**
     * Extract latent features - encoder only
     * @param input     Input images [batch, 3, 32, 32]
     * @param batch     Batch size
     * @return          Latent representation [batch, 128, 8, 8] = 8192 features per image
     */
    float *extract_features(const float *input, int batch);

    /**
     * Backward pass + weight update (SGD)
     * @param target        Target images (same as input for autoencoder)
     * @param learning_rate Learning rate for SGD
     */
    void backward(const float *target, float learning_rate);

    /**
     * Compute MSE loss between output and target
     * @param output    Reconstructed images
     * @param target    Target images
     * @param size      Total number of elements
     * @return          Mean squared error
     */
    static float compute_loss(const float *output, const float *target, size_t size);

    /**
     * Save model weights to binary file
     * @param filename  Path to save file
     */
    void save_weights(const std::string &filename);

    /**
     * Load model weights from binary file
     * @param filename  Path to load file
     */
    void load_weights(const std::string &filename);

    // Getters for dimensions
    int get_latent_size() const { return 128 * 8 * 8; } // 8192 features per image
    int get_output_size() const { return 3 * 32 * 32; } // 3072 pixels per image
};
