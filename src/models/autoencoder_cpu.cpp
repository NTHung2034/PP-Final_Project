#include "models/autoencoder_cpu.h"
#include <fstream>
#include <cmath>
#include <cstring>
#include <stdexcept>

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

AutoencoderCPU::AutoencoderCPU()
{
    // ========================
    // Initialize ENCODER layers
    // ========================
    // Conv1: 3 → 256, kernel=3, stride=1, padding=1
    // Input: [batch, 3, 32, 32] → Output: [batch, 256, 32, 32]
    conv1_ = std::make_unique<Conv2DCPU>(3, 256, 3, 1, 1);
    relu1_ = std::make_unique<ReLUCPU>();
    pool1_ = std::make_unique<MaxPoolCPU>(2); // → [batch, 256, 16, 16]

    // Conv2: 256 → 128, kernel=3, stride=1, padding=1
    // Input: [batch, 256, 16, 16] → Output: [batch, 128, 16, 16]
    conv2_ = std::make_unique<Conv2DCPU>(256, 128, 3, 1, 1);
    relu2_ = std::make_unique<ReLUCPU>();
    pool2_ = std::make_unique<MaxPoolCPU>(2); // → [batch, 128, 8, 8] = LATENT

    // ========================
    // Initialize DECODER layers
    // ========================
    // Conv3: 128 → 128, kernel=3, stride=1, padding=1
    conv3_ = std::make_unique<Conv2DCPU>(128, 128, 3, 1, 1);
    relu3_ = std::make_unique<ReLUCPU>();
    up1_ = std::make_unique<UpsampleCPU>(2); // → [batch, 128, 16, 16]

    // Conv4: 128 → 256, kernel=3, stride=1, padding=1
    conv4_ = std::make_unique<Conv2DCPU>(128, 256, 3, 1, 1);
    relu4_ = std::make_unique<ReLUCPU>();
    up2_ = std::make_unique<UpsampleCPU>(2); // → [batch, 256, 32, 32]

    // Conv5: 256 → 3, kernel=3, stride=1, padding=1 (output layer)
    conv5_ = std::make_unique<Conv2DCPU>(256, 3, 3, 1, 1); // → [batch, 3, 32, 32]
}

AutoencoderCPU::~AutoencoderCPU()
{
    free_activations();
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

void AutoencoderCPU::allocate_activations(int batch)
{
    if (batch == cached_batch_ && act_buffer_size_ > 0)
    {
        return; // Already allocated for this batch size
    }

    free_activations();
    cached_batch_ = batch;

    // Encoder activation sizes
    size_t size_conv1 = batch * 256 * 32 * 32; // After conv1
    size_t size_pool1 = batch * 256 * 16 * 16; // After pool1
    size_t size_conv2 = batch * 128 * 16 * 16; // After conv2
    size_t size_pool2 = batch * 128 * 8 * 8;   // After pool2 (latent)

    // Decoder activation sizes
    size_t size_conv3 = batch * 128 * 8 * 8;   // After conv3
    size_t size_up1 = batch * 128 * 16 * 16;   // After up1
    size_t size_conv4 = batch * 256 * 16 * 16; // After conv4
    size_t size_up2 = batch * 256 * 32 * 32;   // After up2
    size_t size_conv5 = batch * 3 * 32 * 32;   // After conv5 (output)

    // Allocate encoder activations
    act_conv1_ = new float[size_conv1];
    act_relu1_ = new float[size_conv1];
    act_pool1_ = new float[size_pool1];
    act_conv2_ = new float[size_conv2];
    act_relu2_ = new float[size_conv2];
    act_pool2_ = new float[size_pool2];

    // Allocate decoder activations
    act_conv3_ = new float[size_conv3];
    act_relu3_ = new float[size_conv3];
    act_up1_ = new float[size_up1];
    act_conv4_ = new float[size_conv4];
    act_relu4_ = new float[size_conv4];
    act_up2_ = new float[size_up2];
    act_conv5_ = new float[size_conv5];

    // Gradient buffer (same size as output)
    grad_buffer_ = new float[size_conv5];

    act_buffer_size_ = size_conv1; // Mark as allocated
}

void AutoencoderCPU::free_activations()
{
    delete[] act_conv1_;
    act_conv1_ = nullptr;
    delete[] act_relu1_;
    act_relu1_ = nullptr;
    delete[] act_pool1_;
    act_pool1_ = nullptr;
    delete[] act_conv2_;
    act_conv2_ = nullptr;
    delete[] act_relu2_;
    act_relu2_ = nullptr;
    delete[] act_pool2_;
    act_pool2_ = nullptr;

    delete[] act_conv3_;
    act_conv3_ = nullptr;
    delete[] act_relu3_;
    act_relu3_ = nullptr;
    delete[] act_up1_;
    act_up1_ = nullptr;
    delete[] act_conv4_;
    act_conv4_ = nullptr;
    delete[] act_relu4_;
    act_relu4_ = nullptr;
    delete[] act_up2_;
    act_up2_ = nullptr;
    delete[] act_conv5_;
    act_conv5_ = nullptr;

    delete[] grad_buffer_;
    grad_buffer_ = nullptr;

    act_buffer_size_ = 0;
    cached_batch_ = 0;
}

// ============================================================================
// FORWARD PASS
// ============================================================================

float *AutoencoderCPU::forward(const float *input, int batch)
{
    allocate_activations(batch);

    // ========================
    // ENCODER
    // ========================
    // Conv1: [batch, 3, 32, 32] → [batch, 256, 32, 32]
    // Pass act_conv1_ as output buffer - layer writes directly, no copy needed
    conv1_->forward(input, batch, 32, 32, act_conv1_);

    // ReLU1
    relu1_->forward(act_conv1_, batch * 256 * 32 * 32, act_relu1_);

    // Pool1: [batch, 256, 32, 32] → [batch, 256, 16, 16]
    pool1_->forward(act_relu1_, batch, 256, 32, 32, act_pool1_);

    // Conv2: [batch, 256, 16, 16] → [batch, 128, 16, 16]
    conv2_->forward(act_pool1_, batch, 16, 16, act_conv2_);

    // ReLU2
    relu2_->forward(act_conv2_, batch * 128 * 16 * 16, act_relu2_);

    // Pool2: [batch, 128, 16, 16] → [batch, 128, 8, 8] = LATENT
    pool2_->forward(act_relu2_, batch, 128, 16, 16, act_pool2_);

    // ========================
    // DECODER
    // ========================
    // Conv3: [batch, 128, 8, 8] → [batch, 128, 8, 8]
    conv3_->forward(act_pool2_, batch, 8, 8, act_conv3_);

    // ReLU3
    relu3_->forward(act_conv3_, batch * 128 * 8 * 8, act_relu3_);

    // Up1: [batch, 128, 8, 8] → [batch, 128, 16, 16]
    up1_->forward(act_relu3_, batch, 128, 8, 8, act_up1_);

    // Conv4: [batch, 128, 16, 16] → [batch, 256, 16, 16]
    conv4_->forward(act_up1_, batch, 16, 16, act_conv4_);

    // ReLU4
    relu4_->forward(act_conv4_, batch * 256 * 16 * 16, act_relu4_);

    // Up2: [batch, 256, 16, 16] → [batch, 256, 32, 32]
    up2_->forward(act_relu4_, batch, 256, 16, 16, act_up2_);

    // Conv5: [batch, 256, 32, 32] → [batch, 3, 32, 32] = OUTPUT
    conv5_->forward(act_up2_, batch, 32, 32, act_conv5_);

    return act_conv5_;
}

// ============================================================================
// FEATURE EXTRACTION (ENCODER ONLY)
// ============================================================================

float *AutoencoderCPU::extract_features(const float *input, int batch)
{
    allocate_activations(batch);

    // Run encoder only - layers write directly to buffers, no copies
    conv1_->forward(input, batch, 32, 32, act_conv1_);
    relu1_->forward(act_conv1_, batch * 256 * 32 * 32, act_relu1_);
    pool1_->forward(act_relu1_, batch, 256, 32, 32, act_pool1_);
    conv2_->forward(act_pool1_, batch, 16, 16, act_conv2_);
    relu2_->forward(act_conv2_, batch * 128 * 16 * 16, act_relu2_);
    pool2_->forward(act_relu2_, batch, 128, 16, 16, act_pool2_);

    // Result already in act_pool2_, return directly
    return act_pool2_;
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

void AutoencoderCPU::backward(const float *target, float learning_rate)
{
    int batch = cached_batch_;
    size_t output_size = batch * 3 * 32 * 32;

    // Compute gradient of MSE loss w.r.t. output
    // MSE gradient: d/dx [(1/N) * sum((x - t)^2)] = (2/N) * (x - t)
    float scale = 2.0f / output_size;
    for (size_t i = 0; i < output_size; ++i)
    {
        grad_buffer_[i] = scale * (act_conv5_[i] - target[i]);
    }

    // ========================
    // BACKPROP THROUGH DECODER
    // ========================
    // Conv5 backward
    float *grad = conv5_->backward(grad_buffer_);

    // Up2 backward
    grad = up2_->backward(grad);

    // ReLU4 backward
    grad = relu4_->backward(grad);

    // Conv4 backward
    grad = conv4_->backward(grad);

    // Up1 backward
    grad = up1_->backward(grad);

    // ReLU3 backward
    grad = relu3_->backward(grad);

    // Conv3 backward
    grad = conv3_->backward(grad);

    // ========================
    // BACKPROP THROUGH ENCODER
    // ========================
    // Pool2 backward
    grad = pool2_->backward(grad);

    // ReLU2 backward
    grad = relu2_->backward(grad);

    // Conv2 backward
    grad = conv2_->backward(grad);

    // Pool1 backward
    grad = pool1_->backward(grad);

    // ReLU1 backward
    grad = relu1_->backward(grad);

    // Conv1 backward
    grad = conv1_->backward(grad);

    // ========================
    // UPDATE WEIGHTS (SGD)
    // ========================
    conv1_->update_weights(learning_rate);
    conv2_->update_weights(learning_rate);
    conv3_->update_weights(learning_rate);
    conv4_->update_weights(learning_rate);
    conv5_->update_weights(learning_rate);
}

// ============================================================================
// LOSS COMPUTATION
// ============================================================================

float AutoencoderCPU::compute_loss(const float *output, const float *target, size_t size)
{
    double sum_squared_error = 0.0;

    for (size_t i = 0; i < size; ++i)
    {
        float diff = output[i] - target[i];
        sum_squared_error += diff * diff;
    }

    return static_cast<float>(sum_squared_error / size);
}

// ============================================================================
// WEIGHT PERSISTENCE
// ============================================================================

void AutoencoderCPU::save_weights(const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Helper lambda to save a conv layer's weights
    auto save_conv = [&file](Conv2DCPU *layer)
    {
        int w_size = layer->get_weight_size();
        int b_size = layer->get_bias_size();

        std::vector<float> weights(w_size);
        std::vector<float> bias(b_size);
        layer->get_weights(weights.data(), bias.data());

        file.write(reinterpret_cast<const char *>(&w_size), sizeof(int));
        file.write(reinterpret_cast<const char *>(&b_size), sizeof(int));
        file.write(reinterpret_cast<const char *>(weights.data()), w_size * sizeof(float));
        file.write(reinterpret_cast<const char *>(bias.data()), b_size * sizeof(float));
    };

    // Save all 5 convolutional layers
    save_conv(conv1_.get());
    save_conv(conv2_.get());
    save_conv(conv3_.get());
    save_conv(conv4_.get());
    save_conv(conv5_.get());

    file.close();
}

void AutoencoderCPU::load_weights(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Helper lambda to load a conv layer's weights
    auto load_conv = [&file](Conv2DCPU *layer)
    {
        int w_size, b_size;
        file.read(reinterpret_cast<char *>(&w_size), sizeof(int));
        file.read(reinterpret_cast<char *>(&b_size), sizeof(int));

        if (w_size != layer->get_weight_size() || b_size != layer->get_bias_size())
        {
            throw std::runtime_error("Weight size mismatch when loading weights");
        }

        std::vector<float> weights(w_size);
        std::vector<float> bias(b_size);

        file.read(reinterpret_cast<char *>(weights.data()), w_size * sizeof(float));
        file.read(reinterpret_cast<char *>(bias.data()), b_size * sizeof(float));

        layer->set_weights(weights.data(), bias.data());
    };

    // Load all 5 convolutional layers
    load_conv(conv1_.get());
    load_conv(conv2_.get());
    load_conv(conv3_.get());
    load_conv(conv4_.get());
    load_conv(conv5_.get());

    file.close();
}