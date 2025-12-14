#include "models/autoencoder_cpu.h"
#include <fstream>
#include <cmath>

AutoencoderCPU::AutoencoderCPU()
{
    // Initialize encoder layers
    conv1_ = std::make_unique<Conv2DCPU>(3, 256, 3, 1, 1); // (32,32,3) → (32,32,256)
    relu1_ = std::make_unique<ReLUCPU>();
    pool1_ = std::make_unique<MaxPoolCPU>(2); // → (16,16,256)

    conv2_ = std::make_unique<Conv2DCPU>(256, 128, 3, 1, 1); // (16,16,256) → (16,16,128)
    relu2_ = std::make_unique<ReLUCPU>();
    pool2_ = std::make_unique<MaxPoolCPU>(2); // → (8,8,128)

    // Initialize decoder layers
    conv3_ = std::make_unique<Conv2DCPU>(128, 128, 3, 1, 1); // (8,8,128) → (8,8,128)
    relu3_ = std::make_unique<ReLUCPU>();
    up1_ = std::make_unique<UpsampleCPU>(2); // → (16,16,128)

    conv4_ = std::make_unique<Conv2DCPU>(128, 256, 3, 1, 1); // (16,16,128) → (16,16,256)
    relu4_ = std::make_unique<ReLUCPU>();
    up2_ = std::make_unique<UpsampleCPU>(2); // → (32,32,256)

    conv5_ = std::make_unique<Conv2DCPU>(256, 3, 3, 1, 1); // (32,32,256) → (32,32,3)
}

Tensor AutoencoderCPU::forward(const Tensor &input)
{
    // ENCODER
    auto x = conv1_->forward(input); // (32,32,3) → (32,32,256)
    x = relu1_->forward(x);
    x = pool1_->forward(x); // → (16,16,256)

    x = conv2_->forward(x); // → (16,16,128)
    x = relu2_->forward(x);
    latent_ = pool2_->forward(x); // → (8,8,128) = 8192 features

    // DECODER
    x = conv3_->forward(latent_); // (8,8,128) → (8,8,128)
    x = relu3_->forward(x);
    x = up1_->forward(x); // → (16,16,128)

    x = conv4_->forward(x); // → (16,16,256)
    x = relu4_->forward(x);
    x = up2_->forward(x); // → (32,32,256)

    output_ = conv5_->forward(x); // → (32,32,3)

    return output_;
}

void AutoencoderCPU::backward(const Tensor &target, float learning_rate)
{
    // Compute gradient of loss w.r.t. output
    Tensor grad = output_; // Copy
    float *grad_data = grad.data->data();
    const float *target_data = target.data->data();
    size_t size = grad.size();

    // MSE gradient: 2 * (output - target) / N
    float scale = 2.0f / size;
    for (size_t i = 0; i < size; ++i)
    {
        grad_data[i] = scale * (grad_data[i] - target_data[i]);
    }

    // Backpropagate through decoder
    grad = conv5_->backward(grad);
    grad = up2_->backward(grad);
    grad = relu4_->backward(grad);
    grad = conv4_->backward(grad);
    grad = up1_->backward(grad);
    grad = relu3_->backward(grad);
    grad = conv3_->backward(grad);

    // Backpropagate through encoder
    grad = pool2_->backward(grad);
    grad = relu2_->backward(grad);
    grad = conv2_->backward(grad);
    grad = pool1_->backward(grad);
    grad = relu1_->backward(grad);
    grad = conv1_->backward(grad);

    // Update weights
    conv1_->update_weights(learning_rate);
    conv2_->update_weights(learning_rate);
    conv3_->update_weights(learning_rate);
    conv4_->update_weights(learning_rate);
    conv5_->update_weights(learning_rate);
}

Tensor AutoencoderCPU::extract_features(const Tensor &input)
{
    // Run only the encoder path
    auto x = conv1_->forward(input);
    x = relu1_->forward(x);
    x = pool1_->forward(x);

    x = conv2_->forward(x);
    x = relu2_->forward(x);
    x = pool2_->forward(x); // Returns (8,8,128) = 8192 features

    return x;
}

float AutoencoderCPU::compute_loss(const Tensor &output, const Tensor &target)
{
    // Mean Squared Error (MSE) loss
    const float *out_data = output.data->data();
    const float *target_data = target.data->data();
    size_t size = output.size();

    double sum_squared_error = 0.0;

    for (size_t i = 0; i < size; ++i)
    {
        float diff = out_data[i] - target_data[i];
        sum_squared_error += diff * diff;
    }

    return static_cast<float>(sum_squared_error / size);
}

void AutoencoderCPU::save_weights(const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Helper lambda to save layer weights
    auto save_conv_layer = [&file](Conv2DCPU *layer)
    {
        std::vector<float> weights, bias;
        layer->get_weights(weights, bias);

        // Write size headers then data
        int w_size = weights.size();
        int b_size = bias.size();

        file.write(reinterpret_cast<const char *>(&w_size), sizeof(int));
        file.write(reinterpret_cast<const char *>(&b_size), sizeof(int));
        file.write(reinterpret_cast<const char *>(weights.data()), w_size * sizeof(float));
        file.write(reinterpret_cast<const char *>(bias.data()), b_size * sizeof(float));
    };

    // Save all convolutional layers
    save_conv_layer(conv1_.get());
    save_conv_layer(conv2_.get());
    save_conv_layer(conv3_.get());
    save_conv_layer(conv4_.get());
    save_conv_layer(conv5_.get());

    file.close();
}

void AutoencoderCPU::load_weights(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Helper lambda to load layer weights
    auto load_conv_layer = [&file](Conv2DCPU *layer)
    {
        int w_size, b_size;
        file.read(reinterpret_cast<char *>(&w_size), sizeof(int));
        file.read(reinterpret_cast<char *>(&b_size), sizeof(int));

        std::vector<float> weights(w_size);
        std::vector<float> bias(b_size);

        file.read(reinterpret_cast<char *>(weights.data()), w_size * sizeof(float));
        file.read(reinterpret_cast<char *>(bias.data()), b_size * sizeof(float));

        layer->set_weight(weights, bias);
    };

    // Load all convolutional layers
    load_conv_layer(conv1_.get());
    load_conv_layer(conv2_.get());
    load_conv_layer(conv3_.get());
    load_conv_layer(conv4_.get());
    load_conv_layer(conv5_.get());

    file.close();
}