#include "models/autoencoder_cpu.h"

AutoencoderCPU::AutoencoderCPU()
    : conv1_(), relu1_(), latent_(), output_() {}

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