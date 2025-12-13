#pragma once
#include "layers/conv2d_cpu.h"
#include "layers/relu_cpu.h"
#include "layers/maxpool_cpu.h"
#include "layers/upsample_cpu.h"
#include <memory>

class AutoencoderCPU
{
public:
    AutoencoderCPU();

    // Forward pass (training mode: returns reconstruction)
    Tensor forward(const Tensor &input);

    // Extract features (inference mode: returns latent representation)
    Tensor extract_features(const Tensor &input);

    // Backward pass + weight update
    void backward(const Tensor &target, float learning_rate);

    // Compute MSE loss
    float compute_loss(const Tensor &output, const Tensor &target);

    // Save/load weights
    void save_weights(const std::string &filename);
    void load_weights(const std::string &filename);

private:
    // Encoder layers
    std::unique_ptr<Conv2DCPU> conv1_; // 3 → 256
    std::unique_ptr<ReLUCPU> relu1_;
    std::unique_ptr<MaxPoolCPU> pool1_;

    std::unique_ptr<Conv2DCPU> conv2_; // 256 → 128
    std::unique_ptr<ReLUCPU> relu2_;
    std::unique_ptr<MaxPoolCPU> pool2_;

    // Decoder layers
    std::unique_ptr<Conv2DCPU> conv3_; // 128 → 128
    std::unique_ptr<ReLUCPU> relu3_;
    std::unique_ptr<UpsampleCPU> up1_;

    std::unique_ptr<Conv2DCPU> conv4_; // 128 → 256
    std::unique_ptr<ReLUCPU> relu4_;
    std::unique_ptr<UpsampleCPU> up2_;

    std::unique_ptr<Conv2DCPU> conv5_; // 256 → 3

    // Cached tensors for backward pass
    Tensor latent_;
    Tensor output_;
};
