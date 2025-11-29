/**
 * @file conv2d_cpu.h
 * @brief CPU implementation of 2D Convolution Layer
 * 
 * This file implements a standard 2D convolution layer for neural networks.
 * The convolution operation slides a kernel over the input and computes
 * dot products to produce the output feature maps.
 * 
 * Mathematical operation:
 *   output[n,oc,oh,ow] = bias[oc] + 
 *     sum over (ic,kh,kw) of input[n,ic,ih,iw] * kernel[oc,ic,kh,kw]
 *   where ih = oh*stride - padding + kh
 *         iw = ow*stride - padding + kw
 * 
 * Reference: 
 * - https://github.com/turkdogan/autoencoder
 * - CS231n Convolutional Neural Networks: https://cs231n.github.io/convolutional-networks/
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#pragma once

#include "data/data_types.h"
#include <vector>
#include <random>
#include <cmath>

/**
 * @class Conv2DCPU
 * @brief CPU implementation of 2D Convolution with forward and backward pass
 * 
 * Key features:
 * - Supports arbitrary kernel sizes, strides, and padding
 * - Uses NCHW data format [batch, channels, height, width]
 * - Weight format: [out_channels, in_channels, kernel_h, kernel_w]
 * - Xavier/He weight initialization
 * - OpenMP parallelization for multi-core CPUs
 */
class Conv2DCPU {
public:
    /**
     * @brief Construct a Conv2D layer
     * 
     * @param in_channels  Number of input channels (e.g., 3 for RGB)
     * @param out_channels Number of output feature maps (e.g., 256)
     * @param kernel_size  Size of the convolution kernel (e.g., 3 for 3x3)
     * @param stride       Stride of the convolution (default: 1)
     * @param padding      Zero-padding added to both sides (default: 0)
     * 
     * Example for encoder first layer:
     *   Conv2DCPU conv1(3, 256, 3, 1, 1);  // 3->256 channels, 3x3 kernel, pad=1
     */
    Conv2DCPU(int in_channels, int out_channels, int kernel_size,
              int stride = 1, int padding = 0);
    
    /**
     * @brief Forward pass: compute convolution output
     * 
     * @param input Input tensor of shape [N, C_in, H, W]
     * @return Tensor Output tensor of shape [N, C_out, H_out, W_out]
     * 
     * Output dimensions:
     *   H_out = (H_in + 2*padding - kernel_size) / stride + 1
     *   W_out = (W_in + 2*padding - kernel_size) / stride + 1
     */
    Tensor forward(const Tensor& input);
    
    /**
     * @brief Backward pass: compute gradients
     * 
     * Computes:
     * - Gradient w.r.t. input (for backpropagation to previous layer)
     * - Gradient w.r.t. weights (for weight update)
     * - Gradient w.r.t. bias (for bias update)
     * 
     * @param grad_output Gradient from next layer [N, C_out, H_out, W_out]
     * @return Tensor Gradient w.r.t. input [N, C_in, H_in, W_in]
     */
    Tensor backward(const Tensor& grad_output);
    
    /**
     * @brief Update weights using SGD
     * 
     * @param learning_rate Learning rate for SGD update
     */
    void update_weights(float learning_rate);
    
    /**
     * @brief Set weights and biases (for loading pre-trained models)
     * 
     * @param weights Weight tensor flattened [out_c * in_c * k * k]
     * @param bias    Bias vector [out_c]
     */
    void set_weights(const std::vector<float>& weights, const std::vector<float>& bias);
    
    /**
     * @brief Get weight and bias gradients
     * 
     * @param grad_weights Output: weight gradients
     * @param grad_bias    Output: bias gradients
     */
    void get_gradients(std::vector<float>& grad_weights, std::vector<float>& grad_bias) const;
    
    /**
     * @brief Get current weights and biases
     */
    const std::vector<float>& get_weights() const { return weights_; }
    const std::vector<float>& get_bias() const { return bias_; }
    
    /**
     * @brief Get layer configuration
     */
    int get_in_channels() const { return in_channels_; }
    int get_out_channels() const { return out_channels_; }
    int get_kernel_size() const { return kernel_size_; }
    int get_stride() const { return stride_; }
    int get_padding() const { return padding_; }
    
private:
    // Layer configuration
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    
    // Weights and biases
    // Weight shape: [out_channels, in_channels, kernel_size, kernel_size]
    std::vector<float> weights_;
    std::vector<float> bias_;
    
    // Gradients (accumulated during backward pass)
    std::vector<float> grad_weights_;
    std::vector<float> grad_bias_;
    
    // Cached input for backward pass
    Tensor cached_input_;
    
    /**
     * @brief Initialize weights using Xavier/He initialization
     * 
     * Xavier initialization: weights ~ N(0, sqrt(2 / (fan_in + fan_out)))
     * He initialization:     weights ~ N(0, sqrt(2 / fan_in))
     * 
     * We use He initialization as it works better with ReLU activations.
     */
    void initialize_weights();
    
    /**
     * @brief Compute weight index in flattened array
     * 
     * @param oc Output channel index
     * @param ic Input channel index  
     * @param kh Kernel height index
     * @param kw Kernel width index
     * @return Index in weights_ array
     */
    inline int weight_index(int oc, int ic, int kh, int kw) const {
        return ((oc * in_channels_ + ic) * kernel_size_ + kh) * kernel_size_ + kw;
    }
};
