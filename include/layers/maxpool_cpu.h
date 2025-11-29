/**
 * @file maxpool_cpu.h
 * @brief CPU implementation of 2D Max Pooling Layer
 * 
 * Max pooling is a downsampling operation that reduces spatial dimensions
 * by taking the maximum value in each pooling window.
 * 
 * Mathematical operation:
 *   output[n,c,oh,ow] = max over (ph,pw in pool window) of input[n,c,ih+ph,iw+pw]
 *   where ih = oh * stride, iw = ow * stride
 * 
 * Properties:
 * - Reduces spatial dimensions by factor of stride
 * - Provides translation invariance
 * - Reduces computation for subsequent layers
 * - Helps prevent overfitting
 * 
 * Reference:
 * - https://cs231n.github.io/convolutional-networks/#pool
 * - https://github.com/turkdogan/autoencoder
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#pragma once

#include "data/data_types.h"
#include <vector>

/**
 * @class MaxPoolCPU
 * @brief CPU implementation of 2D Max Pooling with forward and backward pass
 * 
 * For the autoencoder, we use 2x2 pooling with stride 2, which halves
 * the spatial dimensions.
 */
class MaxPoolCPU {
public:
    /**
     * @brief Construct a MaxPool layer
     * 
     * @param pool_size Size of the pooling window (e.g., 2 for 2x2)
     * @param stride    Stride of the pooling (default: same as pool_size)
     * 
     * Example:
     *   MaxPoolCPU pool(2);  // 2x2 pooling with stride 2
     */
    explicit MaxPoolCPU(int pool_size, int stride = -1);
    
    /**
     * @brief Forward pass: apply max pooling
     * 
     * @param input Input tensor of shape [N, C, H, W]
     * @return Tensor Output tensor of shape [N, C, H/pool_size, W/pool_size]
     */
    Tensor forward(const Tensor& input);
    
    /**
     * @brief Backward pass: compute gradient w.r.t. input
     * 
     * Gradients are routed only to positions where the max was found.
     * 
     * @param grad_output Gradient from next layer
     * @return Tensor Gradient w.r.t. input (same shape as input to forward)
     */
    Tensor backward(const Tensor& grad_output);
    
    /**
     * @brief Get layer configuration
     */
    int get_pool_size() const { return pool_size_; }
    int get_stride() const { return stride_; }
    
private:
    int pool_size_;
    int stride_;
    
    /**
     * @brief Cached input shape for backward pass
     */
    std::vector<int> cached_input_shape_;
    
    /**
     * @brief Indices of maximum values for backward pass
     * 
     * For each output position, we store the index in the input tensor
     * where the maximum was found. This allows efficient gradient routing.
     * 
     * Shape: [N * C * out_H * out_W]
     */
    std::vector<int> max_indices_;
};
