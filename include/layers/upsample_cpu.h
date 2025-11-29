/**
 * @file upsample_cpu.h
 * @brief CPU implementation of 2D Upsampling Layer (Nearest Neighbor)
 * 
 * Upsampling is used in the decoder part of the autoencoder to increase
 * spatial dimensions. This implementation uses nearest neighbor interpolation.
 * 
 * Mathematical operation:
 *   output[n,c,oh,ow] = input[n,c,oh/scale,ow/scale]
 * 
 * Properties:
 * - Increases spatial dimensions by factor of scale
 * - Computationally efficient (no learned parameters)
 * - Preserves values from input (no interpolation artifacts)
 * 
 * Alternative methods (not implemented here):
 * - Bilinear interpolation (smoother but more computation)
 * - Transposed convolution (learnable but has checkerboard artifacts)
 * 
 * Reference:
 * - https://distill.pub/2016/deconv-checkerboard/
 * - https://github.com/turkdogan/autoencoder
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#pragma once

#include "data/data_types.h"
#include <vector>

/**
 * @class UpsampleCPU
 * @brief CPU implementation of Nearest Neighbor Upsampling
 * 
 * For the autoencoder, we use 2x upsampling to double spatial dimensions.
 */
class UpsampleCPU {
public:
    /**
     * @brief Construct an Upsample layer
     * 
     * @param scale Upsampling factor (e.g., 2 to double dimensions)
     * 
     * Example:
     *   UpsampleCPU upsample(2);  // 2x upsampling
     */
    explicit UpsampleCPU(int scale);
    
    /**
     * @brief Forward pass: apply upsampling
     * 
     * @param input Input tensor of shape [N, C, H, W]
     * @return Tensor Output tensor of shape [N, C, H*scale, W*scale]
     */
    Tensor forward(const Tensor& input);
    
    /**
     * @brief Backward pass: compute gradient w.r.t. input
     * 
     * Since each input pixel is copied to scale*scale output pixels,
     * the gradient is the sum of gradients from those output positions.
     * 
     * @param grad_output Gradient from next layer
     * @return Tensor Gradient w.r.t. input (same shape as input to forward)
     */
    Tensor backward(const Tensor& grad_output);
    
    /**
     * @brief Get layer configuration
     */
    int get_scale() const { return scale_; }
    
private:
    int scale_;
    
    /**
     * @brief Cached input shape for backward pass
     */
    std::vector<int> cached_input_shape_;
};
