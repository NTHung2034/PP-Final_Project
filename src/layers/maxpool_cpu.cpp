/**
 * @file maxpool_cpu.cpp
 * @brief CPU implementation of 2D Max Pooling Layer
 * 
 * This implementation efficiently computes max pooling by iterating
 * through each pooling window and tracking the maximum value and its index.
 * The indices are stored for the backward pass.
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#include "layers/maxpool_cpu.h"
#include <limits>
#include <stdexcept>
#include <algorithm>

MaxPoolCPU::MaxPoolCPU(int pool_size, int stride)
    : pool_size_(pool_size)
    , stride_(stride < 0 ? pool_size : stride) // Default: stride = pool_size
{
    if (pool_size <= 0) {
        throw std::invalid_argument("MaxPool: pool_size must be positive");
    }
    if (stride_ <= 0) {
        throw std::invalid_argument("MaxPool: stride must be positive");
    }
}

Tensor MaxPoolCPU::forward(const Tensor& input) {
    /**
     * Forward pass: Max pooling operation
     * 
     * For each pool window, find the maximum value and store:
     * 1. The max value in the output
     * 2. The index of the max for the backward pass
     */
    
    // Cache input shape for backward pass
    cached_input_shape_ = input.shape;
    
    // Get input dimensions
    int batch = input.batch();
    int channels = input.channels();
    int in_h = input.height();
    int in_w = input.width();
    
    // Calculate output dimensions
    int out_h = (in_h - pool_size_) / stride_ + 1;
    int out_w = (in_w - pool_size_) / stride_ + 1;
    
    if (out_h <= 0 || out_w <= 0) {
        throw std::runtime_error("MaxPool: Invalid output dimensions. Input too small for pool size.");
    }
    
    // Allocate output tensor
    Tensor output({batch, channels, out_h, out_w});
    
    // Allocate max indices storage
    size_t output_size = batch * channels * out_h * out_w;
    max_indices_.resize(output_size);
    
    // Get raw pointers
    const float* in_data = input.data->data();
    float* out_data = output.data->data();
    
    // Compute strides for NCHW layout
    const int in_stride_n = channels * in_h * in_w;
    const int in_stride_c = in_h * in_w;
    const int in_stride_h = in_w;
    
    const int out_stride_n = channels * out_h * out_w;
    const int out_stride_c = out_h * out_w;
    const int out_stride_h = out_w;
    
    // Perform max pooling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    // Find maximum in pool window
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = -1;
                    
                    // Starting input position
                    int ih_start = oh * stride_;
                    int iw_start = ow * stride_;
                    
                    // Iterate over pool window
                    for (int ph = 0; ph < pool_size_; ++ph) {
                        for (int pw = 0; pw < pool_size_; ++pw) {
                            int ih = ih_start + ph;
                            int iw = iw_start + pw;
                            
                            // Compute input index
                            int in_idx = n * in_stride_n + c * in_stride_c + 
                                        ih * in_stride_h + iw;
                            
                            if (in_data[in_idx] > max_val) {
                                max_val = in_data[in_idx];
                                max_idx = in_idx;
                            }
                        }
                    }
                    
                    // Store output and max index
                    int out_idx = n * out_stride_n + c * out_stride_c + 
                                  oh * out_stride_h + ow;
                    out_data[out_idx] = max_val;
                    max_indices_[out_idx] = max_idx;
                }
            }
        }
    }
    
    return output;
}

Tensor MaxPoolCPU::backward(const Tensor& grad_output) {
    /**
     * Backward pass: Route gradients to max positions
     * 
     * The gradient is passed only to the positions where the maximum
     * was found during the forward pass. All other positions receive zero.
     * 
     * grad_input[max_idx] = grad_output[out_idx]
     */
    
    // Get dimensions
    int batch = cached_input_shape_[0];
    int channels = cached_input_shape_[1];
    int in_h = cached_input_shape_[2];
    int in_w = cached_input_shape_[3];
    
    int out_h = grad_output.height();
    int out_w = grad_output.width();
    
    // Allocate gradient tensor (initialized to zero)
    Tensor grad_input(cached_input_shape_);
    
    // Get raw pointers
    const float* grad_out_data = grad_output.data->data();
    float* grad_in_data = grad_input.data->data();
    
    // Compute strides
    const int out_stride_n = channels * out_h * out_w;
    const int out_stride_c = out_h * out_w;
    const int out_stride_h = out_w;
    
    // Route gradients to max positions
    // Note: We cannot parallelize this easily because multiple output positions
    // could theoretically map to the same input position (though rare in practice)
    // For correctness, we use atomic operations or sequential processing
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx = n * out_stride_n + c * out_stride_c + 
                                  oh * out_stride_h + ow;
                    int max_idx = max_indices_[out_idx];
                    
                    // Route gradient to the max position
                    // Use atomic add in case of overlapping (rare but possible)
                    #pragma omp atomic
                    grad_in_data[max_idx] += grad_out_data[out_idx];
                }
            }
        }
    }
    
    return grad_input;
}
