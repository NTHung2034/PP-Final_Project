/**
 * @file conv2d_cpu.cpp
 * @brief CPU implementation of 2D Convolution Layer
 * 
 * This implementation follows the standard convolution algorithm with
 * optimizations for cache locality and OpenMP parallelization.
 * 
 * Algorithm complexity:
 *   Forward:  O(N * C_out * H_out * W_out * C_in * K * K)
 *   Backward: O(N * C_out * H_out * W_out * C_in * K * K) + O(weight_size)
 * 
 * Reference implementations:
 * - turkdogan/autoencoder: https://github.com/turkdogan/autoencoder
 * - Caffe convolution layer
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#include "layers/conv2d_cpu.h"
#include "config.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

Conv2DCPU::Conv2DCPU(int in_channels, int out_channels, int kernel_size,
                     int stride, int padding)
    : in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , cached_input_({1, 1, 1, 1}) // Placeholder, will be replaced during forward
{
    // Validate parameters
    if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0) {
        throw std::invalid_argument("Conv2D: channels and kernel_size must be positive");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Conv2D: stride must be positive");
    }
    if (padding < 0) {
        throw std::invalid_argument("Conv2D: padding must be non-negative");
    }
    
    // Allocate weights and biases
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    weights_.resize(weight_size);
    bias_.resize(out_channels);
    
    // Allocate gradients
    grad_weights_.resize(weight_size, 0.0f);
    grad_bias_.resize(out_channels, 0.0f);
    
    // Initialize weights
    initialize_weights();
}

void Conv2DCPU::initialize_weights() {
    /**
     * He initialization (recommended for ReLU):
     *   std = sqrt(2.0 / fan_in)
     * where fan_in = in_channels * kernel_size * kernel_size
     * 
     * Reference: He et al., "Delving Deep into Rectifiers" (2015)
     * https://arxiv.org/abs/1502.01852
     */
    int fan_in = in_channels_ * kernel_size_ * kernel_size_;
    float std_dev = std::sqrt(2.0f / fan_in);
    
    std::mt19937 rng(RANDOM_SEED);
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    // Initialize weights with He initialization
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = dist(rng);
    }
    
    // Initialize biases to small positive values (helps with ReLU)
    for (size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] = 0.01f;
    }
}

Tensor Conv2DCPU::forward(const Tensor& input) {
    /**
     * Forward pass computation:
     * 
     * For each output position (n, oc, oh, ow):
     *   output[n,oc,oh,ow] = bias[oc] + 
     *     sum_{ic,kh,kw} input[n,ic,ih,iw] * weight[oc,ic,kh,kw]
     * 
     * where:
     *   ih = oh * stride - padding + kh
     *   iw = ow * stride - padding + kw
     * 
     * Zero-padding is handled by checking bounds (implicit padding).
     */
    
    // Cache input for backward pass
    cached_input_ = input;
    
    // Get input dimensions
    int batch = input.batch();
    int in_h = input.height();
    int in_w = input.width();
    
    // Calculate output dimensions
    // Formula: out_size = (in_size + 2*padding - kernel_size) / stride + 1
    int out_h = (in_h + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_w = (in_w + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    // Validate output dimensions
    if (out_h <= 0 || out_w <= 0) {
        throw std::runtime_error("Conv2D: Invalid output dimensions. Check input size, kernel, padding, stride.");
    }
    
    // Allocate output tensor
    Tensor output({batch, out_channels_, out_h, out_w});
    
    // Get raw pointers for efficient access
    const float* in_data = input.data->data();
    float* out_data = output.data->data();
    
    // Compute strides for NCHW layout
    const int in_stride_n = in_channels_ * in_h * in_w;
    const int in_stride_c = in_h * in_w;
    const int in_stride_h = in_w;
    
    const int out_stride_n = out_channels_ * out_h * out_w;
    const int out_stride_c = out_h * out_w;
    const int out_stride_h = out_w;
    
    // Parallelize over batch and output channels (outer loops)
    // This provides good parallelism and cache locality
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            // Compute output for this (n, oc) slice
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    // Start with bias
                    float sum = bias_[oc];
                    
                    // Convolve over all input channels and kernel positions
                    for (int ic = 0; ic < in_channels_; ++ic) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                // Calculate input position
                                int ih = oh * stride_ - padding_ + kh;
                                int iw = ow * stride_ - padding_ + kw;
                                
                                // Check bounds (implicit zero-padding)
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    // Compute indices
                                    int in_idx = n * in_stride_n + ic * in_stride_c + 
                                                 ih * in_stride_h + iw;
                                    int w_idx = weight_index(oc, ic, kh, kw);
                                    
                                    sum += in_data[in_idx] * weights_[w_idx];
                                }
                            }
                        }
                    }
                    
                    // Store result
                    int out_idx = n * out_stride_n + oc * out_stride_c + 
                                  oh * out_stride_h + ow;
                    out_data[out_idx] = sum;
                }
            }
        }
    }
    
    return output;
}

Tensor Conv2DCPU::backward(const Tensor& grad_output) {
    /**
     * Backward pass computes three gradients:
     * 
     * 1. Gradient w.r.t. input (for backpropagation):
     *    grad_input[n,ic,ih,iw] = sum_{oc,kh,kw} grad_output[n,oc,oh,ow] * weight[oc,ic,kh,kw]
     *    where oh,ow are positions where (ih,iw) contributed to the convolution
     * 
     * 2. Gradient w.r.t. weights (for weight update):
     *    grad_weight[oc,ic,kh,kw] = sum_{n,oh,ow} grad_output[n,oc,oh,ow] * input[n,ic,ih,iw]
     * 
     * 3. Gradient w.r.t. bias (for bias update):
     *    grad_bias[oc] = sum_{n,oh,ow} grad_output[n,oc,oh,ow]
     * 
     * Reference: Backpropagation theory for CNNs
     * https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
     */
    
    // Get dimensions
    int batch = cached_input_.batch();
    int in_h = cached_input_.height();
    int in_w = cached_input_.width();
    int out_h = grad_output.height();
    int out_w = grad_output.width();
    
    // Get raw pointers
    const float* in_data = cached_input_.data->data();
    const float* grad_out_data = grad_output.data->data();
    
    // Allocate gradient w.r.t. input
    Tensor grad_input({batch, in_channels_, in_h, in_w});
    float* grad_in_data = grad_input.data->data();
    
    // Reset gradients (accumulate over batch)
    std::fill(grad_weights_.begin(), grad_weights_.end(), 0.0f);
    std::fill(grad_bias_.begin(), grad_bias_.end(), 0.0f);
    
    // Compute strides
    const int in_stride_n = in_channels_ * in_h * in_w;
    const int in_stride_c = in_h * in_w;
    const int in_stride_h = in_w;
    
    const int out_stride_n = out_channels_ * out_h * out_w;
    const int out_stride_c = out_h * out_w;
    const int out_stride_h = out_w;
    
    // =========================================
    // Compute gradient w.r.t. input
    // =========================================
    // This is a "full" convolution of grad_output with flipped weights
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < batch; ++n) {
        for (int ic = 0; ic < in_channels_; ++ic) {
            for (int ih = 0; ih < in_h; ++ih) {
                for (int iw = 0; iw < in_w; ++iw) {
                    float sum = 0.0f;
                    
                    // Find all output positions that this input contributed to
                    for (int oc = 0; oc < out_channels_; ++oc) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                // Calculate corresponding output position
                                // ih = oh * stride - padding + kh
                                // => oh = (ih + padding - kh) / stride
                                int oh_times_stride = ih + padding_ - kh;
                                if (oh_times_stride % stride_ != 0) continue;
                                int oh = oh_times_stride / stride_;
                                
                                int ow_times_stride = iw + padding_ - kw;
                                if (ow_times_stride % stride_ != 0) continue;
                                int ow = ow_times_stride / stride_;
                                
                                // Check bounds
                                if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                    int out_idx = n * out_stride_n + oc * out_stride_c + 
                                                  oh * out_stride_h + ow;
                                    int w_idx = weight_index(oc, ic, kh, kw);
                                    
                                    sum += grad_out_data[out_idx] * weights_[w_idx];
                                }
                            }
                        }
                    }
                    
                    int in_idx = n * in_stride_n + ic * in_stride_c + 
                                 ih * in_stride_h + iw;
                    grad_in_data[in_idx] = sum;
                }
            }
        }
    }
    
    // =========================================
    // Compute gradient w.r.t. weights and bias
    // =========================================
    // Note: We accumulate across the batch
    
    // Use thread-local storage for gradient accumulation to avoid race conditions
    #pragma omp parallel
    {
        std::vector<float> local_grad_weights(grad_weights_.size(), 0.0f);
        std::vector<float> local_grad_bias(grad_bias_.size(), 0.0f);
        
        #pragma omp for collapse(2) schedule(static)
        for (int n = 0; n < batch; ++n) {
            for (int oc = 0; oc < out_channels_; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        int out_idx = n * out_stride_n + oc * out_stride_c + 
                                      oh * out_stride_h + ow;
                        float grad_out_val = grad_out_data[out_idx];
                        
                        // Accumulate bias gradient
                        local_grad_bias[oc] += grad_out_val;
                        
                        // Accumulate weight gradients
                        for (int ic = 0; ic < in_channels_; ++ic) {
                            for (int kh = 0; kh < kernel_size_; ++kh) {
                                for (int kw = 0; kw < kernel_size_; ++kw) {
                                    int ih = oh * stride_ - padding_ + kh;
                                    int iw = ow * stride_ - padding_ + kw;
                                    
                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        int in_idx = n * in_stride_n + ic * in_stride_c + 
                                                     ih * in_stride_h + iw;
                                        int w_idx = weight_index(oc, ic, kh, kw);
                                        
                                        local_grad_weights[w_idx] += grad_out_val * in_data[in_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Reduce local gradients to global
        #pragma omp critical
        {
            for (size_t i = 0; i < grad_weights_.size(); ++i) {
                grad_weights_[i] += local_grad_weights[i];
            }
            for (size_t i = 0; i < grad_bias_.size(); ++i) {
                grad_bias_[i] += local_grad_bias[i];
            }
        }
    }
    
    return grad_input;
}

void Conv2DCPU::update_weights(float learning_rate) {
    /**
     * SGD weight update:
     *   weight = weight - learning_rate * gradient
     * 
     * Note: Gradients are already averaged over the batch in the loss function.
     */
    
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate * grad_weights_[i];
    }
    
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] -= learning_rate * grad_bias_[i];
    }
}

void Conv2DCPU::set_weights(const std::vector<float>& weights, const std::vector<float>& bias) {
    if (weights.size() != weights_.size()) {
        throw std::invalid_argument("Conv2D::set_weights: weight size mismatch");
    }
    if (bias.size() != bias_.size()) {
        throw std::invalid_argument("Conv2D::set_weights: bias size mismatch");
    }
    
    weights_ = weights;
    bias_ = bias;
}

void Conv2DCPU::get_gradients(std::vector<float>& grad_weights, std::vector<float>& grad_bias) const {
    grad_weights = grad_weights_;
    grad_bias = grad_bias_;
}
