#include "layers/conv2d_cpu.h"
#include <cstring>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>

Conv2DCPU::Conv2DCPU(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : in_c_(in_channels), out_c_(out_channels), k_size_(kernel_size), stride_(stride), pad_(padding)
{
    // Initialize weights and biases with Xavier/He initialization
    int weight_size = out_c_ * in_c_ * k_size_ * k_size_;
    weights_.resize(weight_size);
    bias_.resize(out_c_);
    grad_w_.resize(weight_size);
    grad_b_.resize(out_c_);

    // Xavier initialization: scale = sqrt(2.0 / (in_c * k_size * k_size))
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / (in_c_ * k_size_ * k_size_));
    std::normal_distribution<float> dist(0.0f, scale);

    for (int i = 0; i < weight_size; ++i) {
        weights_[i] = dist(gen);
    }

    // Initialize biases to zero
    std::fill(bias_.begin(), bias_.end(), 0.0f);
}

Conv2DCPU::~Conv2DCPU() {
    delete[] cached_input_;
    delete[] output_buffer_;
    delete[] grad_input_buffer_;
}

float* Conv2DCPU::forward(const float* input, int batch, int in_h, int in_w) {
    // Cache input for backward pass
    cached_batch_ = batch;
    cached_in_h_ = in_h;
    cached_in_w_ = in_w;
    
    size_t input_size = static_cast<size_t>(batch) * in_c_ * in_h * in_w;
    if (input_size > cached_input_size_) {
        delete[] cached_input_;
        cached_input_ = new float[input_size];
        cached_input_size_ = input_size;
    }
    std::memcpy(cached_input_, input, input_size * sizeof(float));

    // Calculate output dimensions
    int out_h = get_output_height(in_h);
    int out_w = get_output_width(in_w);
    
    size_t output_size = static_cast<size_t>(batch) * out_c_ * out_h * out_w;
    if (output_size > output_buffer_size_) {
        delete[] output_buffer_;
        output_buffer_ = new float[output_size];
        output_buffer_size_ = output_size;
    }

    // Perform convolution
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < out_c_; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = bias_[oc];

                    for (int ic = 0; ic < in_c_; ++ic) {
                        for (int kh = 0; kh < k_size_; ++kh) {
                            for (int kw = 0; kw < k_size_; ++kw) {
                                int ih = oh * stride_ - pad_ + kh;
                                int iw = ow * stride_ - pad_ + kw;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = ((n * in_c_ + ic) * in_h + ih) * in_w + iw;
                                    int w_idx = ((oc * in_c_ + ic) * k_size_ + kh) * k_size_ + kw;
                                    sum += input[in_idx] * weights_[w_idx];
                                }
                            }
                        }
                    }

                    int out_idx = ((n * out_c_ + oc) * out_h + oh) * out_w + ow;
                    output_buffer_[out_idx] = sum;
                }
            }
        }
    }

    return output_buffer_;
}

float* Conv2DCPU::backward(const float* grad_output) {
    int batch = cached_batch_;
    int in_h = cached_in_h_;
    int in_w = cached_in_w_;
    int out_h = get_output_height(in_h);
    int out_w = get_output_width(in_w);

    // Allocate gradient input buffer
    size_t grad_input_size = static_cast<size_t>(batch) * in_c_ * in_h * in_w;
    if (grad_input_size > grad_input_buffer_size_) {
        delete[] grad_input_buffer_;
        grad_input_buffer_ = new float[grad_input_size];
        grad_input_buffer_size_ = grad_input_size;
    }

    // Zero initialize gradients
    std::memset(grad_input_buffer_, 0, grad_input_size * sizeof(float));
    std::memset(grad_w_.data(), 0, grad_w_.size() * sizeof(float));
    std::memset(grad_b_.data(), 0, grad_b_.size() * sizeof(float));

    // Compute gradients
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < out_c_; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx = ((n * out_c_ + oc) * out_h + oh) * out_w + ow;
                    float grad_out = grad_output[out_idx];
                    grad_b_[oc] += grad_out;

                    for (int ic = 0; ic < in_c_; ++ic) {
                        for (int kh = 0; kh < k_size_; ++kh) {
                            for (int kw = 0; kw < k_size_; ++kw) {
                                int ih = oh * stride_ - pad_ + kh;
                                int iw = ow * stride_ - pad_ + kw;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = ((n * in_c_ + ic) * in_h + ih) * in_w + iw;
                                    int w_idx = ((oc * in_c_ + ic) * k_size_ + kh) * k_size_ + kw;

                                    grad_w_[w_idx] += grad_out * cached_input_[in_idx];
                                    grad_input_buffer_[in_idx] += grad_out * weights_[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input_buffer_;
}

void Conv2DCPU::set_weights(const float* weights, const float* bias) {
    std::memcpy(weights_.data(), weights, weights_.size() * sizeof(float));
    std::memcpy(bias_.data(), bias, bias_.size() * sizeof(float));
}

void Conv2DCPU::get_weights(float* weights, float* bias) const {
    std::memcpy(weights, weights_.data(), weights_.size() * sizeof(float));
    std::memcpy(bias, bias_.data(), bias_.size() * sizeof(float));
}

void Conv2DCPU::get_gradients(float* grad_w, float* grad_b) const {
    std::memcpy(grad_w, grad_w_.data(), grad_w_.size() * sizeof(float));
    std::memcpy(grad_b, grad_b_.data(), grad_b_.size() * sizeof(float));
}

void Conv2DCPU::update_weights(float learning_rate) {
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate * grad_w_[i];
    }
    for (size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] -= learning_rate * grad_b_[i];
    }
}