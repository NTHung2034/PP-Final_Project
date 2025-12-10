#include "layers/maxpool_cpu.h"
#include <limits>
#include <cstring>

MaxPoolCPU::MaxPoolCPU(int pool_size) : pool_size_(pool_size) {}

MaxPoolCPU::~MaxPoolCPU() {
    delete[] output_buffer_;
    delete[] grad_input_buffer_;
}

float* MaxPoolCPU::forward(const float* input, int batch, int channels, int in_h, int in_w) {
    // Cache dimensions for backward pass
    cached_batch_ = batch;
    cached_channels_ = channels;
    cached_in_h_ = in_h;
    cached_in_w_ = in_w;

    int out_h = get_output_height(in_h);
    int out_w = get_output_width(in_w);

    size_t output_size = static_cast<size_t>(batch) * channels * out_h * out_w;
    
    // Allocate output buffer if needed
    if (output_size > output_buffer_size_) {
        delete[] output_buffer_;
        output_buffer_ = new float[output_size];
        output_buffer_size_ = output_size;
    }
    
    // Resize max indices
    max_indices_.resize(output_size);

    // Perform max pooling
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = 0;

                    for (int ph = 0; ph < pool_size_; ++ph) {
                        for (int pw = 0; pw < pool_size_; ++pw) {
                            int ih = oh * pool_size_ + ph;
                            int iw = ow * pool_size_ + pw;
                            int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                            
                            if (input[in_idx] > max_val) {
                                max_val = input[in_idx];
                                max_idx = in_idx;
                            }
                        }
                    }

                    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                    output_buffer_[out_idx] = max_val;
                    max_indices_[out_idx] = max_idx;
                }
            }
        }
    }

    return output_buffer_;
}

float* MaxPoolCPU::backward(const float* grad_output) {
    int batch = cached_batch_;
    int channels = cached_channels_;
    int in_h = cached_in_h_;
    int in_w = cached_in_w_;
    int out_h = get_output_height(in_h);
    int out_w = get_output_width(in_w);

    size_t grad_input_size = static_cast<size_t>(batch) * channels * in_h * in_w;
    
    // Allocate gradient input buffer if needed
    if (grad_input_size > grad_input_buffer_size_) {
        delete[] grad_input_buffer_;
        grad_input_buffer_ = new float[grad_input_size];
        grad_input_buffer_size_ = grad_input_size;
    }

    // Zero initialize gradient input
    std::memset(grad_input_buffer_, 0, grad_input_size * sizeof(float));

    // Distribute gradients only to max positions
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                    int max_idx = max_indices_[out_idx];
                    grad_input_buffer_[max_idx] += grad_output[out_idx];
                }
            }
        }
    }

    return grad_input_buffer_;
}