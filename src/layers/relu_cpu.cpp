#include "layers/relu_cpu.h"
#include <cstring>
#include <algorithm>

ReLUCPU::~ReLUCPU() {
    delete[] cached_input_;
    delete[] output_buffer_;
    delete[] grad_input_buffer_;
}

float* ReLUCPU::forward(const float* input, size_t size) {
    // Cache input for backward pass
    cached_size_ = size;
    
    if (size > buffer_size_) {
        delete[] cached_input_;
        delete[] output_buffer_;
        cached_input_ = new float[size];
        output_buffer_ = new float[size];
        buffer_size_ = size;
    }
    
    std::memcpy(cached_input_, input, size * sizeof(float));

    // Apply ReLU: max(0, x)
    for (size_t i = 0; i < size; ++i) {
        output_buffer_[i] = std::max(0.0f, input[i]);
    }

    return output_buffer_;
}

float* ReLUCPU::backward(const float* grad_output) {
    if (cached_size_ > buffer_size_) {
        delete[] grad_input_buffer_;
        grad_input_buffer_ = new float[cached_size_];
    }
    
    if (grad_input_buffer_ == nullptr) {
        grad_input_buffer_ = new float[cached_size_];
    }

    // Gradient: pass through where input > 0, else 0
    for (size_t i = 0; i < cached_size_; ++i) {
        grad_input_buffer_[i] = (cached_input_[i] > 0.0f) ? grad_output[i] : 0.0f;
    }

    return grad_input_buffer_;
}