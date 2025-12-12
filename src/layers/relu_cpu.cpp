#include "layers/relu_cpu.h"
#include <cstring>
#include <algorithm>

ReLUCPU::~ReLUCPU()
{
    delete[] cached_input_;
    delete[] output_buffer_;
    delete[] grad_input_buffer_;
}

float *ReLUCPU::forward(const float *input, size_t size, float *output)
{
    // Cache input for backward pass
    cached_size_ = size;

    if (size > buffer_size_)
    {
        delete[] cached_input_;
        cached_input_ = new float[size];
        buffer_size_ = size;

        // Only allocate output buffer if we need internal storage
        if (output == nullptr)
        {
            delete[] output_buffer_;
            output_buffer_ = new float[size];
        }
    }

    std::memcpy(cached_input_, input, size * sizeof(float));

    // Use provided buffer or internal buffer
    float *out_ptr = (output != nullptr) ? output : output_buffer_;

    // Apply ReLU: max(0, x)
    for (size_t i = 0; i < size; ++i)
    {
        out_ptr[i] = std::max(0.0f, input[i]);
    }

    return out_ptr;
}

float *ReLUCPU::backward(const float *grad_output)
{
    if (cached_size_ > buffer_size_)
    {
        delete[] grad_input_buffer_;
        grad_input_buffer_ = new float[cached_size_];
    }

    if (grad_input_buffer_ == nullptr)
    {
        grad_input_buffer_ = new float[cached_size_];
    }

    // Gradient: pass through where input > 0, else 0
    for (size_t i = 0; i < cached_size_; ++i)
    {
        grad_input_buffer_[i] = (cached_input_[i] > 0.0f) ? grad_output[i] : 0.0f;
    }

    return grad_input_buffer_;
}