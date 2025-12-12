#include "layers/upsample_cpu.h"
#include <cstring>

UpsampleCPU::UpsampleCPU(int scale) : scale_(scale) {}

UpsampleCPU::~UpsampleCPU()
{
    delete[] output_buffer_;
    delete[] grad_input_buffer_;
}

float *UpsampleCPU::forward(const float *input, int batch, int channels, int in_h, int in_w, float *output)
{
    // Cache dimensions for backward pass
    cached_batch_ = batch;
    cached_channels_ = channels;
    cached_in_h_ = in_h;
    cached_in_w_ = in_w;

    int out_h = get_output_height(in_h);
    int out_w = get_output_width(in_w);

    size_t output_size = static_cast<size_t>(batch) * channels * out_h * out_w;

    // Only allocate output buffer if we need internal storage
    if (output == nullptr && output_size > output_buffer_size_)
    {
        delete[] output_buffer_;
        output_buffer_ = new float[output_size];
        output_buffer_size_ = output_size;
    }

    // Use provided buffer or internal buffer
    float *out_ptr = (output != nullptr) ? output : output_buffer_;

    // Perform nearest neighbor upsampling
    for (int n = 0; n < batch; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int oh = 0; oh < out_h; ++oh)
            {
                for (int ow = 0; ow < out_w; ++ow)
                {
                    // Map output to input coordinates
                    int ih = oh / scale_;
                    int iw = ow / scale_;

                    int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;

                    out_ptr[out_idx] = input[in_idx];
                }
            }
        }
    }

    return out_ptr;
}

float *UpsampleCPU::backward(const float *grad_output)
{
    int batch = cached_batch_;
    int channels = cached_channels_;
    int in_h = cached_in_h_;
    int in_w = cached_in_w_;
    int out_h = get_output_height(in_h);
    int out_w = get_output_width(in_w);

    size_t grad_input_size = static_cast<size_t>(batch) * channels * in_h * in_w;

    // Allocate gradient input buffer if needed
    if (grad_input_size > grad_input_buffer_size_)
    {
        delete[] grad_input_buffer_;
        grad_input_buffer_ = new float[grad_input_size];
        grad_input_buffer_size_ = grad_input_size;
    }

    // Zero initialize
    std::memset(grad_input_buffer_, 0, grad_input_size * sizeof(float));

    // Sum gradients from all upsampled positions back to original
    for (int n = 0; n < batch; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int oh = 0; oh < out_h; ++oh)
            {
                for (int ow = 0; ow < out_w; ++ow)
                {
                    // Map output to input coordinates
                    int ih = oh / scale_;
                    int iw = ow / scale_;

                    int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;

                    grad_input_buffer_[in_idx] += grad_output[out_idx];
                }
            }
        }
    }

    return grad_input_buffer_;
}