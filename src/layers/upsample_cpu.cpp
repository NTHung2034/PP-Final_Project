#include "layers/upsample_cpu.h"
#include <cstring>

UpsampleCPU::UpsampleCPU(int scale) : cached_input_(), scale_(scale) {}
Tensor UpsampleCPU::forward(const Tensor &input)
{
    cached_input_ = input;

    int batch = input.batch();
    int channels = input.channels();
    int in_h = input.height();
    int in_w = input.width();

    int out_h = in_h * scale_;
    int out_w = in_w * scale_;

    Tensor output({batch, channels, out_h, out_w});

    const float *in_data = input.data->data();
    float *out_data = output.data->data();

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

                    out_data[out_idx] = in_data[in_idx];
                }
            }
        }
    }

    return output;
}

Tensor UpsampleCPU::backward(const Tensor &grad_output)
{
    // Create gradient tensor with same shape as input
    int batch = cached_input_.batch();
    int channels = cached_input_.channels();
    int in_h = cached_input_.height();
    int in_w = cached_input_.width();

    Tensor grad_input({batch, channels, in_h, in_w});
    float *grad_in_data = grad_input.data->data();
    const float *grad_out_data = grad_output.data->data();

    // Zero initialize
    std::memset(grad_in_data, 0, grad_input.size() * sizeof(float));

    int out_h = in_h * scale_;
    int out_w = in_w * scale_;

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

                    grad_in_data[in_idx] += grad_out_data[out_idx];
                }
            }
        }
    }

    return grad_input;
}