#include "layers/maxpool_cpu.h"
#include <limits>

Tensor MaxPoolCPU::forward(const Tensor &input)
{
    cached_input_ = input;

    int batch = input.batch();
    int channels = input.channels();
    int in_h = input.height();
    int in_w = input.width();

    int out_h = in_h / pool_size_;
    int out_w = in_w / pool_size_;

    Tensor output({batch, channels, out_h, out_w});
    max_indices_.resize(output.size()); // store indices for backward pass

    const float *in_data = input.data->data();
    float *out_data = output.data->data();

#pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int oh = 0; oh < out_h; ++oh)
            {
                for (int ow = 0; ow < out_w; ++ow)
                {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = 0;

                    for (int ph = 0; ph < pool_size_; ++ph)
                    {
                        for (int pw = 0; pw < pool_size_; ++pw)
                        {
                            int ih = oh * pool_size_ + ph;
                            int iw = ow * pool_size_ + pw;
                            int in_idx = ((n * channels + c) * (in_h + ih) * in_w + iw);
                            if (in_data[in_idx] > max_val)
                            {
                                max_val = in_data[in_idx];
                                max_idx = in_idx;
                            }
                        }
                    }

                    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = max_val;
                    max_indices_[out_idx] = max_idx;
                }
            }
        }
    }

    return output;
}

Tensor MaxPoolCPU::backward(const Tensor &grad_output)
{
    Tensor grad_input = grad_output;
}