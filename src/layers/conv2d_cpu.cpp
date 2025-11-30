#include "layers/conv2d_cpu.h";

Tensor Conv2DCPU::forward(const Tensor &input)
{
    cached_input_ = input;

    int batch = input.batch();
    int in_h = input.height();
    int in_w = input.width();

    int out_h = (in_h + 2 * pad_ - k_size_) / stride_ + 1;
    int out_w = (in_w + 2 * pad_ - k_size_) / stride_ + 1;

    Tensor output({batch, out_c_, out_h, out_w});

    float *out_data = output.data->data();
    const float *in_data = input.data->data();

    // OpenMP, parallel over CPU cores, combines first 2 loops (rule-of-thumb)

#pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n)
    {
        for (int oc = 0; oc < out_c_; ++oc)
        {
            for (int oh = 0; oh < out_h; ++oh)
            {
                for (int ow = 0; ow < out_w; ++ow)
                {
                    float sum = bias_[oc];

                    // Convolution operation
                    for (int ic = 0; ic < in_c_; ++ic)
                    {
                        for (int kh = 0; kh < k_size_; ++kh)
                        {
                            for (int kw = 0; kw < k_size_; ++kw)
                            {
                                int ih = oh * stride_ - pad_ + kh;
                                int iw = ow * stride_ - pad_ + kw;

                                // Handle padding
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                                {
                                    int in_idx = ((n * in_c_ + ic) * in_h + ih) * in_w + iw;
                                    int w_idx = ((oc * in_c_ + ic) * k_size_ + kh) * k_size_ + kw;
                                    sum += in_data[in_idx] * weights_[w_idx];
                                }
                            }
                        }
                    }

                    int out_idx = ((n * out_c_ + oc) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = sum;
                }
            }
        }
    }

    return output;
}