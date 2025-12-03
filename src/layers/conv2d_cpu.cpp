#include "layers/conv2d_cpu.h"
#include <cstring>
#include <random>

Conv2DCPU::Conv2DCPU(int in_channels_num, int out_channels_num, int kernel_size, int stride, int padding)
    : in_c_(in_channels_num), out_c_(out_channels_num), k_size_(kernel_size), stride_(stride), pad_(padding)
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

    for (int i = 0; i < weight_size; ++i)
    {
        weights_[i] = dist(gen);
    }

    // Initialize biases to zero
    std::fill(bias_.begin(), bias_.end(), 0.0f);
}

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

Tensor Conv2DCPU::backward(const Tensor &grad_output)
{
    // Initialize gradient tensors
    Tensor grad_input(cached_input_.shape);
    float *grad_in_data = grad_input.data->data();
    const float *grad_out_data = grad_output.data->data();
    const float *in_data = cached_input_.data->data();

    // Zero initialize gradients
    std::memset(grad_in_data, 0, grad_input.size() * sizeof(float));
    std::memset(grad_w_.data(), 0, grad_w_.size() * sizeof(float));
    std::memset(grad_b_.data(), 0, grad_b_.size() * sizeof(float));

    int batch = cached_input_.batch();
    int in_h = cached_input_.height();
    int in_w = cached_input_.width();
    int out_h = grad_output.height();
    int out_w = grad_output.width();

    // Compute gradients
    for (int n = 0; n < batch; ++n)
    {
        for (int oc = 0; oc < out_c_; ++oc)
        {
            for (int oh = 0; oh < out_h; ++oh)
            {
                for (int ow = 0; ow < out_w; ++ow)
                {
                    int out_idx = ((n * out_c_ + oc) * out_h + oh) * out_w + ow;
                    float grad_out = grad_out_data[out_idx];
                    grad_b_[oc] += grad_out;

                    // Gradient w.r.t. weights and input
                    for (int ic = 0; ic < in_c_; ++ic)
                    {
                        for (int kh = 0; kh < k_size_; ++kh)
                        {
                            for (int kw = 0; kw < k_size_; ++kw)
                            {
                                int ih = oh * stride_ - pad_ + kh;
                                int iw = ow * stride_ - pad_ + kw;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                                {
                                    int in_idx = ((n * in_c_ + ic) * in_h + ih) * in_w + iw;
                                    int w_idx = ((oc * in_c_ + ic) * k_size_ + kh) * k_size_ + kw;

                                    grad_w_[w_idx] += grad_out * in_data[in_idx];
                                    grad_in_data[in_idx] += grad_out * weights_[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

void Conv2DCPU::set_weight(const std::vector<float> &weights, const std::vector<float> &bias)
{
    if (weights.size() != weights_.size())
    {
        throw std::runtime_error("Weight size mismatch");
    }
    if (bias.size() != bias_.size())
    {
        throw std::runtime_error("Bias size mismatch");
    }
    weights_ = weights;
    bias_ = bias;
}

void Conv2DCPU::get_weights(std::vector<float> &weights, std::vector<float> &bias) const
{
    weights = weights_;
    bias = bias_;
}

void Conv2DCPU::get_gradients(std::vector<float> &grad_w, std::vector<float> &grad_b)
{
    grad_w = grad_w_;
    grad_b = grad_b_;
}

void Conv2DCPU::update_weights(float learning_rate)
{
    // SGD update: weight -= learning_rate * gradient
    for (size_t i = 0; i < weights_.size(); ++i)
    {
        weights_[i] -= learning_rate * grad_w_[i];
    }

    for (size_t i = 0; i < bias_.size(); ++i)
    {
        bias_[i] -= learning_rate * grad_b_[i];
    }
}