#pragma once
#include "data/data_types.h"

class Conv2DCPU
{
public:
    Conv2DCPU(int in_channels_num, int out_channels_num, int kernel_size, int stride = 1, int padding = 0);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &input);
    void set_weight(const std::vector<float> &weights, const std::vector<float> &bias);
    void get_gradients(std::vector<float> &grad_w, std::vector<float> &grad_b);
    void update_weights(float learning_rate);

private:
    int in_c_, out_c_, k_size_, stride_, pad_;
    std::vector<float> weights_;
    std::vector<float> bias_;
    std::vector<float> grad_w_, grad_b_;

    Tensor cached_input_; // for backward
};