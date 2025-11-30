#pragma once
#include "data/data_types.h"

class MaxPoolCPU
{
public:
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);

private:
    Tensor cached_input_;
    int pool_size_ = 2;
    std::vector<int> max_indices_;
};