#pragma once
#include "data/data_types.h"
#include <vector>

class MaxPoolCPU
{
public:
    MaxPoolCPU(int pool_size = 2);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);

private:
    Tensor cached_input_;
    int pool_size_;
    std::vector<int> max_indices_;
};