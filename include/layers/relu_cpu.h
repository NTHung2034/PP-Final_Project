#pragma once
#include "data/data_types.h"

class ReLUCPU
{
public:
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_ouput);

private:
    Tensor cached_input_; // for backward
};