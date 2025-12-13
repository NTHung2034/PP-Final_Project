#include "layers/relu_cpu.h"

Tensor ReLUCPU::forward(const Tensor &input)
{
    cached_input_ = input;
    Tensor output = input;

    float *data = output.data->data();
    size_t size = output.size();

    for (size_t i = 0; i < size; ++i)
    {
        data[i] = std::max(0.0f, data[i]);
    }

    return output;
}

Tensor ReLUCPU::backward(const Tensor &grad_output)
{
    Tensor grad_input = grad_output;

    const float *input_data = cached_input_.data->data();
    float *grad_data = grad_input.data->data();

    size_t size = grad_input.size();
    for (size_t i = 0; i < size; ++i)
    {
        grad_data[i] = (input_data[i] > 0.0f) ? grad_data[i] : 0.0f;
    }

    return grad_input;
}