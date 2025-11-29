/**
 * @file relu_cpu.cpp
 * @brief CPU implementation of ReLU activation layer
 * 
 * This is a simple but highly optimized implementation of ReLU.
 * The forward pass applies max(0, x) element-wise.
 * The backward pass passes gradients only where input was positive.
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#include "layers/relu_cpu.h"
#include <algorithm>
#include <cstring>

Tensor ReLUCPU::forward(const Tensor& input) {
    /**
     * Forward pass: ReLU(x) = max(0, x)
     * 
     * We create a copy of the input and apply ReLU in-place.
     * The original input is cached for the backward pass.
     */
    
    // Cache input for backward pass
    cached_input_ = Tensor(input.shape);
    std::copy(input.data->data(), 
              input.data->data() + input.size(),
              cached_input_.data->data());
    
    // Create output tensor (copy of input)
    Tensor output(input.shape);
    
    const float* in_data = input.data->data();
    float* out_data = output.data->data();
    const size_t size = input.size();
    
    // Apply ReLU: out = max(0, in)
    // Parallelized with SIMD for better performance
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; ++i) {
        out_data[i] = std::max(0.0f, in_data[i]);
    }
    
    return output;
}

Tensor ReLUCPU::backward(const Tensor& grad_output) {
    /**
     * Backward pass: d_ReLU/dx = 1 if x > 0, else 0
     * 
     * The gradient is passed through where the input was positive.
     * Where the input was zero or negative, the gradient is zeroed out.
     * 
     * grad_input = grad_output * (input > 0)
     */
    
    // Create gradient tensor
    Tensor grad_input(grad_output.shape);
    
    const float* cached_in_data = cached_input_.data->data();
    const float* grad_out_data = grad_output.data->data();
    float* grad_in_data = grad_input.data->data();
    const size_t size = grad_output.size();
    
    // Compute gradient: pass through if input > 0
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; ++i) {
        // grad_input = grad_output * indicator(cached_input > 0)
        grad_in_data[i] = (cached_in_data[i] > 0.0f) ? grad_out_data[i] : 0.0f;
    }
    
    return grad_input;
}
