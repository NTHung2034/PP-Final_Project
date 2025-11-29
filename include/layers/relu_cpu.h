/**
 * @file relu_cpu.h
 * @brief CPU implementation of ReLU (Rectified Linear Unit) activation
 * 
 * ReLU is the most commonly used activation function in deep neural networks.
 * It introduces non-linearity while being computationally efficient.
 * 
 * Mathematical operation:
 *   ReLU(x) = max(0, x)
 * 
 * Gradient:
 *   d_ReLU/dx = 1 if x > 0, else 0
 * 
 * Properties:
 * - Computationally efficient (simple comparison and multiplication)
 * - Helps mitigate vanishing gradient problem
 * - Can cause "dying ReLU" problem (neurons stuck at 0)
 * 
 * Reference:
 * - Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines"
 * - https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#pragma once

#include "data/data_types.h"

/**
 * @class ReLUCPU
 * @brief CPU implementation of ReLU activation with forward and backward pass
 * 
 * Usage:
 *   ReLUCPU relu;
 *   Tensor output = relu.forward(input);
 *   Tensor grad_input = relu.backward(grad_output);
 */
class ReLUCPU {
public:
    /**
     * @brief Default constructor
     */
    ReLUCPU() : cached_input_({1, 1, 1, 1}) {}
    
    /**
     * @brief Forward pass: apply ReLU activation
     * 
     * @param input Input tensor of any shape
     * @return Tensor Output tensor with same shape, ReLU applied element-wise
     */
    Tensor forward(const Tensor& input);
    
    /**
     * @brief Backward pass: compute gradient w.r.t. input
     * 
     * The gradient is passed through where input was positive,
     * and blocked where input was zero or negative.
     * 
     * @param grad_output Gradient from next layer
     * @return Tensor Gradient w.r.t. input (same shape as input)
     */
    Tensor backward(const Tensor& grad_output);
    
private:
    /**
     * @brief Cached input from forward pass (needed for backward)
     * 
     * We cache the input to determine which elements had positive values.
     */
    Tensor cached_input_;
};
