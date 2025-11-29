/**
 * @file loss_functions.cpp
 * @brief Implementation of loss functions for neural network training
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#include "layers/loss_functions.h"
#include <stdexcept>
#include <cmath>

namespace LossFunctions {

float mse_loss(const Tensor& output, const Tensor& target) {
    /**
     * Mean Squared Error:
     *   MSE = (1/N) * sum((output[i] - target[i])^2)
     * 
     * where N is the total number of elements.
     */
    
    if (output.shape != target.shape) {
        throw std::invalid_argument("mse_loss: output and target shapes must match");
    }
    
    const float* out_data = output.data->data();
    const float* target_data = target.data->data();
    const size_t size = output.size();
    
    // Compute sum of squared differences
    double sum_sq = 0.0;
    
    #pragma omp parallel for reduction(+:sum_sq) schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float diff = out_data[i] - target_data[i];
        sum_sq += diff * diff;
    }
    
    // Return mean squared error
    return static_cast<float>(sum_sq / size);
}

Tensor mse_loss_gradient(const Tensor& output, const Tensor& target) {
    /**
     * Gradient of MSE loss w.r.t. output:
     *   dL/d_output = (2/N) * (output - target)
     * 
     * This is the gradient that gets backpropagated through the network.
     */
    
    if (output.shape != target.shape) {
        throw std::invalid_argument("mse_loss_gradient: output and target shapes must match");
    }
    
    Tensor gradient(output.shape);
    
    const float* out_data = output.data->data();
    const float* target_data = target.data->data();
    float* grad_data = gradient.data->data();
    const size_t size = output.size();
    
    // Scale factor: 2/N
    const float scale = 2.0f / size;
    
    // Compute gradient: (2/N) * (output - target)
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; ++i) {
        grad_data[i] = scale * (out_data[i] - target_data[i]);
    }
    
    return gradient;
}

float mse_loss_with_gradient(const Tensor& output, const Tensor& target, Tensor& gradient) {
    /**
     * Combined function to compute both loss and gradient in one pass.
     * This is more efficient as we only iterate through the data once.
     */
    
    if (output.shape != target.shape) {
        throw std::invalid_argument("mse_loss_with_gradient: output and target shapes must match");
    }
    
    // Allocate gradient tensor
    gradient = Tensor(output.shape);
    
    const float* out_data = output.data->data();
    const float* target_data = target.data->data();
    float* grad_data = gradient.data->data();
    const size_t size = output.size();
    
    // Scale factor for gradient: 2/N
    const float grad_scale = 2.0f / size;
    
    // Compute loss and gradient in one pass
    double sum_sq = 0.0;
    
    #pragma omp parallel for reduction(+:sum_sq) schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float diff = out_data[i] - target_data[i];
        sum_sq += diff * diff;
        grad_data[i] = grad_scale * diff;
    }
    
    // Return mean squared error
    return static_cast<float>(sum_sq / size);
}

} // namespace LossFunctions
