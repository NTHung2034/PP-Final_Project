/**
 * @file loss_functions.h
 * @brief Loss functions for neural network training
 * 
 * This file implements loss functions used for training the autoencoder.
 * The primary loss function is Mean Squared Error (MSE) for reconstruction.
 * 
 * MSE Loss:
 *   L = (1/N) * sum((output - target)^2)
 * 
 * MSE Gradient:
 *   dL/d_output = (2/N) * (output - target)
 * 
 * Reference:
 * - https://en.wikipedia.org/wiki/Mean_squared_error
 * - Deep Learning Book, Chapter 3: Loss Functions
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#pragma once

#include "data/data_types.h"

/**
 * @namespace LossFunctions
 * @brief Collection of loss functions for training
 */
namespace LossFunctions {

/**
 * @brief Compute Mean Squared Error loss
 * 
 * MSE = (1/N) * sum((output[i] - target[i])^2)
 * 
 * @param output Predicted output tensor
 * @param target Target tensor (same shape as output)
 * @return float Scalar MSE loss value
 */
float mse_loss(const Tensor& output, const Tensor& target);

/**
 * @brief Compute gradient of MSE loss w.r.t. output
 * 
 * dL/d_output = (2/N) * (output - target)
 * 
 * @param output Predicted output tensor
 * @param target Target tensor (same shape as output)
 * @return Tensor Gradient tensor (same shape as output)
 */
Tensor mse_loss_gradient(const Tensor& output, const Tensor& target);

/**
 * @brief Combined function: compute both loss and gradient
 * 
 * More efficient than calling mse_loss and mse_loss_gradient separately.
 * 
 * @param output   Predicted output tensor
 * @param target   Target tensor
 * @param gradient Output: gradient tensor (allocated inside)
 * @return float   MSE loss value
 */
float mse_loss_with_gradient(const Tensor& output, const Tensor& target, Tensor& gradient);

} // namespace LossFunctions
