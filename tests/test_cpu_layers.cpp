/**
 * @file test_cpu_layers.cpp
 * @brief Test suite for CPU neural network layers
 * 
 * This test verifies:
 * - Conv2D forward and backward pass
 * - ReLU activation
 * - MaxPooling
 * - Upsampling
 * - MSE loss
 * - Autoencoder forward pass
 * - Shape consistency throughout the network
 * 
 * Usage:
 *   ./test_cpu_layers
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#include "layers/conv2d_cpu.h"
#include "layers/relu_cpu.h"
#include "layers/maxpool_cpu.h"
#include "layers/upsample_cpu.h"
#include "layers/loss_functions.h"
#include "models/autoencoder_cpu.h"
#include "data/data_types.h"
#include "utils/logger.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <cstdio>

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

void test_pass(const std::string& name) {
    std::cout << "  ✓ " << name << std::endl;
    tests_passed++;
}

void test_fail(const std::string& name, const std::string& reason) {
    std::cout << "  ✗ " << name << " - " << reason << std::endl;
    tests_failed++;
}

/**
 * @brief Create a random tensor for testing
 */
Tensor create_random_tensor(const std::vector<int>& shape, unsigned int seed = 42) {
    Tensor t(shape);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float* data = t.data->data();
    for (size_t i = 0; i < t.size(); ++i) {
        data[i] = dist(rng);
    }
    
    return t;
}

/**
 * @brief Test 1: Conv2D layer
 */
void test_conv2d() {
    std::cout << "\n=== Test 1: Conv2D Layer ===" << std::endl;
    
    try {
        // Create a Conv2D layer: 3 -> 64 channels, 3x3 kernel, padding=1
        Conv2DCPU conv(3, 64, 3, 1, 1);
        
        // Create input tensor [batch=2, channels=3, height=32, width=32]
        Tensor input = create_random_tensor({2, 3, 32, 32});
        
        // Forward pass
        Tensor output = conv.forward(input);
        
        // Check output shape
        if (output.batch() == 2 && output.channels() == 64 &&
            output.height() == 32 && output.width() == 32) {
            test_pass("Conv2D output shape correct [2, 64, 32, 32]");
        } else {
            test_fail("Conv2D shape", "Expected [2, 64, 32, 32], got [" +
                      std::to_string(output.batch()) + ", " +
                      std::to_string(output.channels()) + ", " +
                      std::to_string(output.height()) + ", " +
                      std::to_string(output.width()) + "]");
        }
        
        // Test backward pass
        Tensor grad_output = create_random_tensor({2, 64, 32, 32});
        Tensor grad_input = conv.backward(grad_output);
        
        if (grad_input.batch() == 2 && grad_input.channels() == 3 &&
            grad_input.height() == 32 && grad_input.width() == 32) {
            test_pass("Conv2D backward shape correct");
        } else {
            test_fail("Conv2D backward shape", "Shape mismatch");
        }
        
    } catch (const std::exception& e) {
        test_fail("Conv2D test", e.what());
    }
}

/**
 * @brief Test 2: ReLU layer
 */
void test_relu() {
    std::cout << "\n=== Test 2: ReLU Layer ===" << std::endl;
    
    try {
        ReLUCPU relu;
        
        // Create input with some negative values
        Tensor input({2, 3, 4, 4});
        float* data = input.data->data();
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (size_t i = 0; i < input.size(); ++i) {
            data[i] = dist(rng);
        }
        
        // Forward pass
        Tensor output = relu.forward(input);
        
        // Check all values are >= 0
        float* out_data = output.data->data();
        bool all_positive = true;
        for (size_t i = 0; i < output.size(); ++i) {
            if (out_data[i] < 0) {
                all_positive = false;
                break;
            }
        }
        
        if (all_positive) {
            test_pass("ReLU output all non-negative");
        } else {
            test_fail("ReLU output", "Found negative values");
        }
        
        // Check shape preserved
        if (output.shape == input.shape) {
            test_pass("ReLU preserves shape");
        } else {
            test_fail("ReLU shape", "Shape changed");
        }
        
        // Test backward pass
        Tensor grad_output = create_random_tensor({2, 3, 4, 4});
        Tensor grad_input = relu.backward(grad_output);
        
        if (grad_input.shape == input.shape) {
            test_pass("ReLU backward shape correct");
        } else {
            test_fail("ReLU backward", "Shape mismatch");
        }
        
    } catch (const std::exception& e) {
        test_fail("ReLU test", e.what());
    }
}

/**
 * @brief Test 3: MaxPool layer
 */
void test_maxpool() {
    std::cout << "\n=== Test 3: MaxPool Layer ===" << std::endl;
    
    try {
        MaxPoolCPU pool(2);  // 2x2 pooling
        
        // Create input [2, 3, 16, 16]
        Tensor input = create_random_tensor({2, 3, 16, 16});
        
        // Forward pass
        Tensor output = pool.forward(input);
        
        // Check output shape (should be halved)
        if (output.batch() == 2 && output.channels() == 3 &&
            output.height() == 8 && output.width() == 8) {
            test_pass("MaxPool output shape correct [2, 3, 8, 8]");
        } else {
            test_fail("MaxPool shape", "Expected [2, 3, 8, 8], got [" +
                      std::to_string(output.batch()) + ", " +
                      std::to_string(output.channels()) + ", " +
                      std::to_string(output.height()) + ", " +
                      std::to_string(output.width()) + "]");
        }
        
        // Test backward pass
        Tensor grad_output = create_random_tensor({2, 3, 8, 8});
        Tensor grad_input = pool.backward(grad_output);
        
        if (grad_input.batch() == 2 && grad_input.channels() == 3 &&
            grad_input.height() == 16 && grad_input.width() == 16) {
            test_pass("MaxPool backward shape correct");
        } else {
            test_fail("MaxPool backward", "Shape mismatch");
        }
        
    } catch (const std::exception& e) {
        test_fail("MaxPool test", e.what());
    }
}

/**
 * @brief Test 4: Upsample layer
 */
void test_upsample() {
    std::cout << "\n=== Test 4: Upsample Layer ===" << std::endl;
    
    try {
        UpsampleCPU upsample(2);  // 2x upsampling
        
        // Create input [2, 3, 8, 8]
        Tensor input = create_random_tensor({2, 3, 8, 8});
        
        // Forward pass
        Tensor output = upsample.forward(input);
        
        // Check output shape (should be doubled)
        if (output.batch() == 2 && output.channels() == 3 &&
            output.height() == 16 && output.width() == 16) {
            test_pass("Upsample output shape correct [2, 3, 16, 16]");
        } else {
            test_fail("Upsample shape", "Expected [2, 3, 16, 16]");
        }
        
        // Verify nearest neighbor interpolation
        // Each input pixel should appear 4 times in output
        const float* in_data = input.data->data();
        const float* out_data = output.data->data();
        
        // Check first input pixel appears in 2x2 block of output
        bool correct_upscale = true;
        float first_val = in_data[0];
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                int out_idx = dy * 16 + dx;
                if (std::abs(out_data[out_idx] - first_val) > 1e-6) {
                    correct_upscale = false;
                }
            }
        }
        
        if (correct_upscale) {
            test_pass("Upsample values correctly replicated");
        } else {
            test_fail("Upsample values", "Values not correctly replicated");
        }
        
        // Test backward pass
        Tensor grad_output = create_random_tensor({2, 3, 16, 16});
        Tensor grad_input = upsample.backward(grad_output);
        
        if (grad_input.batch() == 2 && grad_input.channels() == 3 &&
            grad_input.height() == 8 && grad_input.width() == 8) {
            test_pass("Upsample backward shape correct");
        } else {
            test_fail("Upsample backward", "Shape mismatch");
        }
        
    } catch (const std::exception& e) {
        test_fail("Upsample test", e.what());
    }
}

/**
 * @brief Test 5: MSE Loss
 */
void test_mse_loss() {
    std::cout << "\n=== Test 5: MSE Loss ===" << std::endl;
    
    try {
        // Create output and target tensors
        Tensor output = create_random_tensor({2, 3, 32, 32});
        Tensor target = create_random_tensor({2, 3, 32, 32}, 123);  // Different seed
        
        // Compute loss
        float loss = LossFunctions::mse_loss(output, target);
        
        // Loss should be positive
        if (loss > 0) {
            test_pass("MSE loss is positive");
        } else {
            test_fail("MSE loss", "Loss should be positive");
        }
        
        // Compute gradient
        Tensor gradient = LossFunctions::mse_loss_gradient(output, target);
        
        // Check gradient shape
        if (gradient.shape == output.shape) {
            test_pass("MSE gradient shape correct");
        } else {
            test_fail("MSE gradient", "Shape mismatch");
        }
        
        // Test combined function
        Tensor combined_grad;
        float combined_loss = LossFunctions::mse_loss_with_gradient(output, target, combined_grad);
        
        if (std::abs(combined_loss - loss) < 1e-5) {
            test_pass("Combined loss matches separate computation");
        } else {
            test_fail("Combined loss", "Loss values differ");
        }
        
        // Test loss is zero when output equals target
        float zero_loss = LossFunctions::mse_loss(output, output);
        if (std::abs(zero_loss) < 1e-6) {
            test_pass("MSE loss is zero for identical tensors");
        } else {
            test_fail("Zero loss", "Loss not zero for identical inputs");
        }
        
    } catch (const std::exception& e) {
        test_fail("MSE loss test", e.what());
    }
}

/**
 * @brief Test 6: Autoencoder architecture
 */
void test_autoencoder() {
    std::cout << "\n=== Test 6: Autoencoder ===" << std::endl;
    
    try {
        AutoencoderCPU model;
        
        // Check parameter count
        size_t num_params = model.get_num_parameters();
        if (num_params > 700000 && num_params < 800000) {
            test_pass("Parameter count in expected range (~751,875)");
            std::cout << "    Actual: " << num_params << std::endl;
        } else {
            test_fail("Parameter count", 
                      "Expected ~751,875, got " + std::to_string(num_params));
        }
        
        // Create input
        Tensor input = create_random_tensor({4, 3, 32, 32});
        
        // Forward pass
        Tensor output = model.forward(input);
        
        // Check output shape matches input (autoencoder reconstructs input)
        if (output.batch() == 4 && output.channels() == 3 &&
            output.height() == 32 && output.width() == 32) {
            test_pass("Autoencoder output shape matches input [4, 3, 32, 32]");
        } else {
            test_fail("Autoencoder output shape", "Shape mismatch");
        }
        
        // Test feature extraction
        Tensor features = model.extract_features(input);
        
        // Check latent shape [N, 128, 8, 8]
        if (features.batch() == 4 && features.channels() == 128 &&
            features.height() == 8 && features.width() == 8) {
            test_pass("Latent features shape correct [4, 128, 8, 8]");
        } else {
            test_fail("Latent shape", "Expected [4, 128, 8, 8]");
        }
        
        // Check latent dimension
        if (model.get_latent_dim() == 8192) {
            test_pass("Latent dimension correct (8192)");
        } else {
            test_fail("Latent dimension", 
                      "Expected 8192, got " + std::to_string(model.get_latent_dim()));
        }
        
    } catch (const std::exception& e) {
        test_fail("Autoencoder test", e.what());
    }
}

/**
 * @brief Test 7: Training step
 */
void test_training_step() {
    std::cout << "\n=== Test 7: Training Step ===" << std::endl;
    
    try {
        AutoencoderCPU model;
        
        // Create a small batch
        Tensor input = create_random_tensor({2, 3, 32, 32});
        
        // Forward pass
        Tensor output = model.forward(input);
        
        // Initial loss
        float initial_loss = model.compute_loss(output, input);
        
        if (initial_loss > 0) {
            test_pass("Initial loss is positive");
        } else {
            test_fail("Initial loss", "Expected positive loss");
        }
        
        // Training step (forward + backward + update)
        float loss1 = model.backward(input, 0.001f);
        
        // Second forward pass
        output = model.forward(input);
        float loss2 = model.compute_loss(output, input);
        
        // After training, loss should typically decrease
        // (not guaranteed for single step, but good indicator)
        std::cout << "    Loss before: " << loss1 << std::endl;
        std::cout << "    Loss after:  " << loss2 << std::endl;
        
        // Just verify no crash or NaN
        if (!std::isnan(loss2) && !std::isinf(loss2)) {
            test_pass("Training step completes without errors");
        } else {
            test_fail("Training step", "Loss is NaN or Inf");
        }
        
    } catch (const std::exception& e) {
        test_fail("Training step test", e.what());
    }
}

/**
 * @brief Test 8: Weight save/load
 */
void test_weight_io() {
    std::cout << "\n=== Test 8: Weight Save/Load ===" << std::endl;
    
    try {
        AutoencoderCPU model1;
        
        // Save weights to temp directory (use current directory for portability)
        std::string path = "test_weights_temp.bin";
        model1.save_weights(path);
        
        test_pass("Weights saved successfully");
        
        // Create new model and load weights
        AutoencoderCPU model2;
        model2.load_weights(path);
        
        test_pass("Weights loaded successfully");
        
        // Verify outputs match
        Tensor input = create_random_tensor({2, 3, 32, 32});
        
        Tensor out1 = model1.forward(input);
        Tensor out2 = model2.forward(input);
        
        // Check outputs are identical
        float diff = 0.0f;
        const float* d1 = out1.data->data();
        const float* d2 = out2.data->data();
        for (size_t i = 0; i < out1.size(); ++i) {
            diff += std::abs(d1[i] - d2[i]);
        }
        
        if (diff < 1e-5) {
            test_pass("Loaded weights produce same output");
        } else {
            test_fail("Weight verification", 
                      "Output difference: " + std::to_string(diff));
        }
        
        // Cleanup temp file
        std::remove(path.c_str());
        
    } catch (const std::exception& e) {
        test_fail("Weight I/O test", e.what());
    }
}

int main() {
    LOG_INIT();
    
    std::cout << "========================================" << std::endl;
    std::cout << "CPU Layers Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Run all tests
    test_conv2d();
    test_relu();
    test_maxpool();
    test_upsample();
    test_mse_loss();
    test_autoencoder();
    test_training_step();
    test_weight_io();
    
    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    
    if (tests_failed == 0) {
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ Some tests failed!" << std::endl;
        return 1;
    }
}
