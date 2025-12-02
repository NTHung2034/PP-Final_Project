#include "data/gpu_data_types.cuh"
#include "layers_gpu/conv2d_gpu.cuh"
#include "layers_gpu/relu_gpu.cuh"
#include "layers_gpu/maxpool_gpu.cuh"
#include "layers_gpu/upsample_gpu.cuh"
#include "layers_gpu/mse_loss_gpu.cuh"

#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdlib>

// Helper to check approximate equality
bool approx_equal(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// =============================================================================
// Conv2D Tests
// =============================================================================

bool test_conv2d_forward_shape() {
    std::cout << "Test: Conv2D GPU forward shape... ";
    
    GPUTensor input(1, 3, 32, 32);
    for (size_t i = 0; i < input.size; ++i) input.h_data[i] = 0.5f;
    input.copyToDevice();
    
    GPUConvWeights weights(64, 3, 3, 3);
    weights.initializeXavier();
    
    GPUTensor output(1, 64, 32, 32);
    
    conv2d_forward_gpu(input, weights, output, 3, 3, 1, 1, false);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    output.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    bool success = (output.batch == 1 && output.channels == 64 &&
                    output.height == 32 && output.width == 32);
    
    float sum = 0.0f;
    for (size_t i = 0; i < output.size; ++i) sum += std::abs(output.h_data[i]);
    success = success && (sum > 0.0f);
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

bool test_conv2d_relu_fusion() {
    std::cout << "Test: Conv2D GPU with fused ReLU... ";
    
    GPUTensor input(1, 3, 16, 16);
    for (size_t i = 0; i < input.size; ++i) 
        input.h_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    input.copyToDevice();
    
    GPUConvWeights weights(32, 3, 3, 3);
    weights.initializeXavier();
    
    GPUTensor output(1, 32, 16, 16);
    
    conv2d_forward_gpu(input, weights, output, 3, 3, 1, 1, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    output.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    bool success = true;
    for (size_t i = 0; i < output.size; ++i) {
        if (output.h_data[i] < 0.0f) { success = false; break; }
    }
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// =============================================================================
// ReLU Tests
// =============================================================================

bool test_relu_forward() {
    std::cout << "Test: ReLU GPU forward... ";
    
    GPUTensor data(1, 1, 2, 2);
    data.h_data[0] = -1.0f; data.h_data[1] = 2.0f;
    data.h_data[2] = -0.5f; data.h_data[3] = 3.0f;
    data.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    relu_forward_gpu(data);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    data.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    bool success = (approx_equal(data.h_data[0], 0.0f) &&
                    approx_equal(data.h_data[1], 2.0f) &&
                    approx_equal(data.h_data[2], 0.0f) &&
                    approx_equal(data.h_data[3], 3.0f));
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

bool test_relu_backward() {
    std::cout << "Test: ReLU GPU backward... ";
    
    GPUTensor input(1, 1, 2, 2);
    input.h_data[0] = -1.0f; input.h_data[1] = 2.0f;
    input.h_data[2] = -0.5f; input.h_data[3] = 3.0f;
    input.copyToDevice();
    
    GPUTensor grad_output(1, 1, 2, 2);
    for (int i = 0; i < 4; ++i) grad_output.h_data[i] = 1.0f;
    grad_output.copyToDevice();
    
    GPUTensor grad_input(1, 1, 2, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    relu_backward_gpu(input, grad_output, grad_input);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    grad_input.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    bool success = (approx_equal(grad_input.h_data[0], 0.0f) &&
                    approx_equal(grad_input.h_data[1], 1.0f) &&
                    approx_equal(grad_input.h_data[2], 0.0f) &&
                    approx_equal(grad_input.h_data[3], 1.0f));
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// =============================================================================
// MaxPool Tests
// =============================================================================

bool test_maxpool_forward() {
    std::cout << "Test: MaxPool2D GPU forward... ";
    
    GPUTensor input(1, 1, 4, 4);
    for (int i = 0; i < 16; ++i) input.h_data[i] = (float)i;
    input.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    GPUTensor output(1, 1, 2, 2);
    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, 4 * sizeof(int)));
    
    maxpool2d_forward_gpu(input, output, d_indices);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    output.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Max of 2x2 blocks: [5, 7, 13, 15]
    bool success = (approx_equal(output.h_data[0], 5.0f) &&
                    approx_equal(output.h_data[1], 7.0f) &&
                    approx_equal(output.h_data[2], 13.0f) &&
                    approx_equal(output.h_data[3], 15.0f));
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    
    CUDA_CHECK(cudaFree(d_indices));
    return success;
}

bool test_maxpool_backward() {
    std::cout << "Test: MaxPool2D GPU backward... ";
    
    GPUTensor input(1, 1, 4, 4);
    for (int i = 0; i < 16; ++i) input.h_data[i] = (float)i;
    input.copyToDevice();
    
    GPUTensor output(1, 1, 2, 2);
    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, 4 * sizeof(int)));
    
    maxpool2d_forward_gpu(input, output, d_indices);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    GPUTensor grad_output(1, 1, 2, 2);
    for (int i = 0; i < 4; ++i) grad_output.h_data[i] = 1.0f;
    grad_output.copyToDevice();
    
    GPUTensor grad_input(1, 1, 4, 4);
    
    maxpool2d_backward_gpu(grad_output, d_indices, grad_input);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    grad_input.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int non_zero = 0;
    for (size_t i = 0; i < 16; ++i)
        if (grad_input.h_data[i] != 0.0f) non_zero++;
    
    bool success = (non_zero == 4);
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    
    CUDA_CHECK(cudaFree(d_indices));
    return success;
}

// =============================================================================
// Upsample Tests
// =============================================================================

bool test_upsample_forward() {
    std::cout << "Test: Upsample2D GPU forward... ";
    
    GPUTensor input(1, 1, 2, 2);
    input.h_data[0] = 1.0f; input.h_data[1] = 2.0f;
    input.h_data[2] = 3.0f; input.h_data[3] = 4.0f;
    input.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    GPUTensor output(1, 1, 4, 4);
    
    upsample2d_forward_gpu(input, output, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    output.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    bool shape_ok = (output.height == 4 && output.width == 4);
    bool values_ok = (approx_equal(output.h_data[0], 1.0f) &&
                      approx_equal(output.h_data[1], 1.0f) &&
                      approx_equal(output.h_data[2], 2.0f));
    
    bool success = shape_ok && values_ok;
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

bool test_upsample_backward() {
    std::cout << "Test: Upsample2D GPU backward... ";
    
    GPUTensor grad_output(1, 1, 4, 4);
    for (int i = 0; i < 16; ++i) grad_output.h_data[i] = 1.0f;
    grad_output.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    GPUTensor grad_input(1, 1, 2, 2);
    
    upsample2d_backward_gpu(grad_output, grad_input, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    grad_input.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    bool success = true;
    for (int i = 0; i < 4; ++i) {
        if (!approx_equal(grad_input.h_data[i], 4.0f)) {
            success = false; break;
        }
    }
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// =============================================================================
// MSE Loss Tests
// =============================================================================

bool test_mse_loss_zero() {
    std::cout << "Test: MSE Loss GPU (zero loss)... ";
    
    GPUTensor pred(1, 1, 2, 2);
    GPUTensor target(1, 1, 2, 2);
    
    for (int i = 0; i < 4; ++i) {
        pred.h_data[i] = (float)i;
        target.h_data[i] = (float)i;
    }
    pred.copyToDevice();
    target.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float loss = mse_loss_forward_gpu(pred, target);
    
    bool success = approx_equal(loss, 0.0f);
    std::cout << (success ? "PASSED" : "FAILED") 
              << " (loss=" << loss << ")" << std::endl;
    return success;
}

bool test_mse_loss_value() {
    std::cout << "Test: MSE Loss GPU (computed value)... ";
    
    GPUTensor pred(1, 1, 2, 2);
    GPUTensor target(1, 1, 2, 2);
    
    for (int i = 0; i < 4; ++i) {
        pred.h_data[i] = 0.0f;
        target.h_data[i] = 1.0f;
    }
    pred.copyToDevice();
    target.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float loss = mse_loss_forward_gpu(pred, target);
    
    // MSE = (1+1+1+1)/4 = 1.0
    bool success = approx_equal(loss, 1.0f);
    std::cout << (success ? "PASSED" : "FAILED")
              << " (expected=1.0, got=" << loss << ")" << std::endl;
    return success;
}

bool test_mse_loss_backward() {
    std::cout << "Test: MSE Loss GPU backward... ";
    
    GPUTensor pred(1, 1, 2, 2);
    GPUTensor target(1, 1, 2, 2);
    GPUTensor grad(1, 1, 2, 2);
    
    pred.h_data[0] = 1.0f; pred.h_data[1] = 2.0f;
    pred.h_data[2] = 3.0f; pred.h_data[3] = 4.0f;
    for (int i = 0; i < 4; ++i) target.h_data[i] = 0.0f;
    
    pred.copyToDevice();
    target.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    mse_loss_backward_gpu(pred, target, grad);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    grad.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // grad = 2*(pred-target)/N = [0.5, 1.0, 1.5, 2.0]
    bool success = (approx_equal(grad.h_data[0], 0.5f) &&
                    approx_equal(grad.h_data[1], 1.0f) &&
                    approx_equal(grad.h_data[2], 1.5f) &&
                    approx_equal(grad.h_data[3], 2.0f));
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    srand(42);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "\n========================================\n";
    std::cout << "  GPU Layer Unit Tests\n";
    std::cout << "========================================\n";
    std::cout << "  Device: " << prop.name << "\n";
    std::cout << "========================================\n\n";
    
    int passed = 0, total = 0;
    
    std::cout << "--- Conv2D ---\n";
    total++; if (test_conv2d_forward_shape()) passed++;
    total++; if (test_conv2d_relu_fusion()) passed++;
    
    std::cout << "\n--- ReLU ---\n";
    total++; if (test_relu_forward()) passed++;
    total++; if (test_relu_backward()) passed++;
    
    std::cout << "\n--- MaxPool2D ---\n";
    total++; if (test_maxpool_forward()) passed++;
    total++; if (test_maxpool_backward()) passed++;
    
    std::cout << "\n--- Upsample2D ---\n";
    total++; if (test_upsample_forward()) passed++;
    total++; if (test_upsample_backward()) passed++;
    
    std::cout << "\n--- MSE Loss ---\n";
    total++; if (test_mse_loss_zero()) passed++;
    total++; if (test_mse_loss_value()) passed++;
    total++; if (test_mse_loss_backward()) passed++;
    
    std::cout << "\n========================================\n";
    std::cout << "  Results: " << passed << "/" << total << " tests passed\n";
    std::cout << "========================================\n\n";
    
    cudaDeviceReset();
    return (passed == total) ? 0 : 1;
}
