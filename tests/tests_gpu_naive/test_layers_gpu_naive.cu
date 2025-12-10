#include "layers_gpu_naive/conv2d_gpu_naive.cuh"
#include "layers_gpu_naive/relu_gpu_naive.cuh"
#include "layers_gpu_naive/maxpool_gpu_naive.cuh"
#include "layers_gpu_naive/upsample_gpu_naive.cuh"
#include "layers_gpu_naive/mse_loss_gpu_naive.cuh"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

#define TEST_ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "❌ TEST FAILED: " << message << std::endl; \
        return false; \
    }

void print_separator() {
    std::cout << std::string(70, '-') << std::endl;
}

void print_test_header(const std::string& test_name) {
    std::cout << "\n";
    print_separator();
    std::cout << "Testing: " << test_name << std::endl;
    print_separator();
}

// Test Conv2D Forward
bool test_conv2d_forward_naive() {
    print_test_header("Conv2D Forward (Naive GPU)");
    
    // Create small test tensors
    int batch = 2;
    int in_c = 3, out_c = 4;
    int h = 8, w = 8;
    int kh = 3, kw = 3;
    
    GPUTensor input(batch, in_c, h, w, false);
    GPUTensor output(batch, out_c, h, w, false);
    GPUConvWeights weights(out_c, in_c, kh, kw);
    
    // Initialize with simple values
    for (int i = 0; i < input.size; i++) {
        input.h_data[i] = 0.1f * (i % 10);
    }
    weights.initializeXavier();
    
    input.copyToDevice();
    
    // Run forward pass
    conv2d_forward_gpu_naive(input, weights, output, kh, kw, 1, 1, false);
    
    output.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check output is not all zeros
    bool has_nonzero = false;
    for (int i = 0; i < output.size; i++) {
        if (std::abs(output.h_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    
    TEST_ASSERT(has_nonzero, "Output should contain non-zero values");
    
    std::cout << "✓ Conv2D forward produces valid output" << std::endl;
    std::cout << "  Input: [" << batch << "," << in_c << "," << h << "," << w << "]" << std::endl;
    std::cout << "  Output: [" << batch << "," << out_c << "," << h << "," << w << "]" << std::endl;
    std::cout << "  Sample output[0]: " << output.h_data[0] << std::endl;
    
    return true;
}

// Test ReLU Forward
bool test_relu_forward_naive() {
    print_test_header("ReLU Forward (Naive GPU)");
    
    GPUTensor input(1, 1, 4, 4, false);
    
    // Set test values (mix of positive and negative)
    float test_data[] = {
        -1.0f, 2.0f, -3.0f, 4.0f,
        5.0f, -6.0f, 7.0f, -8.0f,
        -9.0f, 10.0f, -11.0f, 12.0f,
        13.0f, -14.0f, 15.0f, -16.0f
    };
    
    for (int i = 0; i < 16; i++) {
        input.h_data[i] = test_data[i];
    }
    
    input.copyToDevice();
    
    // Apply ReLU
    relu_forward_gpu_naive(input);
    
    input.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Verify: negative values should be 0, positive unchanged
    for (int i = 0; i < 16; i++) {
        float expected = test_data[i] > 0 ? test_data[i] : 0.0f;
        TEST_ASSERT(std::abs(input.h_data[i] - expected) < 1e-5f,
                   "ReLU output mismatch at index " + std::to_string(i));
    }
    
    std::cout << "✓ ReLU correctly zeros negative values" << std::endl;
    std::cout << "  Input: [-1, 2, -3, 4, ...]" << std::endl;
    std::cout << "  Output: [0, 2, 0, 4, ...]" << std::endl;
    
    return true;
}

// Test MaxPool Forward
bool test_maxpool_forward_naive() {
    print_test_header("MaxPool2D Forward (Naive GPU)");
    
    GPUTensor input(1, 1, 4, 4, false);
    GPUTensor output(1, 1, 2, 2, false);
    
    int* pool_indices;
    CUDA_CHECK(cudaMalloc(&pool_indices, output.size * sizeof(int)));
    
    // Set test pattern
    float test_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    for (int i = 0; i < 16; i++) {
        input.h_data[i] = test_data[i];
    }
    
    input.copyToDevice();
    
    // Run maxpool (2x2, stride 2)
    maxpool2d_forward_gpu_naive(input, output, pool_indices);
    
    output.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Expected maxpool output: [6, 8, 14, 16]
    float expected[] = {6.0f, 8.0f, 14.0f, 16.0f};
    
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(std::abs(output.h_data[i] - expected[i]) < 1e-5f,
                   "MaxPool output mismatch at index " + std::to_string(i));
    }
    
    std::cout << "✓ MaxPool2D correctly computes max values" << std::endl;
    std::cout << "  Pool size: 2x2, Stride: 2" << std::endl;
    std::cout << "  Output: [" << output.h_data[0] << ", " << output.h_data[1] << ", "
              << output.h_data[2] << ", " << output.h_data[3] << "]" << std::endl;
    
    CUDA_CHECK(cudaFree(pool_indices));
    return true;
}

// Test Upsample Forward
bool test_upsample_forward_naive() {
    print_test_header("Upsample2D Forward (Naive GPU)");
    
    GPUTensor input(1, 1, 2, 2, false);
    GPUTensor output(1, 1, 4, 4, false);
    
    // Set test values
    input.h_data[0] = 1.0f;
    input.h_data[1] = 2.0f;
    input.h_data[2] = 3.0f;
    input.h_data[3] = 4.0f;
    
    input.copyToDevice();
    
    // Upsample 2x
    upsample2d_forward_gpu_naive(input, output, 2);
    
    output.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Expected: each value duplicated in 2x2 blocks
    float expected[] = {
        1.0f, 1.0f, 2.0f, 2.0f,
        1.0f, 1.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
        3.0f, 3.0f, 4.0f, 4.0f
    };
    
    for (int i = 0; i < 16; i++) {
        TEST_ASSERT(std::abs(output.h_data[i] - expected[i]) < 1e-5f,
                   "Upsample output mismatch at index " + std::to_string(i));
    }
    
    std::cout << "✓ Upsample2D correctly duplicates values" << std::endl;
    std::cout << "  Scale factor: 2x" << std::endl;
    std::cout << "  Input [2x2] -> Output [4x4]" << std::endl;
    
    return true;
}

// Test MSE Loss
bool test_mse_loss_naive() {
    print_test_header("MSE Loss (Naive GPU)");
    
    GPUTensor pred(1, 1, 2, 2, false);
    GPUTensor target(1, 1, 2, 2, false);
    
    // Set values
    pred.h_data[0] = 1.0f; target.h_data[0] = 1.5f;  // diff = -0.5, sq = 0.25
    pred.h_data[1] = 2.0f; target.h_data[1] = 2.0f;  // diff = 0, sq = 0
    pred.h_data[2] = 3.0f; target.h_data[2] = 2.0f;  // diff = 1.0, sq = 1.0
    pred.h_data[3] = 4.0f; target.h_data[3] = 5.0f;  // diff = -1.0, sq = 1.0
    
    pred.copyToDevice();
    target.copyToDevice();
    
    // Compute MSE
    float loss = mse_loss_forward_gpu_naive(pred, target);
    
    // Expected: (0.25 + 0 + 1.0 + 1.0) / 4 = 0.5625
    float expected_loss = 0.5625f;
    
    TEST_ASSERT(std::abs(loss - expected_loss) < 1e-4f,
               "MSE loss mismatch. Expected: " + std::to_string(expected_loss) + 
               ", Got: " + std::to_string(loss));
    
    std::cout << "✓ MSE loss correctly computed" << std::endl;
    std::cout << "  Loss: " << loss << " (expected: " << expected_loss << ")" << std::endl;
    
    return true;
}

// Test Conv2D + ReLU Backward
bool test_conv2d_relu_backward_naive() {
    print_test_header("Conv2D + ReLU Backward (Naive GPU)");
    
    int batch = 1, in_c = 2, out_c = 3;
    int h = 4, w = 4;
    
    GPUTensor input(batch, in_c, h, w, false);
    GPUTensor output(batch, out_c, h, w, false);
    GPUTensor grad_output(batch, out_c, h, w, false);
    GPUTensor grad_input(batch, in_c, h, w, false);
    GPUConvWeights weights(out_c, in_c, 3, 3);
    
    // Initialize
    for (int i = 0; i < input.size; i++) input.h_data[i] = 0.1f;
    for (int i = 0; i < grad_output.size; i++) grad_output.h_data[i] = 1.0f;
    weights.initializeXavier();
    
    input.copyToDevice();
    grad_output.copyToDevice();
    
    // Forward + backward
    conv2d_forward_gpu_naive(input, weights, output, 3, 3, 1, 1, true);
    conv2d_backward_gpu_naive(input, grad_output, weights, grad_input, 3, 3, 1, 1);
    
    grad_input.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check gradients exist
    bool has_gradient = false;
    for (int i = 0; i < grad_input.size; i++) {
        if (std::abs(grad_input.h_data[i]) > 1e-6f) {
            has_gradient = true;
            break;
        }
    }
    
    TEST_ASSERT(has_gradient, "Backward pass should produce gradients");
    
    std::cout << "✓ Conv2D backward produces valid gradients" << std::endl;
    std::cout << "  Sample grad_input[0]: " << grad_input.h_data[0] << std::endl;
    
    return true;
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         NAIVE GPU LAYERS TEST SUITE - PHASE 2                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "\nUsing GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n" << std::endl;
    
    int passed = 0, total = 0;
    
    // Run tests
    #define RUN_TEST(test_func) \
        total++; \
        if (test_func()) { \
            passed++; \
            std::cout << "✅ PASSED\n"; \
        } else { \
            std::cout << "❌ FAILED\n"; \
        }
    
    RUN_TEST(test_conv2d_forward_naive);
    RUN_TEST(test_relu_forward_naive);
    RUN_TEST(test_maxpool_forward_naive);
    RUN_TEST(test_upsample_forward_naive);
    RUN_TEST(test_mse_loss_naive);
    RUN_TEST(test_conv2d_relu_backward_naive);
    
    // Summary
    std::cout << "\n";
    print_separator();
    std::cout << "TEST SUMMARY" << std::endl;
    print_separator();
    std::cout << "Total: " << total << " | Passed: " << passed << " | Failed: " << (total - passed) << std::endl;
    
    if (passed == total) {
        std::cout << "\n🎉 ALL TESTS PASSED! 🎉\n" << std::endl;
        return 0;
    } else {
        std::cout << "\n⚠️  SOME TESTS FAILED ⚠️\n" << std::endl;
        return 1;
    }
}
