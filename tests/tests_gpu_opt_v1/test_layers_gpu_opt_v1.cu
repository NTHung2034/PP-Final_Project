// Test suite for GPU Optimized v1 Layers
#include "data/gpu_tensor_opt.cuh"
#include "layers_gpu_opt_v1/conv2d_gpu_opt_v1.cuh"
#include "layers_gpu_opt_v1/relu_gpu_opt_v1.cuh"
#include "layers_gpu_opt_v1/maxpool_gpu_opt_v1.cuh"
#include "layers_gpu_opt_v1/upsample_gpu_opt_v1.cuh"
#include "layers_gpu_opt_v1/mse_loss_gpu_opt_v1.cuh"

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

// Helper: copy data to device tensor
void copy_to_device(GPUTensorOpt& tensor, const float* h_data) {
    CUDA_CHECK(cudaMemcpy(tensor.d_data, h_data, tensor.size * sizeof(float), cudaMemcpyHostToDevice));
}

// Helper: copy data from device tensor
void copy_to_host(const GPUTensorOpt& tensor, float* h_data) {
    CUDA_CHECK(cudaMemcpy(h_data, tensor.d_data, tensor.size * sizeof(float), cudaMemcpyDeviceToHost));
}

// Test Conv2D Forward with Shared Memory Tiling
bool test_conv2d_forward_opt() {
    print_test_header("Conv2D Forward (Tiled + Shared Memory)");
    
    int batch = 2, in_c = 3, out_c = 4;
    int h = 8, w = 8;
    
    GPUTensorOpt input, output;
    input.allocate(batch, in_c, h, w);
    output.allocate(batch, out_c, h, w);
    
    GPUConvWeightsOpt weights(out_c, in_c, 3, 3);
    weights.initXavier();
    
    // Initialize input
    float* h_input = new float[input.size];
    for (int i = 0; i < input.size; i++) {
        h_input[i] = 0.1f * (i % 10);
    }
    copy_to_device(input, h_input);
    
    // Run forward (with ReLU by default)
    conv2d_forward_opt_v1(input, weights, output, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get output
    float* h_output = new float[output.size];
    copy_to_host(output, h_output);
    
    // Check output has non-zero values
    bool has_nonzero = false;
    for (int i = 0; i < output.size; i++) {
        if (std::abs(h_output[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    
    TEST_ASSERT(has_nonzero, "Output should contain non-zero values");
    
    std::cout << "✓ Conv2D (tiled) forward produces valid output" << std::endl;
    std::cout << "  Input: [" << batch << "," << in_c << "," << h << "," << w << "]" << std::endl;
    std::cout << "  Output: [" << batch << "," << out_c << "," << h << "," << w << "]" << std::endl;
    std::cout << "  Sample output[0]: " << h_output[0] << std::endl;
    
    delete[] h_input;
    delete[] h_output;
    input.free();
    output.free();
    
    return true;
}

// Test ReLU Forward
bool test_relu_forward_opt() {
    print_test_header("ReLU Forward (Optimized v1)");
    
    GPUTensorOpt tensor;
    tensor.allocate(1, 1, 4, 4);
    
    float test_data[] = {
        -1.0f, 2.0f, -3.0f, 4.0f,
        5.0f, -6.0f, 7.0f, -8.0f,
        -9.0f, 10.0f, -11.0f, 12.0f,
        13.0f, -14.0f, 15.0f, -16.0f
    };
    
    copy_to_device(tensor, test_data);
    
    // Apply ReLU (in-place)
    relu_forward_opt_v1(tensor);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float* h_output = new float[16];
    copy_to_host(tensor, h_output);
    
    // Verify
    for (int i = 0; i < 16; i++) {
        float expected = test_data[i] > 0 ? test_data[i] : 0.0f;
        TEST_ASSERT(std::abs(h_output[i] - expected) < 1e-5f,
                   "ReLU output mismatch at index " + std::to_string(i));
    }
    
    std::cout << "✓ ReLU correctly zeros negative values" << std::endl;
    std::cout << "  Input: [-1, 2, -3, 4, ...]" << std::endl;
    std::cout << "  Output: [0, 2, 0, 4, ...]" << std::endl;
    
    delete[] h_output;
    tensor.free();
    
    return true;
}

// Test MaxPool Forward
bool test_maxpool_forward_opt() {
    print_test_header("MaxPool2D Forward (Optimized v1)");
    
    GPUTensorOpt input, output;
    input.allocate(1, 1, 4, 4);
    output.allocate(1, 1, 2, 2);
    
    int* pool_indices;
    CUDA_CHECK(cudaMalloc(&pool_indices, output.size * sizeof(int)));
    
    float test_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    copy_to_device(input, test_data);
    
    // Run maxpool
    maxpool2d_forward_opt_v1(input, output, pool_indices);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float* h_output = new float[4];
    copy_to_host(output, h_output);
    
    // Expected: [6, 8, 14, 16]
    float expected[] = {6.0f, 8.0f, 14.0f, 16.0f};
    
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(std::abs(h_output[i] - expected[i]) < 1e-5f,
                   "MaxPool output mismatch at index " + std::to_string(i));
    }
    
    std::cout << "✓ MaxPool2D correctly computes max values" << std::endl;
    std::cout << "  Pool size: 2x2, Stride: 2" << std::endl;
    std::cout << "  Output: [" << h_output[0] << ", " << h_output[1] << ", "
              << h_output[2] << ", " << h_output[3] << "]" << std::endl;
    
    delete[] h_output;
    CUDA_CHECK(cudaFree(pool_indices));
    input.free();
    output.free();
    
    return true;
}

// Test Upsample Forward
bool test_upsample_forward_opt() {
    print_test_header("Upsample2D Forward (Optimized v1)");
    
    GPUTensorOpt input, output;
    input.allocate(1, 1, 2, 2);
    output.allocate(1, 1, 4, 4);
    
    float test_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    copy_to_device(input, test_data);
    
    // Upsample 2x
    upsample2d_forward_opt_v1(input, output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float* h_output = new float[16];
    copy_to_host(output, h_output);
    
    // Expected: each value duplicated in 2x2 blocks
    float expected[] = {
        1.0f, 1.0f, 2.0f, 2.0f,
        1.0f, 1.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
        3.0f, 3.0f, 4.0f, 4.0f
    };
    
    for (int i = 0; i < 16; i++) {
        TEST_ASSERT(std::abs(h_output[i] - expected[i]) < 1e-5f,
                   "Upsample output mismatch at index " + std::to_string(i));
    }
    
    std::cout << "✓ Upsample2D correctly duplicates values" << std::endl;
    std::cout << "  Scale factor: 2x" << std::endl;
    std::cout << "  Input [2x2] -> Output [4x4]" << std::endl;
    
    delete[] h_output;
    input.free();
    output.free();
    
    return true;
}

// Test MSE Loss
bool test_mse_loss_opt() {
    print_test_header("MSE Loss (Optimized v1)");
    
    GPUTensorOpt pred, target;
    pred.allocate(1, 1, 2, 2);
    target.allocate(1, 1, 2, 2);
    
    float pred_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float target_data[] = {1.5f, 2.0f, 2.0f, 5.0f};
    // diff = [-0.5, 0, 1.0, -1.0], sq = [0.25, 0, 1.0, 1.0]
    // MSE = (0.25 + 0 + 1.0 + 1.0) / 4 = 0.5625
    
    copy_to_device(pred, pred_data);
    copy_to_device(target, target_data);
    
    float loss = mse_loss_forward_opt_v1(pred, target);
    
    float expected_loss = 0.5625f;
    
    TEST_ASSERT(std::abs(loss - expected_loss) < 1e-4f,
               "MSE loss mismatch. Expected: " + std::to_string(expected_loss) + 
               ", Got: " + std::to_string(loss));
    
    std::cout << "✓ MSE loss correctly computed" << std::endl;
    std::cout << "  Loss: " << loss << " (expected: " << expected_loss << ")" << std::endl;
    
    pred.free();
    target.free();
    
    return true;
}

// Test Conv2D Backward
bool test_conv2d_backward_opt() {
    print_test_header("Conv2D Backward (Tiled + Shared Memory)");
    
    int batch = 1, in_c = 2, out_c = 3;
    int h = 4, w = 4;
    
    GPUTensorOpt input, output, grad_output, grad_input;
    input.allocate(batch, in_c, h, w);
    output.allocate(batch, out_c, h, w);
    grad_output.allocate(batch, out_c, h, w);
    grad_input.allocate(batch, in_c, h, w);
    
    GPUConvWeightsOpt weights(out_c, in_c, 3, 3);
    weights.initXavier();
    
    // Initialize
    float* h_input = new float[input.size];
    float* h_grad_out = new float[grad_output.size];
    for (int i = 0; i < input.size; i++) h_input[i] = 0.1f;
    for (int i = 0; i < grad_output.size; i++) h_grad_out[i] = 1.0f;
    
    copy_to_device(input, h_input);
    copy_to_device(grad_output, h_grad_out);
    
    // Forward then backward
    conv2d_forward_opt_v1(input, weights, output, true);
    conv2d_backward_opt_v1(input, grad_output, weights, grad_input);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get gradients
    float* h_grad_input = new float[grad_input.size];
    copy_to_host(grad_input, h_grad_input);
    
    bool has_gradient = false;
    for (int i = 0; i < grad_input.size; i++) {
        if (std::abs(h_grad_input[i]) > 1e-6f) {
            has_gradient = true;
            break;
        }
    }
    
    TEST_ASSERT(has_gradient, "Backward pass should produce gradients");
    
    std::cout << "✓ Conv2D backward produces valid gradients" << std::endl;
    std::cout << "  Sample grad_input[0]: " << h_grad_input[0] << std::endl;
    
    delete[] h_input;
    delete[] h_grad_out;
    delete[] h_grad_input;
    input.free();
    output.free();
    grad_output.free();
    grad_input.free();
    
    return true;
}

// Test Memory Pool Allocation
bool test_memory_pool() {
    print_test_header("GPU Memory Pool");
    
    GPUMemoryPool pool;
    
    TEST_ASSERT(!pool.allocated, "Pool should not be allocated initially");
    
    // Allocate for batch size 4
    pool.allocate(4);
    
    TEST_ASSERT(pool.allocated, "Pool should be allocated after allocate()");
    TEST_ASSERT(pool.batch_size == 4, "Pool batch size should be 4");
    
    // Check tensor dimensions
    TEST_ASSERT(pool.act1.batch == 4 && pool.act1.channels == 256 && pool.act1.height == 32,
               "act1 dimensions incorrect");
    TEST_ASSERT(pool.act4.batch == 4 && pool.act4.channels == 128 && pool.act4.height == 8,
               "act4 (latent) dimensions incorrect");
    
    // Test re-allocation with same batch size (should be no-op)
    float* old_ptr = pool.act1.d_data;
    pool.allocate(4);
    TEST_ASSERT(pool.act1.d_data == old_ptr, "Re-allocation with same batch size should be no-op");
    
    std::cout << "✓ Memory pool allocates correctly" << std::endl;
    std::cout << "  Batch size: 4" << std::endl;
    std::cout << "  Forward buffers: act1-act8, output" << std::endl;
    std::cout << "  Backward buffers: grad1-grad8, grad_out, grad_in" << std::endl;
    std::cout << "  Pooling indices: pool1_idx, pool2_idx" << std::endl;
    
    // Cleanup happens in destructor
    return true;
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║      OPTIMIZED v1 GPU LAYERS TEST SUITE - PHASE 3             ║\n";
    std::cout << "║  (Memory Pool + Shared Memory Tiling + Constant Memory)       ║\n";
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
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n" << std::endl;
    
    int passed = 0, total = 0;
    
    #define RUN_TEST(test_func) \
        total++; \
        if (test_func()) { \
            passed++; \
            std::cout << "✅ PASSED\n"; \
        } else { \
            std::cout << "❌ FAILED\n"; \
        }
    
    RUN_TEST(test_memory_pool);
    RUN_TEST(test_conv2d_forward_opt);
    RUN_TEST(test_relu_forward_opt);
    RUN_TEST(test_maxpool_forward_opt);
    RUN_TEST(test_upsample_forward_opt);
    RUN_TEST(test_mse_loss_opt);
    RUN_TEST(test_conv2d_backward_opt);
    
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
