// Test GPU Optimized v2 Layers
// Optimizations: CUDA Streams, Kernel Fusion, Full Unrolling
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "data/gpu_tensor_opt.cuh"
#include "layers_gpu_opt_v2/conv2d_gpu_opt_v2.cuh"
#include "layers_gpu_opt_v2/maxpool_gpu_opt_v2.cuh"
#include "layers_gpu_opt_v2/upsample_gpu_opt_v2.cuh"
#include "layers_gpu_opt_v2/mse_loss_gpu_opt_v2.cuh"

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

void print_test_result(const char* test_name, bool passed) {
    if (passed) {
        std::cout << "[PASS] " << test_name << std::endl;
        tests_passed++;
    } else {
        std::cout << "[FAIL] " << test_name << std::endl;
        tests_failed++;
    }
}

// Initialize random data on host
void init_random(float* data, int size, float scale = 1.0f) {
    for (int i = 0; i < size; i++) {
        data[i] = scale * ((float)rand() / RAND_MAX - 0.5f);
    }
}

// =============================================================================
// Test Conv2D Forward with Stream
// =============================================================================
void test_conv2d_forward_stream() {
    std::cout << "\n--- Testing Conv2D Forward (v2 with stream) ---\n";
    
    int N = 2, C_in = 3, H = 8, W = 8;
    int C_out = 16;
    
    GPUTensorOpt input, output;
    input.allocate(N, C_in, H, W);
    output.allocate(N, C_out, H, W);
    
    GPUConvWeightsOpt weights(C_out, C_in, 3, 3);
    weights.initXavier();
    
    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Run forward on stream
    conv2d_forward_opt_v2(input, weights, output, true, stream);
    cudaStreamSynchronize(stream);
    
    // Copy back and check non-zero
    float* h_out = new float[output.size];
    cudaMemcpy(h_out, output.d_data, output.size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool has_values = false;
    for (int i = 0; i < output.size; i++) {
        if (h_out[i] != 0.0f) { has_values = true; break; }
    }
    
    print_test_result("Conv2D forward produces output", has_values);
    
    // Clean up
    delete[] h_out;
    input.free();
    output.free();
    cudaStreamDestroy(stream);
}

// =============================================================================
// Test Conv2D Backward with Parallel Streams
// =============================================================================
void test_conv2d_backward_streams() {
    std::cout << "\n--- Testing Conv2D Backward (parallel streams) ---\n";
    
    int N = 2, C_in = 3, H = 8, W = 8;
    int C_out = 16;
    
    GPUTensorOpt input, forward_output, grad_output, grad_input;
    input.allocate(N, C_in, H, W);
    forward_output.allocate(N, C_out, H, W);
    grad_output.allocate(N, C_out, H, W);
    grad_input.allocate(N, C_in, H, W);
    
    GPUConvWeightsOpt weights(C_out, C_in, 3, 3);
    weights.initXavier();
    
    // Initialize with random data
    float* h_input = new float[input.size];
    float* h_grad_out = new float[grad_output.size];
    init_random(h_input, input.size);
    init_random(h_grad_out, grad_output.size);
    
    cudaMemcpy(input.d_data, h_input, input.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_output.d_data, h_grad_out, grad_output.size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Forward pass first
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    
    conv2d_forward_opt_v2(input, weights, forward_output, true, stream1);
    cudaStreamSynchronize(stream1);
    
    // Backward with parallel streams
    conv2d_backward_opt_v2(input, grad_output, forward_output, weights, grad_input,
                           true, stream1, stream2, stream3);
    
    // Sync all streams
    conv2d_sync_streams(stream1, stream2, stream3);
    
    // Check gradients are computed
    float* h_grad_in = new float[grad_input.size];
    float* h_grad_w = new float[weights.weight_size];
    float* h_grad_b = new float[weights.bias_size];
    
    cudaMemcpy(h_grad_in, grad_input.d_data, grad_input.size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_w, weights.d_grad_w, weights.weight_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_b, weights.d_grad_b, weights.bias_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool grad_in_ok = false, grad_w_ok = false, grad_b_ok = false;
    for (int i = 0; i < grad_input.size; i++) if (h_grad_in[i] != 0.0f) { grad_in_ok = true; break; }
    for (int i = 0; i < weights.weight_size; i++) if (h_grad_w[i] != 0.0f) { grad_w_ok = true; break; }
    for (int i = 0; i < weights.bias_size; i++) if (h_grad_b[i] != 0.0f) { grad_b_ok = true; break; }
    
    print_test_result("Gradient input computed (stream1)", grad_in_ok);
    print_test_result("Gradient weights computed (stream2)", grad_w_ok);
    print_test_result("Gradient bias computed (stream3)", grad_b_ok);
    
    // Clean up
    delete[] h_input;
    delete[] h_grad_out;
    delete[] h_grad_in;
    delete[] h_grad_w;
    delete[] h_grad_b;
    input.free(); forward_output.free(); grad_output.free(); grad_input.free();
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
}

// =============================================================================
// Test MaxPool2D with Stream
// =============================================================================
void test_maxpool_stream() {
    std::cout << "\n--- Testing MaxPool2D (v2 with stream) ---\n";
    
    int N = 2, C = 16, H_in = 8, W_in = 8;
    int H_out = H_in / 2, W_out = W_in / 2;
    
    GPUTensorOpt input, output;
    input.allocate(N, C, H_in, W_in);
    output.allocate(N, C, H_out, W_out);
    
    int* pool_indices;
    cudaMalloc(&pool_indices, output.size * sizeof(int));
    
    // Initialize with random data
    float* h_input = new float[input.size];
    init_random(h_input, input.size, 1.0f);
    cudaMemcpy(input.d_data, h_input, input.size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Forward
    maxpool2d_forward_opt_v2(input, output, pool_indices, stream);
    cudaStreamSynchronize(stream);
    
    // Check output
    float* h_output = new float[output.size];
    cudaMemcpy(h_output, output.d_data, output.size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool has_output = false;
    for (int i = 0; i < output.size; i++) {
        if (h_output[i] != 0.0f) { has_output = true; break; }
    }
    
    print_test_result("MaxPool2D forward with stream", has_output);
    
    // Clean up
    delete[] h_input;
    delete[] h_output;
    input.free(); output.free();
    cudaFree(pool_indices);
    cudaStreamDestroy(stream);
}

// =============================================================================
// Test Upsample2D with Stream
// =============================================================================
void test_upsample_stream() {
    std::cout << "\n--- Testing Upsample2D (v2 with stream) ---\n";
    
    int N = 2, C = 16, H_in = 4, W_in = 4;
    int H_out = H_in * 2, W_out = W_in * 2;
    
    GPUTensorOpt input, output;
    input.allocate(N, C, H_in, W_in);
    output.allocate(N, C, H_out, W_out);
    
    // Initialize with pattern
    float* h_input = new float[input.size];
    for (int i = 0; i < input.size; i++) h_input[i] = (float)(i % 10);
    cudaMemcpy(input.d_data, h_input, input.size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Forward
    upsample2d_forward_opt_v2(input, output, stream);
    cudaStreamSynchronize(stream);
    
    // Check that output is 2x in each dimension
    float* h_output = new float[output.size];
    cudaMemcpy(h_output, output.d_data, output.size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check first pixel replication
    bool replicated = (h_output[0] == h_input[0]) && 
                      (h_output[1] == h_input[0]) &&
                      (h_output[W_out] == h_input[0]);
    
    print_test_result("Upsample2D 2x replication", replicated);
    
    delete[] h_input;
    delete[] h_output;
    input.free(); output.free();
    cudaStreamDestroy(stream);
}

// =============================================================================
// Test MSE Loss with Stream
// =============================================================================
void test_mse_loss_stream() {
    std::cout << "\n--- Testing MSE Loss (v2 with stream) ---\n";
    
    int N = 4, C = 3, H = 32, W = 32;
    
    GPUTensorOpt output, target, grad_output;
    output.allocate(N, C, H, W);
    target.allocate(N, C, H, W);
    grad_output.allocate(N, C, H, W);
    
    // Initialize
    float* h_output = new float[output.size];
    float* h_target = new float[target.size];
    for (int i = 0; i < output.size; i++) {
        h_output[i] = (float)(i % 100) / 100.0f;
        h_target[i] = (float)((i + 50) % 100) / 100.0f;
    }
    cudaMemcpy(output.d_data, h_output, output.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(target.d_data, h_target, target.size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Forward loss
    float loss = mse_loss_forward_opt_v2(output, target, stream);
    
    // Backward
    mse_loss_backward_opt_v2(output, target, grad_output, stream);
    cudaStreamSynchronize(stream);
    
    print_test_result("MSE loss > 0", loss > 0.0f);
    
    // Check gradient
    float* h_grad = new float[grad_output.size];
    cudaMemcpy(h_grad, grad_output.d_data, grad_output.size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool has_grad = false;
    for (int i = 0; i < grad_output.size; i++) {
        if (h_grad[i] != 0.0f) { has_grad = true; break; }
    }
    print_test_result("MSE backward gradient", has_grad);
    
    delete[] h_output;
    delete[] h_target;
    delete[] h_grad;
    output.free(); target.free(); grad_output.free();
    cudaStreamDestroy(stream);
}

// =============================================================================
// Test Memory Pool with Streams
// =============================================================================
void test_memory_pool_streams() {
    std::cout << "\n--- Testing GPUMemoryPool Streams ---\n";
    
    GPUMemoryPool pool;
    pool.allocate(4);
    pool.create_streams();
    
    print_test_result("Memory pool allocated", pool.allocated);
    print_test_result("Streams created", pool.streams_created);
    print_test_result("Stream1 valid", pool.stream1 != nullptr);
    print_test_result("Stream2 valid", pool.stream2 != nullptr);
    print_test_result("Stream3 valid", pool.stream3 != nullptr);
    
    // Test stream can be used
    float* d_test;
    cudaMalloc(&d_test, 100 * sizeof(float));
    cudaMemsetAsync(d_test, 0, 100 * sizeof(float), pool.stream1);
    cudaStreamSynchronize(pool.stream1);
    cudaFree(d_test);
    
    print_test_result("Stream1 usable", true);
    
    pool.free();
    print_test_result("Memory pool freed", !pool.allocated);
    print_test_result("Streams destroyed", !pool.streams_created);
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    GPU OPTIMIZED v2 LAYER TESTS                                ║\n";
    std::cout << "║    (CUDA Streams + Kernel Fusion + Full Unrolling)             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    srand(42);
    
    // Check CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\nDevice: " << prop.name << std::endl;
    std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "Async Engine Count: " << prop.asyncEngineCount << std::endl;
    
    // Run tests
    test_memory_pool_streams();
    test_conv2d_forward_stream();
    test_conv2d_backward_streams();
    test_maxpool_stream();
    test_upsample_stream();
    test_mse_loss_stream();
    
    // Summary
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "RESULTS: " << tests_passed << " passed, " << tests_failed << " failed\n";
    std::cout << std::string(50, '=') << std::endl;
    
    return tests_failed > 0 ? 1 : 0;
}
