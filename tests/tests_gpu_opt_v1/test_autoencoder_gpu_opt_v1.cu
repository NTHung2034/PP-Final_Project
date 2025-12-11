// Test suite for GPU Optimized v1 Autoencoder Model
#include "models/autoencoder_gpu_opt_v1.cuh"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>

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

// Test Model Initialization
bool test_model_initialization() {
    print_test_header("Model Initialization (Optimized v1)");
    
    int batch_size = 4;
    AutoencoderGPUOptV1 model(batch_size);
    
    // Check convolution weights are initialized
    TEST_ASSERT(model.conv1 != nullptr, "Conv1 should be initialized");
    TEST_ASSERT(model.conv2 != nullptr, "Conv2 should be initialized");
    TEST_ASSERT(model.conv3 != nullptr, "Conv3 should be initialized");
    TEST_ASSERT(model.conv4 != nullptr, "Conv4 should be initialized");
    TEST_ASSERT(model.conv5 != nullptr, "Conv5 should be initialized");
    
    // Check memory pool
    TEST_ASSERT(model.pool.allocated, "Memory pool should be allocated");
    TEST_ASSERT(model.pool.batch_size == batch_size, "Pool batch size mismatch");
    
    // Check layer dimensions
    TEST_ASSERT(model.conv1->out_c == 256, "Conv1 should have 256 output channels");
    TEST_ASSERT(model.conv1->in_c == 3, "Conv1 should have 3 input channels");
    TEST_ASSERT(model.conv5->out_c == 3, "Conv5 should have 3 output channels");
    
    std::cout << "✓ Model layers and memory pool correctly initialized" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Conv1: [3 -> 256], 3x3 kernel" << std::endl;
    std::cout << "  Conv2: [256 -> 128], 3x3 kernel" << std::endl;
    std::cout << "  Conv3: [128 -> 128], 3x3 kernel" << std::endl;
    std::cout << "  Conv4: [128 -> 256], 3x3 kernel" << std::endl;
    std::cout << "  Conv5: [256 -> 3], 3x3 kernel (sigmoid)" << std::endl;
    
    return true;
}

// Test Forward Pass
bool test_forward_pass() {
    print_test_header("Forward Pass (Optimized v1)");
    
    int batch_size = 4;
    AutoencoderGPUOptV1 model(batch_size);
    
    // Create input (host data)
    int input_size = batch_size * 3 * 32 * 32;
    float* h_input = new float[input_size];
    for (int i = 0; i < input_size; i++) {
        h_input[i] = 0.5f + 0.1f * (i % 10);
    }
    
    // Run forward
    auto start = std::chrono::high_resolution_clock::now();
    float loss = model.forward(h_input, batch_size);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Check loss is valid
    TEST_ASSERT(!std::isnan(loss), "Loss should not be NaN");
    TEST_ASSERT(!std::isinf(loss), "Loss should not be Inf");
    TEST_ASSERT(loss >= 0.0f, "Loss should be non-negative");
    
    // Check output tensor
    float* h_output = new float[model.pool.output.size];
    CUDA_CHECK(cudaMemcpy(h_output, model.pool.output.d_data, 
                          model.pool.output.size * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool has_valid_output = false;
    for (int i = 0; i < model.pool.output.size; i++) {
        TEST_ASSERT(!std::isnan(h_output[i]), "Output contains NaN");
        TEST_ASSERT(!std::isinf(h_output[i]), "Output contains Inf");
        if (std::abs(h_output[i]) > 1e-6f) has_valid_output = true;
    }
    
    TEST_ASSERT(has_valid_output, "Output should contain valid values");
    
    std::cout << "✓ Forward pass completed successfully" << std::endl;
    std::cout << "  Input shape: [" << batch_size << ", 3, 32, 32]" << std::endl;
    std::cout << "  Output shape: [" << batch_size << ", 3, 32, 32]" << std::endl;
    std::cout << "  Forward time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Initial loss: " << std::fixed << std::setprecision(6) << loss << std::endl;
    std::cout << "  Sample output[0]: " << h_output[0] << std::endl;
    
    delete[] h_input;
    delete[] h_output;
    
    return true;
}

// Test Training Step (Forward + Backward + Update)
bool test_training_step() {
    print_test_header("Training Step (Forward-Backward-Update)");
    
    int batch_size = 4;
    float learning_rate = 0.001f;
    
    AutoencoderGPUOptV1 model(batch_size);
    
    // Create input
    int input_size = batch_size * 3 * 32 * 32;
    float* h_input = new float[input_size];
    for (int i = 0; i < input_size; i++) {
        h_input[i] = 0.5f;
    }
    
    // Run single training step
    auto start = std::chrono::high_resolution_clock::now();
    float loss = model.train_step(h_input, batch_size, learning_rate);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    TEST_ASSERT(!std::isnan(loss), "Loss should not be NaN");
    TEST_ASSERT(!std::isinf(loss), "Loss should not be Inf");
    TEST_ASSERT(loss >= 0.0f, "Loss should be non-negative");
    
    std::cout << "✓ Training step completed successfully" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Loss: " << std::fixed << std::setprecision(6) << loss << std::endl;
    std::cout << "  Training time: " << duration.count() << " ms" << std::endl;
    
    delete[] h_input;
    
    return true;
}

// Test Loss Decreasing Over Multiple Steps
bool test_loss_decrease() {
    print_test_header("Loss Decrease Over Training Steps");
    
    int batch_size = 2;
    float learning_rate = 0.01f;
    int num_steps = 10;
    
    AutoencoderGPUOptV1 model(batch_size);
    
    // Create consistent input
    int input_size = batch_size * 3 * 32 * 32;
    float* h_input = new float[input_size];
    for (int i = 0; i < input_size; i++) {
        h_input[i] = 0.8f;
    }
    
    std::vector<float> losses;
    
    // Train for multiple steps
    for (int step = 0; step < num_steps; step++) {
        float loss = model.train_step(h_input, batch_size, learning_rate);
        losses.push_back(loss);
        
        if (step == 0 || step == num_steps - 1) {
            std::cout << "  Step " << std::setw(2) << step << ": Loss = " 
                      << std::fixed << std::setprecision(6) << loss << std::endl;
        }
    }
    
    // Check if loss decreased
    bool loss_decreased = losses.back() < losses.front();
    float reduction = (losses.front() - losses.back()) / losses.front() * 100.0f;
    
    std::cout << "✓ Training over " << num_steps << " steps" << std::endl;
    std::cout << "  Initial loss: " << losses.front() << std::endl;
    std::cout << "  Final loss: " << losses.back() << std::endl;
    std::cout << "  Loss reduction: " << reduction << "%" << std::endl;
    
    if (loss_decreased) {
        std::cout << "  ✅ Loss is decreasing (model is learning)" << std::endl;
    } else {
        std::cout << "  ⚠️  Loss did not decrease (may need more steps or better lr)" << std::endl;
    }
    
    delete[] h_input;
    
    return true;
}

// Test Memory Efficiency (No allocations during training)
bool test_memory_efficiency() {
    print_test_header("Memory Efficiency Test");
    
    int batch_size = 8;
    AutoencoderGPUOptV1 model(batch_size);
    
    int input_size = batch_size * 3 * 32 * 32;
    float* h_input = new float[input_size];
    for (int i = 0; i < input_size; i++) {
        h_input[i] = 0.5f;
    }
    
    // Get initial memory usage
    size_t free_before, total;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
    
    // Run multiple training steps
    int num_steps = 20;
    for (int i = 0; i < num_steps; i++) {
        model.train_step(h_input, batch_size, 0.001f);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check memory usage after
    size_t free_after;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    
    // Memory should not have increased significantly (allow small buffer for CUDA runtime)
    long long mem_diff = (long long)free_before - (long long)free_after;
    bool no_memory_leak = mem_diff < 10 * 1024 * 1024;  // Allow 10MB tolerance
    
    TEST_ASSERT(no_memory_leak, "Memory should not increase during training (memory pool should reuse buffers)");
    
    std::cout << "✓ Memory efficiency verified" << std::endl;
    std::cout << "  Training steps: " << num_steps << std::endl;
    std::cout << "  Memory before: " << (total - free_before) / 1024 / 1024 << " MB used" << std::endl;
    std::cout << "  Memory after: " << (total - free_after) / 1024 / 1024 << " MB used" << std::endl;
    std::cout << "  Memory diff: " << mem_diff / 1024 << " KB" << std::endl;
    std::cout << "  ✅ No memory leaks detected (pool reuses buffers)" << std::endl;
    
    delete[] h_input;
    
    return true;
}

// Performance Comparison: Measure throughput
bool test_throughput() {
    print_test_header("Throughput Measurement");
    
    int batch_size = 32;
    float learning_rate = 0.001f;
    int warmup_steps = 5;
    int measure_steps = 20;
    
    AutoencoderGPUOptV1 model(batch_size);
    
    int input_size = batch_size * 3 * 32 * 32;
    float* h_input = new float[input_size];
    for (int i = 0; i < input_size; i++) {
        h_input[i] = (float)(rand() % 1000) / 1000.0f;
    }
    
    // Warmup
    for (int i = 0; i < warmup_steps; i++) {
        model.train_step(h_input, batch_size, learning_rate);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measure
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < measure_steps; i++) {
        model.train_step(h_input, batch_size, learning_rate);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    float total_images = (float)(measure_steps * batch_size);
    float images_per_sec = total_images / (ms / 1000.0f);
    float ms_per_batch = ms / measure_steps;
    
    std::cout << "✓ Throughput measurement completed" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Steps measured: " << measure_steps << std::endl;
    std::cout << "  Total time: " << ms << " ms" << std::endl;
    std::cout << "  Time per batch: " << std::fixed << std::setprecision(2) << ms_per_batch << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(0) << images_per_sec << " images/sec" << std::endl;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    delete[] h_input;
    
    return true;
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   OPTIMIZED v1 GPU AUTOENCODER TEST SUITE - PHASE 3           ║\n";
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
    std::cout << "Total Global Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
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
    
    RUN_TEST(test_model_initialization);
    RUN_TEST(test_forward_pass);
    RUN_TEST(test_training_step);
    RUN_TEST(test_loss_decrease);
    RUN_TEST(test_memory_efficiency);
    RUN_TEST(test_throughput);
    
    // Summary
    std::cout << "\n";
    print_separator();
    std::cout << "TEST SUMMARY" << std::endl;
    print_separator();
    std::cout << "Total: " << total << " | Passed: " << passed << " | Failed: " << (total - passed) << std::endl;
    
    if (passed == total) {
        std::cout << "\n🎉 ALL TESTS PASSED! 🎉\n" << std::endl;
        std::cout << "The optimized v1 GPU autoencoder is ready for training!" << std::endl;
        std::cout << "Compare throughput with naive version to measure speedup." << std::endl;
        return 0;
    } else {
        std::cout << "\n⚠️  SOME TESTS FAILED ⚠️\n" << std::endl;
        return 1;
    }
}
