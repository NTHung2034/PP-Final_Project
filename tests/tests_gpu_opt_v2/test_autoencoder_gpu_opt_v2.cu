// Test GPU Optimized v2 Autoencoder
// Optimizations: CUDA Streams, Kernel Fusion, Double Buffering Pipeline
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>

#include "models/autoencoder_gpu_opt_v2.cuh"

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

void init_random(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

// =============================================================================
// Test Model Construction
// =============================================================================
void test_model_construction() {
    std::cout << "\n--- Testing Model Construction ---\n";
    
    AutoencoderGPUOptV2 model(4);
    
    print_test_result("Model created", true);
    print_test_result("Memory pool allocated", model.pool.allocated);
    print_test_result("Streams created", model.pool.streams_created);
    print_test_result("Input buffer[0] allocated", model.input_buffer[0].d_data != nullptr);
    print_test_result("Input buffer[1] allocated", model.input_buffer[1].d_data != nullptr);
    print_test_result("Conv1 weights allocated", model.conv1->d_weights != nullptr);
}

// =============================================================================
// Test Forward Pass
// =============================================================================
void test_forward_pass() {
    std::cout << "\n--- Testing Forward Pass ---\n";
    
    int batch_size = 4;
    AutoencoderGPUOptV2 model(batch_size);
    
    int input_size = batch_size * 3 * 32 * 32;
    float* h_input = new float[input_size];
    init_random(h_input, input_size);
    
    // Use new double-buffer API
    model.copy_input_async(h_input, batch_size);
    model.swap_buffers();
    float loss = model.forward(batch_size);
    
    print_test_result("Forward pass completed", true);
    print_test_result("Loss is positive", loss > 0.0f);
    print_test_result("Loss is finite", std::isfinite(loss));
    
    std::cout << "  Loss: " << std::fixed << std::setprecision(6) << loss << std::endl;
    
    delete[] h_input;
}

// =============================================================================
// Test Backward Pass (with Streams)
// =============================================================================
void test_backward_pass() {
    std::cout << "\n--- Testing Backward Pass (with streams) ---\n";
    
    int batch_size = 4;
    float learning_rate = 0.001f;
    AutoencoderGPUOptV2 model(batch_size);
    
    int input_size = batch_size * 3 * 32 * 32;
    float* h_input = new float[input_size];
    init_random(h_input, input_size);
    
    // Forward with new API
    model.copy_input_async(h_input, batch_size);
    model.swap_buffers();
    float loss = model.forward(batch_size);
    
    // Save weights before backward
    float w_before;
    cudaMemcpy(&w_before, model.conv1->d_weights, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Backward + update (model syncs all streams internally)
    model.backward(learning_rate);
    
    // Check weights changed
    float w_after;
    cudaMemcpy(&w_after, model.conv1->d_weights, sizeof(float), cudaMemcpyDeviceToHost);
    
    print_test_result("Backward pass completed", true);
    print_test_result("Weights updated", w_before != w_after);
    
    std::cout << "  Weight before: " << w_before << ", after: " << w_after << std::endl;
    
    delete[] h_input;
}

// =============================================================================
// Test Training Step
// =============================================================================
void test_train_step() {
    std::cout << "\n--- Testing Train Step ---\n";
    
    int batch_size = 8;
    float learning_rate = 0.001f;
    AutoencoderGPUOptV2 model(batch_size);
    
    int input_size = batch_size * 3 * 32 * 32;
    float* h_input = new float[input_size];
    init_random(h_input, input_size);
    
    // Train step 1
    model.copy_input_async(h_input, batch_size);
    model.swap_buffers();
    float loss1 = model.train_step(learning_rate);
    
    // Train step 2
    model.copy_input_async(h_input, batch_size);
    model.swap_buffers();
    float loss2 = model.train_step(learning_rate);
    
    // Train step 3
    model.copy_input_async(h_input, batch_size);
    model.swap_buffers();
    float loss3 = model.train_step(learning_rate);
    
    print_test_result("Multiple train steps", true);
    print_test_result("Loss decreasing", loss3 < loss1);
    
    std::cout << "  Loss progression: " << loss1 << " -> " << loss2 << " -> " << loss3 << std::endl;
    
    delete[] h_input;
}

// =============================================================================
// Test Training Loop (Multiple Epochs) - With Pipelining
// =============================================================================
void test_training_loop() {
    std::cout << "\n--- Testing Training Loop (with double buffering) ---\n";
    
    int batch_size = 16;
    float learning_rate = 0.001f;
    int num_batches = 10;
    int num_epochs = 3;
    
    AutoencoderGPUOptV2 model(batch_size);
    
    int input_size = batch_size * 3 * 32 * 32;
    float* h_input = new float[input_size];
    
    std::vector<float> epoch_losses;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        // Prime the pipeline
        init_random(h_input, input_size);
        model.copy_input_async(h_input, batch_size);
        model.swap_buffers();
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Start copying next batch while training current
            if (batch < num_batches - 1) {
                init_random(h_input, input_size);
                model.copy_input_async(h_input, batch_size);
            }
            
            float loss = model.train_step(learning_rate);
            epoch_loss += loss;
            
            // Swap for next batch
            if (batch < num_batches - 1) {
                model.swap_buffers();
            }
        }
        
        epoch_losses.push_back(epoch_loss / num_batches);
        std::cout << "  Epoch " << (epoch + 1) << ": " << std::fixed 
                  << std::setprecision(6) << epoch_losses.back() << std::endl;
    }
    
    print_test_result("Training loop completed", true);
    print_test_result("Loss decreased over epochs", epoch_losses.back() < epoch_losses.front());
    
    delete[] h_input;
}

// =============================================================================
// Performance Benchmark (with double buffering)
// =============================================================================
void benchmark_performance() {
    std::cout << "\n--- Performance Benchmark (with double buffering) ---\n";
    
    int batch_size = 32;
    float learning_rate = 0.001f;
    int warmup_iterations = 5;
    int benchmark_iterations = 20;
    
    AutoencoderGPUOptV2 model(batch_size);
    
    int input_size = batch_size * 3 * 32 * 32;
    float* h_input = new float[input_size];
    init_random(h_input, input_size);
    
    // Warmup
    model.copy_input_async(h_input, batch_size);
    model.swap_buffers();
    for (int i = 0; i < warmup_iterations; i++) {
        model.copy_input_async(h_input, batch_size);
        model.train_step(learning_rate);
        model.swap_buffers();
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_iterations; i++) {
        model.copy_input_async(h_input, batch_size);
        model.train_step(learning_rate);
        model.swap_buffers();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    float avg_time_ms = milliseconds / benchmark_iterations;
    float throughput = batch_size / (avg_time_ms / 1000.0f);
    
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Avg iteration time: " << std::fixed << std::setprecision(2) 
              << avg_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << throughput << " images/sec" << std::endl;
    
    print_test_result("Benchmark completed", avg_time_ms > 0);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    GPU OPTIMIZED v2 AUTOENCODER TESTS                          ║\n";
    std::cout << "║    (CUDA Streams + Kernel Fusion + Double Buffering)           ║\n";
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
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    
    // Run tests
    test_model_construction();
    test_forward_pass();
    test_backward_pass();
    test_train_step();
    test_training_loop();
    benchmark_performance();
    
    // Summary
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "RESULTS: " << tests_passed << " passed, " << tests_failed << " failed\n";
    std::cout << std::string(50, '=') << std::endl;
    
    return tests_failed > 0 ? 1 : 0;
}
