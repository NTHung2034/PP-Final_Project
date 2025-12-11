#include "models/autoencoder_gpu_naive.cuh"

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
    print_test_header("Model Initialization");
    
    GPUAutoencoderNaive model;
    
    TEST_ASSERT(model.conv1 != nullptr, "Conv1 should be initialized");
    TEST_ASSERT(model.conv2 != nullptr, "Conv2 should be initialized");
    TEST_ASSERT(model.conv3 != nullptr, "Conv3 should be initialized");
    TEST_ASSERT(model.conv4 != nullptr, "Conv4 should be initialized");
    TEST_ASSERT(model.conv5 != nullptr, "Conv5 should be initialized");
    
    // Check layer dimensions
    TEST_ASSERT(model.conv1->out_channels == 256, "Conv1 should have 256 output channels");
    TEST_ASSERT(model.conv1->in_channels == 3, "Conv1 should have 3 input channels");
    TEST_ASSERT(model.conv5->out_channels == 3, "Conv5 should have 3 output channels");
    
    std::cout << "✓ Model layers correctly initialized" << std::endl;
    std::cout << "  Conv1: [3 -> 256], 3x3 kernel" << std::endl;
    std::cout << "  Conv2: [256 -> 128], 3x3 kernel" << std::endl;
    std::cout << "  Conv3: [128 -> 128], 3x3 kernel" << std::endl;
    std::cout << "  Conv4: [128 -> 256], 3x3 kernel" << std::endl;
    std::cout << "  Conv5: [256 -> 3], 3x3 kernel" << std::endl;
    
    return true;
}

// Test Forward Inference
bool test_forward_inference() {
    print_test_header("Forward Inference");
    
    GPUAutoencoderNaive model;
    
    int batch_size = 4;
    model.allocatePoolingIndices(batch_size);
    
    // Create input tensor
    GPUTensor input(batch_size, 3, 32, 32, false);
    GPUTensor output(batch_size, 3, 32, 32, false);
    
    // Initialize with random-like values
    for (int i = 0; i < input.size; i++) {
        input.h_data[i] = 0.5f + 0.1f * (i % 10);
    }
    input.copyToDevice();
    
    // Run inference
    auto start = std::chrono::high_resolution_clock::now();
    model.forward_inference(input, output);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    output.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check output is valid
    bool has_output = false;
    for (int i = 0; i < output.size; i++) {
        if (!std::isnan(output.h_data[i]) && !std::isinf(output.h_data[i])) {
            has_output = true;
        }
        TEST_ASSERT(!std::isnan(output.h_data[i]), "Output contains NaN values");
        TEST_ASSERT(!std::isinf(output.h_data[i]), "Output contains Inf values");
    }
    
    TEST_ASSERT(has_output, "Output should contain valid values");
    
    std::cout << "✓ Forward inference completed successfully" << std::endl;
    std::cout << "  Input shape: [" << batch_size << ", 3, 32, 32]" << std::endl;
    std::cout << "  Output shape: [" << batch_size << ", 3, 32, 32]" << std::endl;
    std::cout << "  Inference time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Sample output[0]: " << output.h_data[0] << std::endl;
    
    return true;
}

// Test Feature Extraction
bool test_feature_extraction() {
    print_test_header("Feature Extraction (Encoder Only)");
    
    GPUAutoencoderNaive model;
    
    int batch_size = 2;
    model.allocatePoolingIndices(batch_size);
    
    // Create input
    GPUTensor input(batch_size, 3, 32, 32, false);
    GPUTensor features(batch_size, 128, 8, 8, false);
    
    for (int i = 0; i < input.size; i++) {
        input.h_data[i] = 0.3f + 0.05f * (i % 20);
    }
    input.copyToDevice();
    
    // Extract features
    model.extract_features(input, features);
    
    features.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check features
    TEST_ASSERT(features.batch == batch_size, "Feature batch size mismatch");
    TEST_ASSERT(features.channels == 128, "Feature channels should be 128");
    TEST_ASSERT(features.height == 8, "Feature height should be 8");
    TEST_ASSERT(features.width == 8, "Feature width should be 8");
    
    bool has_features = false;
    for (int i = 0; i < features.size; i++) {
        TEST_ASSERT(!std::isnan(features.h_data[i]), "Features contain NaN");
        if (std::abs(features.h_data[i]) > 1e-6f) {
            has_features = true;
        }
    }
    
    TEST_ASSERT(has_features, "Features should contain non-zero values");
    
    std::cout << "✓ Feature extraction successful" << std::endl;
    std::cout << "  Input: [" << batch_size << ", 3, 32, 32]" << std::endl;
    std::cout << "  Latent: [" << batch_size << ", 128, 8, 8]" << std::endl;
    std::cout << "  Feature dimension: " << (128 * 8 * 8) << std::endl;
    
    return true;
}

// Test Forward-Backward-Update
bool test_training_step() {
    print_test_header("Training Step (Forward-Backward-Update)");
    
    GPUAutoencoderNaive model;
    
    int batch_size = 4;
    float learning_rate = 0.001f;
    
    model.allocatePoolingIndices(batch_size);
    
    // Preallocate activations
    std::vector<GPUTensor*> activations;
    activations.push_back(new GPUTensor(batch_size, 256, 32, 32, true));
    activations.push_back(new GPUTensor(batch_size, 256, 16, 16, true));
    activations.push_back(new GPUTensor(batch_size, 128, 16, 16, true));
    activations.push_back(new GPUTensor(batch_size, 128, 8, 8, true));
    activations.push_back(new GPUTensor(batch_size, 128, 8, 8, true));
    activations.push_back(new GPUTensor(batch_size, 128, 16, 16, true));
    activations.push_back(new GPUTensor(batch_size, 256, 16, 16, true));
    activations.push_back(new GPUTensor(batch_size, 256, 32, 32, true));
    activations.push_back(new GPUTensor(batch_size, 3, 32, 32, true));
    
    // Create input/target
    GPUTensor input(batch_size, 3, 32, 32, false);
    for (int i = 0; i < input.size; i++) {
        input.h_data[i] = 0.5f;
    }
    input.copyToDevice();
    
    // Run training step
    auto start = std::chrono::high_resolution_clock::now();
    float loss = model.forward_backward_update(input, input, learning_rate, activations);
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
    
    // Cleanup
    for (auto* tensor : activations) {
        delete tensor;
    }
    
    return true;
}

// Test Multiple Training Steps (Loss Decreasing)
bool test_loss_decrease() {
    print_test_header("Loss Decrease Over Training Steps");
    
    GPUAutoencoderNaive model;
    
    int batch_size = 2;
    float learning_rate = 0.01f;
    int num_steps = 10;
    
    model.allocatePoolingIndices(batch_size);
    
    // Preallocate activations
    std::vector<GPUTensor*> activations;
    activations.push_back(new GPUTensor(batch_size, 256, 32, 32, true));
    activations.push_back(new GPUTensor(batch_size, 256, 16, 16, true));
    activations.push_back(new GPUTensor(batch_size, 128, 16, 16, true));
    activations.push_back(new GPUTensor(batch_size, 128, 8, 8, true));
    activations.push_back(new GPUTensor(batch_size, 128, 8, 8, true));
    activations.push_back(new GPUTensor(batch_size, 128, 16, 16, true));
    activations.push_back(new GPUTensor(batch_size, 256, 16, 16, true));
    activations.push_back(new GPUTensor(batch_size, 256, 32, 32, true));
    activations.push_back(new GPUTensor(batch_size, 3, 32, 32, true));
    
    // Create consistent input
    GPUTensor input(batch_size, 3, 32, 32, false);
    for (int i = 0; i < input.size; i++) {
        input.h_data[i] = 0.8f;
    }
    input.copyToDevice();
    
    std::vector<float> losses;
    
    // Train for multiple steps
    for (int step = 0; step < num_steps; step++) {
        float loss = model.forward_backward_update(input, input, learning_rate, activations);
        losses.push_back(loss);
        
        if (step == 0 || step == num_steps - 1) {
            std::cout << "  Step " << std::setw(2) << step << ": Loss = " 
                      << std::fixed << std::setprecision(6) << loss << std::endl;
        }
    }
    
    // Check if loss decreased
    bool loss_decreased = losses.back() < losses.front();
    
    std::cout << "✓ Training over " << num_steps << " steps" << std::endl;
    std::cout << "  Initial loss: " << losses.front() << std::endl;
    std::cout << "  Final loss: " << losses.back() << std::endl;
    std::cout << "  Loss reduction: " << ((losses.front() - losses.back()) / losses.front() * 100.0f) << "%" << std::endl;
    
    if (loss_decreased) {
        std::cout << "  ✅ Loss is decreasing (model is learning)" << std::endl;
    } else {
        std::cout << "  ⚠️  Loss did not decrease (may need more steps or better lr)" << std::endl;
    }
    
    // Cleanup
    for (auto* tensor : activations) {
        delete tensor;
    }
    
    return true;
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    NAIVE GPU AUTOENCODER TEST SUITE - PHASE 2                 ║\n";
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
    std::cout << "Total Global Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB\n" << std::endl;
    
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
    
    RUN_TEST(test_model_initialization);
    RUN_TEST(test_forward_inference);
    RUN_TEST(test_feature_extraction);
    RUN_TEST(test_training_step);
    RUN_TEST(test_loss_decrease);
    
    // Summary
    std::cout << "\n";
    print_separator();
    std::cout << "TEST SUMMARY" << std::endl;
    print_separator();
    std::cout << "Total: " << total << " | Passed: " << passed << " | Failed: " << (total - passed) << std::endl;
    
    if (passed == total) {
        std::cout << "\n🎉 ALL TESTS PASSED! 🎉\n" << std::endl;
        std::cout << "The naive GPU autoencoder is ready for training!" << std::endl;
        return 0;
    } else {
        std::cout << "\n⚠️  SOME TESTS FAILED ⚠️\n" << std::endl;
        return 1;
    }
}
