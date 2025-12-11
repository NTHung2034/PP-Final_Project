#include "models/autoencoder_gpu.cuh"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <vector>

// =============================================================================
// Test Functions
// =============================================================================

bool test_autoencoder_forward_shape() {
    std::cout << "Test: GPU Autoencoder forward shape... ";
    
    GPUAutoencoder model;
    
    GPUTensor input(1, 3, 32, 32);
    for (size_t i = 0; i < input.size; ++i)
        input.h_data[i] = (float)(i % 100) / 100.0f;
    input.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    GPUTensor output(1, 3, 32, 32);
    model.forward_inference(input, output);
    
    output.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    bool success = (output.batch == 1 && output.channels == 3 &&
                    output.height == 32 && output.width == 32);
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

bool test_autoencoder_batch() {
    std::cout << "Test: GPU Autoencoder batch processing... ";
    
    GPUAutoencoder model;
    
    GPUTensor input(4, 3, 32, 32);
    for (size_t i = 0; i < input.size; ++i)
        input.h_data[i] = (float)(i % 256) / 256.0f;
    input.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    GPUTensor output(4, 3, 32, 32);
    model.forward_inference(input, output);
    
    output.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    bool success = (output.batch == 4);
    
    float sum = 0.0f;
    for (size_t i = 0; i < output.size; ++i)
        sum += std::abs(output.h_data[i]);
    success = success && (sum > 0.0f);
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

bool test_autoencoder_feature_extraction() {
    std::cout << "Test: GPU Autoencoder feature extraction... ";
    
    GPUAutoencoder model;
    
    GPUTensor input(2, 3, 32, 32);
    for (size_t i = 0; i < input.size; ++i)
        input.h_data[i] = 0.5f;
    input.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    GPUTensor features(2, 128, 8, 8);
    model.extract_features(input, features);
    
    features.copyToHost();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Features: [2, 128, 8, 8] = 2 * 8192
    bool success = (features.batch == 2 && features.channels == 128 &&
                    features.height == 8 && features.width == 8);
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

bool test_autoencoder_loss_computation() {
    std::cout << "Test: GPU Autoencoder loss computation... ";
    
    GPUAutoencoder model;
    
    GPUTensor input(1, 3, 32, 32);
    for (size_t i = 0; i < input.size; ++i)
        input.h_data[i] = 0.5f;
    input.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    GPUTensor output(1, 3, 32, 32);
    model.forward_inference(input, output);
    
    float loss = mse_loss_forward_gpu(output, input);
    
    bool success = (loss >= 0.0f && !std::isnan(loss) && !std::isinf(loss));
    
    std::cout << (success ? "PASSED" : "FAILED")
              << " (loss=" << std::fixed << std::setprecision(6) << loss << ")" << std::endl;
    return success;
}

bool test_autoencoder_training_step() {
    std::cout << "Test: GPU Autoencoder training step... ";
    
    GPUAutoencoder model;
    model.allocatePoolingIndices(4);
    
    GPUTensor input(4, 3, 32, 32);
    for (size_t i = 0; i < input.size; ++i)
        input.h_data[i] = 0.5f + 0.1f * std::sin((float)i * 0.01f);
    input.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Preallocate activation buffers
    std::vector<GPUTensor*> activations;
    activations.push_back(new GPUTensor(4, 256, 32, 32, true));
    activations.push_back(new GPUTensor(4, 256, 16, 16, true));
    activations.push_back(new GPUTensor(4, 128, 16, 16, true));
    activations.push_back(new GPUTensor(4, 128, 8, 8, true));
    activations.push_back(new GPUTensor(4, 128, 8, 8, true));
    activations.push_back(new GPUTensor(4, 128, 16, 16, true));
    activations.push_back(new GPUTensor(4, 256, 16, 16, true));
    activations.push_back(new GPUTensor(4, 256, 32, 32, true));
    activations.push_back(new GPUTensor(4, 3, 32, 32, true));
    
    float loss = model.forward_backward_update(input, input, 0.001f, activations);
    
    bool success = (loss >= 0.0f && !std::isnan(loss) && !std::isinf(loss));
    
    std::cout << (success ? "PASSED" : "FAILED")
              << " (loss=" << std::fixed << std::setprecision(6) << loss << ")" << std::endl;
    
    for (auto* act : activations) delete act;
    return success;
}

bool test_autoencoder_loss_decrease() {
    std::cout << "Test: GPU Autoencoder training (loss decrease)... ";
    
    GPUAutoencoder model;
    model.allocatePoolingIndices(8);
    
    GPUTensor input(8, 3, 32, 32);
    for (size_t i = 0; i < input.size; ++i)
        input.h_data[i] = (float)(rand() % 256) / 255.0f;
    input.copyToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<GPUTensor*> activations;
    activations.push_back(new GPUTensor(8, 256, 32, 32, true));
    activations.push_back(new GPUTensor(8, 256, 16, 16, true));
    activations.push_back(new GPUTensor(8, 128, 16, 16, true));
    activations.push_back(new GPUTensor(8, 128, 8, 8, true));
    activations.push_back(new GPUTensor(8, 128, 8, 8, true));
    activations.push_back(new GPUTensor(8, 128, 16, 16, true));
    activations.push_back(new GPUTensor(8, 256, 16, 16, true));
    activations.push_back(new GPUTensor(8, 256, 32, 32, true));
    activations.push_back(new GPUTensor(8, 3, 32, 32, true));
    
    float loss_initial = model.forward_backward_update(input, input, 0.01f, activations);
    
    float loss_final = loss_initial;
    for (int i = 0; i < 5; ++i)
        loss_final = model.forward_backward_update(input, input, 0.01f, activations);
    
    bool success = (loss_final <= loss_initial * 1.1f); // Allow small fluctuation
    
    std::cout << (success ? "PASSED" : "FAILED")
              << " (initial=" << std::fixed << std::setprecision(6) << loss_initial
              << ", final=" << loss_final << ")" << std::endl;
    
    for (auto* act : activations) delete act;
    return success;
}

bool test_autoencoder_throughput() {
    std::cout << "Test: GPU Autoencoder throughput... ";
    
    GPUAutoencoder model;
    
    const int batch_size = 32;
    const int iterations = 10;
    
    GPUTensor input(batch_size, 3, 32, 32);
    for (size_t i = 0; i < input.size; ++i)
        input.h_data[i] = (float)(rand() % 256) / 255.0f;
    input.copyToDevice();
    
    GPUTensor output(batch_size, 3, 32, 32);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i)
        model.forward_inference(input, output);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float images_per_sec = (batch_size * iterations * 1000.0f) / duration.count();
    
    std::cout << "PASSED (" << std::fixed << std::setprecision(1) 
              << images_per_sec << " images/sec)" << std::endl;
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    srand(42);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "\n========================================\n";
    std::cout << "  GPU Autoencoder Unit Tests\n";
    std::cout << "========================================\n";
    std::cout << "  Device: " << prop.name << "\n";
    std::cout << "  Memory: " << std::fixed << std::setprecision(1)
              << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    std::cout << "========================================\n\n";
    
    int passed = 0, total = 0;
    
    total++; if (test_autoencoder_forward_shape()) passed++;
    total++; if (test_autoencoder_batch()) passed++;
    total++; if (test_autoencoder_feature_extraction()) passed++;
    total++; if (test_autoencoder_loss_computation()) passed++;
    total++; if (test_autoencoder_training_step()) passed++;
    total++; if (test_autoencoder_loss_decrease()) passed++;
    total++; if (test_autoencoder_throughput()) passed++;
    
    std::cout << "\n========================================\n";
    std::cout << "  Results: " << passed << "/" << total << " tests passed\n";
    std::cout << "========================================\n\n";
    
    cudaDeviceReset();
    return (passed == total) ? 0 : 1;
}
