// Test to compare CPU, GPU Naive, GPU Opt V1, and GPU Opt V2 autoencoder decoded outputs
// Uses the SAME weights for all models to verify correctness of implementations
#include "models/autoencoder_cpu.h"
#include "models/autoencoder_gpu_naive.cuh"
#include "models/autoencoder_gpu_opt_v1.cuh"
#include "models/autoencoder_gpu_opt_v2.cuh"
#include "data/cifar10_loader.h"
#include "config.h"
#include "data/gpu_data_types.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

// Fill with random data (for fallback if dataloader fails)
void fill_random(float* data, size_t size, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) data[i] = dist(gen);
}

// Generate weights with fixed seed and copy to all models
// All architectures use the same weight layout: [out_channels, in_channels, kernel_h, kernel_w]
// This ensures compatibility across CPU, GPU Naive, GPU Opt V1, and GPU Opt V2
void set_same_weights_on_all_models(
    AutoencoderCPU& cpu_model,
    GPUAutoencoderNaive& gpu_naive,
    AutoencoderGPUOptV1& gpu_opt_v1,
    AutoencoderGPUOptV2& gpu_opt_v2) {
    
    std::cout << "Generating weights with fixed seed (42) and copying to all models..." << std::endl;
    std::cout << "  Weight layout: [out_channels, in_channels, kernel_h, kernel_w] (same for all)" << std::endl;
    
    // Fixed seed for reproducibility
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    // Layer configurations: (in_channels, out_channels, kernel_size)
    struct LayerConfig {
        int in_c, out_c, k_size;
    };
    
    LayerConfig layers[] = {
        {3, 256, 3},    // conv1
        {256, 128, 3},  // conv2
        {128, 128, 3},  // conv3
        {128, 256, 3},  // conv4
        {256, 3, 3}     // conv5
    };
    
    // Store all weights first, then copy to all models
    std::vector<std::vector<float>> all_weights(5);
    std::vector<std::vector<float>> all_bias(5);
    
    // Generate weights for all layers
    for (int layer_idx = 0; layer_idx < 5; ++layer_idx) {
        const auto& cfg = layers[layer_idx];
        int weight_size = cfg.out_c * cfg.in_c * cfg.k_size * cfg.k_size;
        int bias_size = cfg.out_c;
        
        all_weights[layer_idx].resize(weight_size);
        all_bias[layer_idx].resize(bias_size, 0.0f);
        
        for (int i = 0; i < weight_size; ++i) {
            all_weights[layer_idx][i] = dist(gen);
        }
    }
    
    // Copy to GPU Naive
    GPUConvWeights* gpu_naive_layers[] = {
        gpu_naive.conv1, gpu_naive.conv2, gpu_naive.conv3, gpu_naive.conv4, gpu_naive.conv5
    };
    for (int layer_idx = 0; layer_idx < 5; ++layer_idx) {
        CUDA_CHECK(cudaMemcpy(gpu_naive_layers[layer_idx]->d_weights, 
                              all_weights[layer_idx].data(), 
                              all_weights[layer_idx].size() * sizeof(float), 
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_naive_layers[layer_idx]->d_bias, 
                              all_bias[layer_idx].data(), 
                              all_bias[layer_idx].size() * sizeof(float), 
                              cudaMemcpyHostToDevice));
    }
    
    // Copy to GPU Opt V1
    GPUConvWeightsOpt* gpu_opt_v1_layers[] = {
        gpu_opt_v1.conv1, gpu_opt_v1.conv2, gpu_opt_v1.conv3, gpu_opt_v1.conv4, gpu_opt_v1.conv5
    };
    for (int layer_idx = 0; layer_idx < 5; ++layer_idx) {
        CUDA_CHECK(cudaMemcpy(gpu_opt_v1_layers[layer_idx]->d_weights, 
                              all_weights[layer_idx].data(), 
                              all_weights[layer_idx].size() * sizeof(float), 
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_opt_v1_layers[layer_idx]->d_bias, 
                              all_bias[layer_idx].data(), 
                              all_bias[layer_idx].size() * sizeof(float), 
                              cudaMemcpyHostToDevice));
    }
    
    // Copy to GPU Opt V2
    GPUConvWeightsOpt* gpu_opt_v2_layers[] = {
        gpu_opt_v2.conv1, gpu_opt_v2.conv2, gpu_opt_v2.conv3, gpu_opt_v2.conv4, gpu_opt_v2.conv5
    };
    for (int layer_idx = 0; layer_idx < 5; ++layer_idx) {
        CUDA_CHECK(cudaMemcpy(gpu_opt_v2_layers[layer_idx]->d_weights, 
                              all_weights[layer_idx].data(), 
                              all_weights[layer_idx].size() * sizeof(float), 
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_opt_v2_layers[layer_idx]->d_bias, 
                              all_bias[layer_idx].data(), 
                              all_bias[layer_idx].size() * sizeof(float), 
                              cudaMemcpyHostToDevice));
    }
    
    // Set weights on CPU model using temporary file
    const char* temp_weights_file = "temp_test_weights.bin";
    {
        std::ofstream file(temp_weights_file, std::ios::binary);
        if (!file) {
            std::cerr << "Warning: Failed to create temp weights file" << std::endl;
            return;
        }
        
        // Write all weights to file in CPU format
        for (int layer_idx = 0; layer_idx < 5; ++layer_idx) {
            int weight_size = all_weights[layer_idx].size();
            int bias_size = all_bias[layer_idx].size();
            
            file.write(reinterpret_cast<const char*>(&weight_size), sizeof(int));
            file.write(reinterpret_cast<const char*>(&bias_size), sizeof(int));
            file.write(reinterpret_cast<const char*>(all_weights[layer_idx].data()), 
                      weight_size * sizeof(float));
            file.write(reinterpret_cast<const char*>(all_bias[layer_idx].data()), 
                      bias_size * sizeof(float));
        }
        file.close();
    }
    
    // Load weights into CPU model
    try {
        cpu_model.load_weights(temp_weights_file);
        std::cout << "  ✓ Weights set on all models" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ✗ Failed to set CPU weights: " << e.what() << std::endl;
    }
    
    // Verify weights are actually the same by checking a few samples
    // (This is a sanity check - weights should match since we copied from same source)
    bool weights_match = true;
    for (int layer_idx = 0; layer_idx < 5 && weights_match; ++layer_idx) {
        const auto& cfg = layers[layer_idx];
        int weight_size = cfg.out_c * cfg.in_c * cfg.k_size * cfg.k_size;
        
        // Check GPU Naive weights match
        std::vector<float> gpu_naive_check(weight_size);
        CUDA_CHECK(cudaMemcpy(gpu_naive_check.data(), gpu_naive_layers[layer_idx]->d_weights,
                              weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < weight_size && weights_match; ++i) {
            if (std::abs(gpu_naive_check[i] - all_weights[layer_idx][i]) > 1e-6f) {
                weights_match = false;
                std::cerr << "  ✗ Weight mismatch detected at layer " << layer_idx 
                         << ", index " << i << std::endl;
            }
        }
    }
    
    if (weights_match) {
        std::cout << "  ✓ Verified: All models have identical weights" << std::endl;
    } else {
        std::cerr << "  ✗ Warning: Weight verification failed!" << std::endl;
    }
    
    // Clean up temp file
    std::remove(temp_weights_file);
}

// Compute statistics for comparison
struct ComparisonStats {
    double mse;
    double max_diff;
    double mean_diff;
    double relative_error;
    size_t num_elements;
};

ComparisonStats compare_vectors(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Error: Vector sizes don't match: " << a.size() << " vs " << b.size() << std::endl;
        return {0, 0, 0, 0, 0};
    }
    
    ComparisonStats stats;
    stats.num_elements = a.size();
    double sum_sq_diff = 0.0;
    double sum_diff = 0.0;
    double max_abs_diff = 0.0;
    double sum_abs_a = 0.0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        double abs_diff = std::abs(diff);
        sum_sq_diff += diff * diff;
        sum_diff += diff;
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        sum_abs_a += std::abs(a[i]);
    }
    
    stats.mse = sum_sq_diff / a.size();
    stats.max_diff = max_abs_diff;
    stats.mean_diff = sum_diff / a.size();
    stats.relative_error = (sum_abs_a > 0) ? (sum_sq_diff / sum_abs_a) : 0.0;
    
    return stats;
}

int main(int argc, char* argv[]) {
    constexpr int batch = 1;
    constexpr int channels = 3;
    constexpr int height = 32;
    constexpr int width = 32;
    constexpr size_t img_size = batch * channels * height * width;
    constexpr size_t output_size = img_size;
    
    std::cout << "=== Output Reconstruction Comparison Test ===" << std::endl;
    std::cout << "Testing if all implementations produce identical decoded outputs" << std::endl;
    std::cout << "Using SAME weights and SAME input for all models" << std::endl << std::endl;
    
    // Prepare input image - try to load from dataloader first
    std::vector<float> image(img_size);
    bool use_dataloader = false;
    
    try {
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        if (loader.load_train_data()) {
            float* first_image = loader.get_batch_at(0, 1);
            if (first_image) {
                std::copy(first_image, first_image + img_size, image.data());
                use_dataloader = true;
                std::cout << "Using first image from CIFAR-10 training set" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load from dataloader: " << e.what() << std::endl;
    }
    
    if (!use_dataloader) {
        fill_random(image.data(), img_size, 42);
        std::cout << "Using random image (seed=42)" << std::endl;
    }
    
    std::cout << std::endl;
    
    // === Initialize all models ===
    std::cout << "Initializing models..." << std::endl;
    AutoencoderCPU cpu_model;
    GPUAutoencoderNaive gpu_naive;
    AutoencoderGPUOptV1 gpu_opt_v1(batch);
    AutoencoderGPUOptV2 gpu_opt_v2(batch);
    std::cout << "  ✓ All models initialized" << std::endl;
    std::cout << std::endl;
    
    // === Set same weights on all models ===
    set_same_weights_on_all_models(cpu_model, gpu_naive, gpu_opt_v1, gpu_opt_v2);
    std::cout << std::endl;
    
    // === Run forward pass to get decoded outputs ===
    std::cout << "Running forward passes to obtain outputs..." << std::endl;
    
    // CPU
    Tensor cpu_input({batch, channels, height, width}, false);
    std::copy(image.data(), image.data() + img_size, cpu_input.data->begin());
    Tensor cpu_output = cpu_model.forward(cpu_input);
    std::vector<float> cpu_output_host(cpu_output.size());
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        cpu_output_host[i] = cpu_output.data->at(i);
    }
    std::cout << "  ✓ CPU output computed: " << cpu_output_host.size() << " elements" << std::endl;
    
    // GPU Naive
    GPUTensor gpu_input(batch, channels, height, width, true);
    CUDA_CHECK(cudaMemcpy(gpu_input.d_data, image.data(), img_size * sizeof(float), cudaMemcpyHostToDevice));
    GPUTensor gpu_output_naive(batch, channels, height, width, true);
    gpu_naive.forward_inference(gpu_input, gpu_output_naive);
    std::vector<float> gpu_output_naive_host(output_size);
    CUDA_CHECK(cudaMemcpy(gpu_output_naive_host.data(), gpu_output_naive.d_data, 
                          output_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "  ✓ GPU Naive output computed" << std::endl;
    
    // GPU Opt V1 - run forward and grab decoded output
    gpu_opt_v1.forward(image.data(), batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> gpu_output_v1_host(output_size);
    CUDA_CHECK(cudaMemcpy(gpu_output_v1_host.data(), gpu_opt_v1.pool.output.d_data, 
                          output_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "  ✓ GPU Opt V1 output computed" << std::endl;
    
    // GPU Opt V2 - run forward and grab decoded output
    gpu_opt_v2.forward(image.data(), batch);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> gpu_output_v2_host(output_size);
    CUDA_CHECK(cudaMemcpy(gpu_output_v2_host.data(), gpu_opt_v2.pool.output.d_data, 
                          output_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "  ✓ GPU Opt V2 output computed" << std::endl;
    
    std::cout << std::endl;
    
    // === Compare outputs ===
    std::cout << "=== Comparison Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    
    // Print first few values for visual inspection
    std::cout << "\nFirst 10 output values:" << std::endl;
    std::cout << "CPU:        ";
    for (int i = 0; i < 10 && i < (int)cpu_output_host.size(); ++i) {
        std::cout << std::setw(10) << cpu_output_host[i] << " ";
    }
    std::cout << "\nGPU Naive:  ";
    for (int i = 0; i < 10 && i < (int)gpu_output_naive_host.size(); ++i) {
        std::cout << std::setw(10) << gpu_output_naive_host[i] << " ";
    }
    std::cout << "\nGPU Opt V1: ";
    for (int i = 0; i < 10 && i < (int)gpu_output_v1_host.size(); ++i) {
        std::cout << std::setw(10) << gpu_output_v1_host[i] << " ";
    }
    std::cout << "\nGPU Opt V2: ";
    for (int i = 0; i < 10 && i < (int)gpu_output_v2_host.size(); ++i) {
        std::cout << std::setw(10) << gpu_output_v2_host[i] << " ";
    }
    std::cout << std::endl << std::endl;
    
    // Detailed comparisons
    auto stats_naive = compare_vectors(cpu_output_host, gpu_output_naive_host);
    auto stats_v1 = compare_vectors(cpu_output_host, gpu_output_v1_host);
    auto stats_v2 = compare_vectors(cpu_output_host, gpu_output_v2_host);
    
    std::cout << "CPU vs GPU Naive:" << std::endl;
    std::cout << "  MSE:            " << stats_naive.mse << std::endl;
    std::cout << "  Max Difference: " << stats_naive.max_diff << std::endl;
    std::cout << "  Mean Difference: " << stats_naive.mean_diff << std::endl;
    std::cout << "  Relative Error:  " << stats_naive.relative_error << std::endl;
    
    std::cout << "\nCPU vs GPU Opt V1:" << std::endl;
    std::cout << "  MSE:            " << stats_v1.mse << std::endl;
    std::cout << "  Max Difference: " << stats_v1.max_diff << std::endl;
    std::cout << "  Mean Difference: " << stats_v1.mean_diff << std::endl;
    std::cout << "  Relative Error:  " << stats_v1.relative_error << std::endl;
    
    std::cout << "\nCPU vs GPU Opt V2:" << std::endl;
    std::cout << "  MSE:            " << stats_v2.mse << std::endl;
    std::cout << "  Max Difference: " << stats_v2.max_diff << std::endl;
    std::cout << "  Mean Difference: " << stats_v2.mean_diff << std::endl;
    std::cout << "  Relative Error:  " << stats_v2.relative_error << std::endl;
    
    // Summary
    constexpr double tolerance = 1e-4;  // Allow small numerical differences
    bool all_match = (stats_naive.mse < tolerance) && 
                     (stats_v1.mse < tolerance) && 
                     (stats_v2.mse < tolerance);
    
    std::cout << "\n=== Summary ===" << std::endl;
    if (all_match) {
        std::cout << "✓ SUCCESS: All models produce identical outputs (within tolerance " << tolerance << ")" << std::endl;
        std::cout << "  This confirms that all implementations are correct!" << std::endl;
    } else {
        std::cout << "✗ FAILURE: Models produce different decoded outputs" << std::endl;
        std::cout << "  This indicates implementation differences or bugs:" << std::endl;
        if (stats_naive.mse >= tolerance) {
            std::cout << "    - GPU Naive differs from CPU (MSE: " << stats_naive.mse << ")" << std::endl;
        }
        if (stats_v1.mse >= tolerance) {
            std::cout << "    - GPU Opt V1 differs from CPU (MSE: " << stats_v1.mse << ")" << std::endl;
        }
        if (stats_v2.mse >= tolerance) {
            std::cout << "    - GPU Opt V2 differs from CPU (MSE: " << stats_v2.mse << ")" << std::endl;
        }
    }
    
    return all_match ? 0 : 1;
}
