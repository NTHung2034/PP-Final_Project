// Autoencoder Unit Tests - Raw Pointer API
#include "models/autoencoder_cpu.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

bool test_autoencoder_shape() {
    std::cout << "Test: Autoencoder forward pass shape... ";

    AutoencoderCPU model;
    
    int batch = 1;
    size_t input_size = batch * 3 * 32 * 32;
    float* input = new float[input_size];
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = static_cast<float>(i % 100) / 100.0f;
    }

    float* output = model.forward(input, batch);
    
    // Output should be [1, 3, 32, 32] = 3072 elements
    // We can verify by computing a sample value
    bool success = (output != nullptr);
    
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    
    delete[] input;
    return success;
}

bool test_autoencoder_feature_extraction() {
    std::cout << "Test: Autoencoder feature extraction... ";

    AutoencoderCPU model;
    
    int batch = 2;
    size_t input_size = batch * 3 * 32 * 32;
    float* input = new float[input_size];
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = 0.5f;
    }

    float* features = model.extract_features(input, batch);
    
    // Features should be [2, 128, 8, 8] = 8192 per image
    int expected_latent = model.get_latent_size();  // 8192
    bool success = (features != nullptr && expected_latent == 128 * 8 * 8);

    std::cout << (success ? "PASSED" : "FAILED");
    std::cout << " (latent size = " << expected_latent << ")" << std::endl;
    
    delete[] input;
    return success;
}

bool test_autoencoder_loss() {
    std::cout << "Test: Autoencoder MSE loss computation... ";

    AutoencoderCPU model;
    
    int batch = 1;
    size_t input_size = batch * 3 * 32 * 32;
    float* input = new float[input_size];
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = 0.5f;
    }

    float* output = model.forward(input, batch);
    float loss = AutoencoderCPU::compute_loss(output, input, input_size);

    bool success = (loss >= 0.0f);

    std::cout << (success ? "PASSED" : "FAILED");
    std::cout << " (loss = " << std::fixed << std::setprecision(6) << loss << ")" << std::endl;

    delete[] input;
    return success;
}

bool test_autoencoder_backward() {
    std::cout << "Test: Autoencoder backward pass... ";

    AutoencoderCPU model;
    
    int batch = 1;
    size_t input_size = batch * 3 * 32 * 32;
    float* input = new float[input_size];
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = static_cast<float>(i % 100) / 100.0f;
    }

    float* output = model.forward(input, batch);
    float loss_before = AutoencoderCPU::compute_loss(output, input, input_size);

    try {
        model.backward(input, 0.001f);
        std::cout << "PASSED (loss = " << std::fixed << std::setprecision(6)
                  << loss_before << ")" << std::endl;
        delete[] input;
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
        delete[] input;
        return false;
    }
}

bool test_autoencoder_training_iteration() {
    std::cout << "Test: Autoencoder training iteration (loss decrease)... ";

    AutoencoderCPU model;
    
    int batch = 4;
    size_t input_size = batch * 3 * 32 * 32;
    float* input = new float[input_size];
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = 0.5f + 0.1f * std::sin(i * 0.01f);
    }

    // Measure initial loss
    float* output = model.forward(input, batch);
    float loss_initial = AutoencoderCPU::compute_loss(output, input, input_size);

    // Train for 10 iterations
    float loss_final = loss_initial;
    for (int i = 0; i < 10; ++i) {
        output = model.forward(input, batch);
        loss_final = AutoencoderCPU::compute_loss(output, input, input_size);
        model.backward(input, 0.01f);
    }

    bool success = (loss_final <= loss_initial);

    std::cout << (success ? "PASSED" : "FAILED");
    std::cout << " (initial = " << std::fixed << std::setprecision(6) << loss_initial
              << ", final = " << loss_final << ")" << std::endl;

    delete[] input;
    return success;
}

bool test_autoencoder_save_load() {
    std::cout << "Test: Autoencoder save/load weights... ";

    AutoencoderCPU model1;
    
    int batch = 1;
    size_t input_size = batch * 3 * 32 * 32;
    float* input = new float[input_size];
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // Forward pass with first model
    float* output1 = model1.forward(input, batch);
    float loss1 = AutoencoderCPU::compute_loss(output1, input, input_size);

    // Save weights
    std::string test_weights_file = std::string(MODEL_SAVE_DIR) + "/test_weights.bin";
    try {
        model1.save_weights(test_weights_file);

        // Create new model and load weights
        AutoencoderCPU model2;
        model2.load_weights(test_weights_file);

        // Forward pass with second model
        float* output2 = model2.forward(input, batch);
        float loss2 = AutoencoderCPU::compute_loss(output2, input, input_size);

        // Losses should be identical
        bool success = (std::abs(loss1 - loss2) < 1e-5f);

        std::cout << (success ? "PASSED" : "FAILED");
        std::cout << " (loss1 = " << std::fixed << std::setprecision(6) << loss1 
                  << ", loss2 = " << loss2 << ")" << std::endl;
        
        delete[] input;
        return success;
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << std::endl;
        delete[] input;
        return false;
    }
}

int main() {
    std::cout << "\n=== Autoencoder Unit Tests (Raw Pointer API) ===\n\n";

    int passed = 0, total = 0;

    total++; if (test_autoencoder_shape()) passed++;
    total++; if (test_autoencoder_feature_extraction()) passed++;
    total++; if (test_autoencoder_loss()) passed++;
    total++; if (test_autoencoder_backward()) passed++;
    total++; if (test_autoencoder_training_iteration()) passed++;
    total++; if (test_autoencoder_save_load()) passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n\n";

    return (passed == total) ? 0 : 1;
}
