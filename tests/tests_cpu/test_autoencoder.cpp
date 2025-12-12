// Autoencoder Unit Tests - Tensor API
#include "models/autoencoder_cpu.h"
#include "data/data_types.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

bool test_autoencoder_shape()
{
    std::cout << "Test: Autoencoder forward pass shape... ";

    AutoencoderCPU model;

    int batch = 1;
    Tensor input({batch, 3, 32, 32});
    for (size_t i = 0; i < input.size(); ++i)
    {
        input.raw_data()[i] = static_cast<float>(i % 100) / 100.0f;
    }

    Tensor output = model.forward(input);

    // Output should be [1, 3, 32, 32] = 3072 elements
    bool success = (output.size() == 3072 && output.shape[0] == 1 &&
                    output.shape[1] == 3 && output.shape[2] == 32 && output.shape[3] == 32);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;

    return success;
}

bool test_autoencoder_feature_extraction()
{
    std::cout << "Test: Autoencoder feature extraction... ";

    AutoencoderCPU model;

    int batch = 2;
    Tensor input({batch, 3, 32, 32});
    for (size_t i = 0; i < input.size(); ++i)
    {
        input.raw_data()[i] = 0.5f;
    }

    Tensor features = model.extract_features(input);

    // Features should be [2, 128, 8, 8]
    int expected_latent = 128 * 8 * 8;
    bool success = (features.shape[0] == 2 && features.shape[1] == 128 &&
                    features.shape[2] == 8 && features.shape[3] == 8);

    std::cout << (success ? "PASSED" : "FAILED");
    std::cout << " (latent size = " << expected_latent << ")" << std::endl;

    return success;
}

bool test_autoencoder_loss()
{
    std::cout << "Test: Autoencoder MSE loss computation... ";

    AutoencoderCPU model;

    int batch = 1;
    Tensor input({batch, 3, 32, 32});
    for (size_t i = 0; i < input.size(); ++i)
    {
        input.raw_data()[i] = 0.5f;
    }

    Tensor output = model.forward(input);
    float loss = model.compute_loss(output, input);

    bool success = (loss >= 0.0f);

    std::cout << (success ? "PASSED" : "FAILED");
    std::cout << " (loss = " << std::fixed << std::setprecision(6) << loss << ")" << std::endl;

    return success;
}

bool test_autoencoder_backward()
{
    std::cout << "Test: Autoencoder backward pass... ";

    AutoencoderCPU model;

    int batch = 1;
    Tensor input({batch, 3, 32, 32});
    for (size_t i = 0; i < input.size(); ++i)
    {
        input.raw_data()[i] = static_cast<float>(i % 100) / 100.0f;
    }

    Tensor output = model.forward(input);
    float loss_before = model.compute_loss(output, input);

    try
    {
        model.backward(input, 0.001f);
        std::cout << "PASSED (loss = " << std::fixed << std::setprecision(6)
                  << loss_before << ")" << std::endl;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAILED: " << e.what() << std::endl;
        return false;
    }
}

bool test_autoencoder_training_iteration()
{
    std::cout << "Test: Autoencoder training iteration (loss decrease)... ";

    AutoencoderCPU model;

    int batch = 4;
    Tensor input({batch, 3, 32, 32});
    for (size_t i = 0; i < input.size(); ++i)
    {
        input.raw_data()[i] = 0.5f + 0.1f * std::sin(i * 0.01f);
    }

    // Measure initial loss
    Tensor output = model.forward(input);
    float loss_initial = model.compute_loss(output, input);

    // Train for 10 iterations
    float loss_final = loss_initial;
    for (int i = 0; i < 10; ++i)
    {
        output = model.forward(input);
        loss_final = model.compute_loss(output, input);
        model.backward(input, 0.01f);
    }

    bool success = (loss_final <= loss_initial);

    std::cout << (success ? "PASSED" : "FAILED");
    std::cout << " (initial = " << std::fixed << std::setprecision(6) << loss_initial
              << ", final = " << loss_final << ")" << std::endl;

    return success;
}

bool test_autoencoder_save_load()
{
    std::cout << "Test: Autoencoder save/load weights... ";

    AutoencoderCPU model1;

    int batch = 1;
    Tensor input({batch, 3, 32, 32});
    for (size_t i = 0; i < input.size(); ++i)
    {
        input.raw_data()[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // Forward pass with first model
    Tensor output1 = model1.forward(input);
    float loss1 = model1.compute_loss(output1, input);

    // Save weights
    std::string test_weights_file = std::string(MODEL_SAVE_DIR) + "/test_weights.bin";
    try
    {
        model1.save_weights(test_weights_file);

        // Create new model and load weights
        AutoencoderCPU model2;
        model2.load_weights(test_weights_file);

        // Forward pass with second model
        Tensor output2 = model2.forward(input);
        float loss2 = model2.compute_loss(output2, input);

        // Losses should be identical
        bool success = (std::abs(loss1 - loss2) < 1e-5f);

        std::cout << (success ? "PASSED" : "FAILED");
        std::cout << " (loss1 = " << std::fixed << std::setprecision(6) << loss1
                  << ", loss2 = " << loss2 << ")" << std::endl;

        return success;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAILED: " << e.what() << std::endl;
        return false;
    }
}

int main()
{
    std::cout << "\n=== Autoencoder Unit Tests (Tensor API) ===\n\n";

    int passed = 0, total = 0;

    total++;
    if (test_autoencoder_shape())
        passed++;
    total++;
    if (test_autoencoder_feature_extraction())
        passed++;
    total++;
    if (test_autoencoder_loss())
        passed++;
    total++;
    if (test_autoencoder_backward())
        passed++;
    total++;
    if (test_autoencoder_training_iteration())
        passed++;
    total++;
    if (test_autoencoder_save_load())
        passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n\n";

    return (passed == total) ? 0 : 1;
}
