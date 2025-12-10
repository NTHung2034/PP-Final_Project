#include "models/autoencoder_cpu.h"
#include "utils/logger.h"
#include <iostream>
#include <iomanip>
#include <cmath>

bool test_autoencoder_shape()
{
    std::cout << "Test: Autoencoder forward pass shape... ";

    AutoencoderCPU model;
    Tensor input({1, 3, 32, 32}); // Single CIFAR-10 image

    // Initialize input with random values
    float *data = input.data->data();
    for (size_t i = 0; i < input.size(); ++i)
    {
        data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    Tensor output = model.forward(input);

    bool success = (output.batch() == 1 &&
                    output.channels() == 3 &&
                    output.height() == 32 &&
                    output.width() == 32);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    if (!success)
    {
        std::cout << "  Expected: [1, 3, 32, 32], Got: ["
                  << output.batch() << ", " << output.channels() << ", "
                  << output.height() << ", " << output.width() << "]" << std::endl;
    }
    return success;
}

bool test_autoencoder_feature_extraction()
{
    std::cout << "Test: Autoencoder feature extraction... ";

    AutoencoderCPU model;
    Tensor input({2, 3, 32, 32}); // Batch of 2 images

    // Initialize input
    float *data = input.data->data();
    for (size_t i = 0; i < input.size(); ++i)
    {
        data[i] = 0.5f;
    }

    Tensor features = model.extract_features(input);

    // Features should be [2, 128, 8, 8] = [2, 8192]
    bool success = (features.batch() == 2 &&
                    features.channels() == 128 &&
                    features.height() == 8 &&
                    features.width() == 8);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    if (!success)
    {
        std::cout << "  Expected: [2, 128, 8, 8], Got: ["
                  << features.batch() << ", " << features.channels() << ", "
                  << features.height() << ", " << features.width() << "]" << std::endl;
    }
    return success;
}

bool test_autoencoder_loss()
{
    std::cout << "Test: Autoencoder MSE loss computation... ";

    AutoencoderCPU model;
    Tensor input({1, 3, 32, 32});

    // Initialize input
    float *data = input.data->data();
    for (size_t i = 0; i < input.size(); ++i)
    {
        data[i] = 0.5f;
    }

    Tensor output = model.forward(input);
    float loss = model.compute_loss(output, input);

    // Loss should be non-negative
    bool success = (loss >= 0.0f);

    std::cout << (success ? "PASSED" : "FAILED");
    std::cout << " (loss = " << std::fixed << std::setprecision(6) << loss << ")" << std::endl;

    return success;
}

bool test_autoencoder_backward()
{
    std::cout << "Test: Autoencoder backward pass... ";

    AutoencoderCPU model;
    Tensor input({1, 3, 32, 32});

    // Initialize input
    float *data = input.data->data();
    for (size_t i = 0; i < input.size(); ++i)
    {
        data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // Forward pass
    Tensor output = model.forward(input);
    float loss_before = model.compute_loss(output, input);

    // Backward pass
    try
    {
        model.backward(input, 0.001f);
        std::cout << "PASSED (loss before = " << std::fixed << std::setprecision(6)
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
    Tensor input({4, 3, 32, 32}); // Small batch

    // Initialize input with consistent pattern
    float *data = input.data->data();
    for (size_t i = 0; i < input.size(); ++i)
    {
        data[i] = 0.5f + 0.1f * std::sin(i * 0.01f);
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
        model.backward(input, 0.01f); // Higher learning rate for faster convergence
    }

    // Loss should decrease (model is learning)
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
    Tensor input({1, 3, 32, 32});

    // Initialize input
    float *data = input.data->data();
    for (size_t i = 0; i < input.size(); ++i)
    {
        data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // Forward pass with first model
    Tensor output1 = model1.forward(input);

    // Save weights
    std::string test_weights_file = "models/saved_weights/test_weights.bin";
    try
    {
        model1.save_weights(test_weights_file);

        // Create new model and load weights
        AutoencoderCPU model2;
        model2.load_weights(test_weights_file);

        // Forward pass with second model
        Tensor output2 = model2.forward(input);

        // Outputs should be identical
        bool outputs_match = true;
        const float *out1_data = output1.data->data();
        const float *out2_data = output2.data->data();

        for (size_t i = 0; i < output1.size(); ++i)
        {
            if (std::abs(out1_data[i] - out2_data[i]) > 1e-5f)
            {
                outputs_match = false;
                break;
            }
        }

        std::cout << (outputs_match ? "PASSED" : "FAILED") << std::endl;
        return outputs_match;
    }
    catch (const std::exception &e)
    {
        std::cout << "FAILED: " << e.what() << std::endl;
        return false;
    }
}

int main()
{
    LOG_INIT();

    std::cout << "\n========================================\n";
    std::cout << "  Autoencoder Unit Tests\n";
    std::cout << "========================================\n\n";

    int passed = 0;
    int total = 0;

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

    std::cout << "\n========================================\n";
    std::cout << "  Results: " << passed << "/" << total << " tests passed\n";
    std::cout << "========================================\n\n";

    return (passed == total) ? 0 : 1;
}
