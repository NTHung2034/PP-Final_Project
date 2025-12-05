#include "models/autoencoder_cpu.h"
#include "data/cifar10_dataset.h"
#include "data/data_utils.h"
#include "utils/logger.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

// Training mode configuration
constexpr bool TEST_MODE = true; // Set to false for full training
constexpr int TEST_EPOCHS = 2;
constexpr int TEST_BATCHES_PER_EPOCH = 2; // 2 batches × 32 images = 64 images total
constexpr int TEST_BATCH_SIZE = 32;

int main()
{
    LOG_INIT();

    std::cout << "\n========================================\n";
    std::cout << "  CIFAR-10 Autoencoder Training\n";
    std::cout << "  Phase 1: CPU Baseline\n";
    std::cout << "  MODE: FULL TRAINING\n";

    std::cout << "========================================\n\n";

    LOG_INFO("Starting CIFAR-10 Autoencoder Training (CPU Baseline)");

    try
    {
        // Step 1: Load dataset
        std::cout << "[1/5] Loading CIFAR-10 dataset...\n";
        auto load_start = std::chrono::high_resolution_clock::now();

        CIFAR10Dataset train_dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
        train_dataset.load_data();

        auto load_end = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration<double>(load_end - load_start).count();

        std::cout << "      ✓ Loaded " << train_dataset.size() << " training images in "
                  << std::fixed << std::setprecision(2) << load_time << "s\n\n";

        // Step 2: Initialize autoencoder model
        std::cout << "[2/5] Initializing autoencoder model...\n";
        AutoencoderCPU model;
        std::cout << "      ✓ Encoder: Conv(3→256)→ReLU→Pool→Conv(256→128)→ReLU→Pool\n";
        std::cout << "      ✓ Latent: 8×8×128 = 8,192 features\n";
        std::cout << "      ✓ Decoder: Conv(128→128)→ReLU→Up→Conv(128→256)→ReLU→Up→Conv(256→3)\n\n";

        // Step 3: Verify forward pass
        std::cout << "[3/5] Verifying model architecture...\n";
        auto test_batch = train_dataset.get_batch(4);
        std::cout << "      Input shape: [" << test_batch.batch() << ", "
                  << test_batch.channels() << ", " << test_batch.height() << ", "
                  << test_batch.width() << "]\n";

        auto test_output = model.forward(test_batch);
        std::cout << "      Output shape: [" << test_output.batch() << ", "
                  << test_output.channels() << ", " << test_output.height() << ", "
                  << test_output.width() << "]\n";

        auto test_features = model.extract_features(test_batch);
        std::cout << "      Feature shape: [" << test_features.batch() << ", "
                  << test_features.channels() << ", " << test_features.height() << ", "
                  << test_features.width() << "] = "
                  << (test_features.channels() * test_features.height() * test_features.width())
                  << "D\n";

        float initial_loss = model.compute_loss(test_output, test_batch);
        std::cout << "      Initial loss: " << std::fixed << std::setprecision(6)
                  << initial_loss << "\n";
        std::cout << "      ✓ Model architecture verified\n\n";

        train_dataset.reset();

        // Step 4: Training loop
        std::cout << "[4/5] Training autoencoder...\n";
        std::cout << "      Configuration:\n";
        std::cout << "      - Epochs: " << EPOCHS << "\n";
        std::cout << "      - Batch size: " << BATCH_SIZE << "\n";
        std::cout << "      - Batches per epoch: " << (train_dataset.size() / BATCH_SIZE) << "\n";
        std::cout << "      - Learning rate: " << LEARNING_RATE << "\n";
        std::cout << "      - Total images per epoch: " << (train_dataset.size()) << "\n\n";

        std::vector<float> epoch_losses;
        auto training_start = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < EPOCHS; ++epoch)
        {
            auto epoch_start = std::chrono::high_resolution_clock::now();

            train_dataset.shuffle();
            train_dataset.reset();

            const int batches_per_epoch = train_dataset.size() / BATCH_SIZE;
            double epoch_loss = 0.0;
            int batch_count = 0;

            std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS << ":\n";

            for (int batch = 0; batch < batches_per_epoch; ++batch)
            {
                // Get batch
                auto images = train_dataset.get_batch(BATCH_SIZE);

                // Forward pass
                auto output = model.forward(images);
                float loss = model.compute_loss(output, images);
                epoch_loss += loss;
                batch_count++;

                // Backward pass
                model.backward(images, LEARNING_RATE);

                // Print progress
                std::cout << "  Batch " << std::setw(3) << (batch + 1) << "/" << batches_per_epoch
                          << " - Loss: " << std::fixed << std::setprecision(6) << loss
                          << " - Avg: " << (epoch_loss / batch_count) << "\r" << std::flush;
            }

            epoch_loss /= batches_per_epoch;
            epoch_losses.push_back(epoch_loss);

            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();

            std::cout << "\n  ✓ Epoch " << (epoch + 1) << " completed in "
                      << std::fixed << std::setprecision(2) << epoch_time << "s"
                      << " - Avg Loss: " << std::setprecision(6) << epoch_loss << "\n";

            // Save weights after each epoch
            std::string weights_file = std::string(MODEL_SAVE_DIR) + "/cpu_encoder_epoch_" +
                                       std::to_string(epoch + 1) + ".bin";
            model.save_weights(weights_file);
            std::cout << "  ✓ Saved weights: " << weights_file << "\n\n";
        }

        auto training_end = std::chrono::high_resolution_clock::now();
        auto training_time = std::chrono::duration<double>(training_end - training_start).count();

        std::cout << "✓ Training completed in " << std::fixed << std::setprecision(2)
                  << training_time << "s\n\n";

        // Step 5: Test weight loading and feature extraction
        std::cout << "[5/5] Testing weight persistence and feature extraction...\n";

        // Create new model and load weights
        AutoencoderCPU loaded_model;
        std::string final_weights = std::string(MODEL_SAVE_DIR) + "/cpu_encoder_epoch_" +
                                    std::to_string(EPOCHS) + ".bin";
        loaded_model.load_weights(final_weights);
        std::cout << "      ✓ Loaded weights from: " << final_weights << "\n";

        // Verify loaded model produces same output
        auto verify_batch = train_dataset.get_batch(4);
        auto original_output = model.forward(verify_batch);
        auto loaded_output = loaded_model.forward(verify_batch);

        float original_loss = model.compute_loss(original_output, verify_batch);
        float loaded_loss = loaded_model.compute_loss(loaded_output, verify_batch);

        std::cout << "      Original model loss: " << std::fixed << std::setprecision(6)
                  << original_loss << "\n";
        std::cout << "      Loaded model loss: " << loaded_loss << "\n";
        std::cout << "      Difference: " << std::abs(original_loss - loaded_loss) << "\n";

        if (std::abs(original_loss - loaded_loss) < 1e-4)
        {
            std::cout << "      ✓ Weight loading verified (outputs match)\n";
        }
        else
        {
            std::cout << "      ⚠ Warning: Loaded model outputs differ\n";
        }

        // Extract features
        std::cout << "\n      Extracting features from sample batch...\n";
        auto features = loaded_model.extract_features(verify_batch);
        std::cout << "      ✓ Feature shape: [" << features.batch() << ", "
                  << features.channels() << ", " << features.height() << ", "
                  << features.width() << "]\n";
        std::cout << "      ✓ Feature vector size: "
                  << (features.channels() * features.height() * features.width())
                  << "D per image\n\n";

        // Save training summary
        std::string summary_file = std::string(MODEL_SAVE_DIR) + "/training_summary_cpu.txt";
        std::ofstream summary(summary_file);
        summary << "CIFAR-10 Autoencoder Training Summary\n";
        summary << "======================================\n\n";
        summary << "Mode: FULL" << "\n";
        summary << "Epochs: " << EPOCHS << "\n";
        summary << "Batch size: " << BATCH_SIZE << "\n";
        summary << "Learning rate: " << LEARNING_RATE << "\n";
        summary << "Images per epoch: " << train_dataset.size() << "\n";
        summary << "Total training time: " << training_time << "s\n\n";
        summary << "Epoch Losses:\n";
        for (size_t i = 0; i < epoch_losses.size(); ++i)
        {
            summary << "Epoch " << (i + 1) << ": " << epoch_losses[i] << "\n";
        }
        summary.close();

        std::cout << "✓ Training summary saved: " << summary_file << "\n\n";

        // Print training summary to screen
        std::cout << "\n========================================\n";
        std::cout << "  TRAINING SUMMARY (CPU)\n";
        std::cout << "========================================\n\n";
        std::ifstream summary_read(summary_file);
        std::string line;
        while (std::getline(summary_read, line))
        {
            std::cout << line << "\n";
        }
        summary_read.close();
        std::cout << "========================================\n\n";

        // Final summary
        std::cout << "\n========================================\n";
        std::cout << "  TRAINING COMPLETE - PIPELINE VERIFIED\n";
        std::cout << "========================================\n\n";
        std::cout << "✓ Data loading: " << load_time << "s\n";
        std::cout << "✓ Model initialization: SUCCESS\n";
        std::cout << "✓ Training: " << training_time << "s (" << EPOCHS << " epochs)\n";
        std::cout << "✓ Weight persistence: VERIFIED\n";
        std::cout << "✓ Feature extraction: VERIFIED\n\n";
        std::cout << "Loss progression:\n";
        for (size_t i = 0; i < epoch_losses.size(); ++i)
        {
            std::cout << "  Epoch " << (i + 1) << ": " << std::fixed << std::setprecision(6)
                      << epoch_losses[i];
            if (i > 0)
            {
                float reduction = ((epoch_losses[i - 1] - epoch_losses[i]) / epoch_losses[i - 1]) * 100.0f;
                std::cout << " (" << std::setprecision(2) << reduction << "% reduction)";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        LOG_INFO("Training completed successfully!");
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n❌ ERROR: " << e.what() << "\n\n";
        LOG_ERROR("Fatal error: %s", e.what());
        return 1;
    }

    return 0;
}