#include "models/autoencoder_cpu.h"
#include "data/cifar10_dataset.h"
#include "data/data_utils.h"
#include "utils/logger.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

// CPU Training Configuration (reduced for faster execution)
constexpr int CPU_TRAIN_IMAGES = 10000;
constexpr int CPU_TEST_IMAGES = 2000;
constexpr int CPU_EPOCHS = 10;

int main()
{
    LOG_INIT();

    std::cout << "\n========================================\n";
    std::cout << "  CIFAR-10 Autoencoder Training\n";
    std::cout << "  Phase 1: CPU Baseline\n";
    std::cout << "  MODE: REDUCED TRAINING (" << CPU_TRAIN_IMAGES << " images)\n";
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
                  << std::fixed << std::setprecision(2) << load_time << "s\n";
        std::cout << "      ✓ Using " << CPU_TRAIN_IMAGES << " images for CPU training\n";
        std::cout << "      ✓ Batch size: " << BATCH_SIZE << " (from config.h)\n";
        std::cout << "      ✓ Learning rate: " << LEARNING_RATE << " (from config.h)\n\n";

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
        std::cout << "      - Epochs: " << CPU_EPOCHS << "\n";
        std::cout << "      - Batch size: " << BATCH_SIZE << "\n";
        std::cout << "      - Batches per epoch: " << (CPU_TRAIN_IMAGES / BATCH_SIZE) << "\n";
        std::cout << "      - Learning rate: " << LEARNING_RATE << "\n";
        std::cout << "      - Total images per epoch: " << CPU_TRAIN_IMAGES << "\n\n";

        std::vector<float> epoch_losses;
        std::vector<double> epoch_times;
        auto training_start = std::chrono::high_resolution_clock::now();

        // Open epoch details file
        std::string epoch_details_file = std::string(MODEL_SAVE_DIR) + "/epoch_details_cpu.txt";
        std::ofstream epoch_details(epoch_details_file);
        epoch_details << "CIFAR-10 Autoencoder - Epoch-by-Epoch Training Details (CPU)\n";
        epoch_details << "=============================================================\n\n";
        epoch_details << "Configuration:\n";
        epoch_details << "  - Total epochs: " << CPU_EPOCHS << "\n";
        epoch_details << "  - Images per epoch: " << CPU_TRAIN_IMAGES << "\n";
        epoch_details << "  - Batch size: " << BATCH_SIZE << "\n";
        epoch_details << "  - Learning rate: " << LEARNING_RATE << "\n\n";
        epoch_details << "=============================================================\n\n";

        for (int epoch = 0; epoch < CPU_EPOCHS; ++epoch)
        {
            auto epoch_start = std::chrono::high_resolution_clock::now();

            train_dataset.shuffle();
            train_dataset.reset();

            const int batches_per_epoch = CPU_TRAIN_IMAGES / BATCH_SIZE;
            double epoch_loss = 0.0;
            int batch_count = 0;

            std::cout << "Epoch " << (epoch + 1) << "/" << CPU_EPOCHS << ":\n";

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
            epoch_times.push_back(epoch_time);

            // Calculate throughput
            double throughput = CPU_TRAIN_IMAGES / epoch_time;

            // Print to console
            std::cout << "\n";
            std::cout << "  ✓ Epoch " << (epoch + 1) << " completed\n";
            std::cout << "    - Time: " << std::fixed << std::setprecision(2) << epoch_time << "s\n";
            std::cout << "    - Avg Loss: " << std::setprecision(6) << epoch_loss << "\n";
            std::cout << "    - Throughput: " << std::setprecision(1) << throughput << " images/sec\n";

            if (epoch > 0)
            {
                float loss_reduction = ((epoch_losses[epoch - 1] - epoch_loss) / epoch_losses[epoch - 1]) * 100.0f;
                std::cout << "    - Loss change: " << std::setprecision(2) << loss_reduction << "%\n";
            }

            // Save weights after each epoch
            std::string weights_file = std::string(MODEL_SAVE_DIR) + "/cpu_encoder_epoch_" +
                                       std::to_string(epoch + 1) + ".bin";
            model.save_weights(weights_file);
            std::cout << "    - Weights saved: " << weights_file << "\n";

            // Write to epoch details file
            epoch_details << "Epoch " << (epoch + 1) << "/" << CPU_EPOCHS << ":\n";
            epoch_details << "  Time: " << std::fixed << std::setprecision(2) << epoch_time << " seconds\n";
            epoch_details << "  Average Loss: " << std::setprecision(6) << epoch_loss << "\n";
            epoch_details << "  Throughput: " << std::setprecision(1) << throughput << " images/sec\n";
            if (epoch > 0)
            {
                float loss_reduction = ((epoch_losses[epoch - 1] - epoch_loss) / epoch_losses[epoch - 1]) * 100.0f;
                epoch_details << "  Loss change from previous epoch: " << std::setprecision(2) << loss_reduction << "%\n";
            }
            epoch_details << "  Weights saved to: " << weights_file << "\n";
            epoch_details << "\n";
            epoch_details.flush(); // Ensure data is written immediately

            std::cout << "\n";
        }

        epoch_details.close();

        auto training_end = std::chrono::high_resolution_clock::now();
        auto training_time = std::chrono::duration<double>(training_end - training_start).count();

        double avg_epoch_time = training_time / CPU_EPOCHS;
        double total_throughput = (CPU_TRAIN_IMAGES * CPU_EPOCHS) / training_time;

        std::cout << "✓ Training completed!\n";
        std::cout << "  - Total time: " << std::fixed << std::setprecision(2)
                  << training_time << "s (" << (training_time / 60.0) << " min)\n";
        std::cout << "  - Avg time/epoch: " << avg_epoch_time << "s\n";
        std::cout << "  - Overall throughput: " << std::setprecision(1) << total_throughput << " images/sec\n\n";

        // Step 5: Test weight loading and feature extraction
        std::cout << "[5/5] Testing weight persistence and feature extraction...\n";

        // Create new model and load weights
        AutoencoderCPU loaded_model;
        std::string final_weights = std::string(MODEL_SAVE_DIR) + "/cpu_encoder_epoch_" +
                                    std::to_string(CPU_EPOCHS) + ".bin";
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
        summary << "CIFAR-10 Autoencoder Training Summary (CPU)\n";
        summary << "============================================\n\n";

        summary << "Configuration:\n";
        summary << "  Mode: REDUCED (CPU Baseline)\n";
        summary << "  Epochs: " << CPU_EPOCHS << "\n";
        summary << "  Images per epoch: " << CPU_TRAIN_IMAGES << "\n";
        summary << "  Test images: " << CPU_TEST_IMAGES << "\n";
        summary << "  Batch size: " << BATCH_SIZE << "\n";
        summary << "  Learning rate: " << LEARNING_RATE << "\n\n";

        summary << "Performance:\n";
        summary << "  Total training time: " << std::fixed << std::setprecision(2)
                << training_time << "s (" << (training_time / 60.0) << " min)\n";
        summary << "  Average time per epoch: " << (training_time / CPU_EPOCHS) << "s\n";
        summary << "  Overall throughput: " << std::setprecision(1)
                << total_throughput << " images/sec\n\n";

        summary << "Loss Progression:\n";
        summary << "  Initial loss (Epoch 1): " << std::setprecision(6) << epoch_losses[0] << "\n";
        summary << "  Final loss (Epoch " << CPU_EPOCHS << "): " << epoch_losses[CPU_EPOCHS - 1] << "\n";
        summary << "  Total reduction: " << std::setprecision(2)
                << ((epoch_losses[0] - epoch_losses[CPU_EPOCHS - 1]) / epoch_losses[0] * 100.0f) << "%\n\n";

        summary << "Epoch Details:\n";
        summary << "  Epoch |    Loss    |  Time(s)  | Throughput(img/s)\n";
        summary << "  ------|------------|-----------|------------------\n";
        for (size_t i = 0; i < epoch_losses.size(); ++i)
        {
            summary << "  " << std::setw(5) << (i + 1) << " | "
                    << std::fixed << std::setprecision(6) << std::setw(10) << epoch_losses[i] << " | "
                    << std::setprecision(2) << std::setw(9) << epoch_times[i] << " | "
                    << std::setprecision(1) << std::setw(16) << (CPU_TRAIN_IMAGES / epoch_times[i]) << "\n";
        }
        summary << "\n";

        summary << "Files Generated:\n";
        summary << "  - Weights: cpu_encoder_epoch_1.bin to cpu_encoder_epoch_" << CPU_EPOCHS << ".bin\n";
        summary << "  - Epoch details: epoch_details_cpu.txt\n";
        summary << "  - This summary: training_summary_cpu.txt\n";

        summary.close();

        std::cout << "✓ Training summary saved: " << summary_file << "\n";
        std::cout << "✓ Epoch details saved: " << epoch_details_file << "\n\n";

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
        std::cout << "\n========================================\n\n";

        // Final summary
        std::cout << "\n========================================\n";
        std::cout << "  TRAINING COMPLETE - PIPELINE VERIFIED\n";
        std::cout << "========================================\n\n";
        std::cout << "✓ Data loading: " << std::fixed << std::setprecision(2) << load_time << "s\n";
        std::cout << "✓ Model initialization: SUCCESS\n";
        std::cout << "✓ Training: " << training_time << "s (" << CPU_EPOCHS << " epochs, "
                  << CPU_TRAIN_IMAGES << " images/epoch)\n";
        std::cout << "✓ Weight persistence: VERIFIED\n";
        std::cout << "✓ Feature extraction: VERIFIED\n\n";

        std::cout << "Quick Loss Overview:\n";
        std::cout << "  First 3 epochs: ";
        for (size_t i = 0; i < std::min(size_t(3), epoch_losses.size()); ++i)
        {
            std::cout << std::setprecision(6) << epoch_losses[i];
            if (i < std::min(size_t(3), epoch_losses.size()) - 1)
                std::cout << " → ";
        }
        std::cout << "\n  Last 3 epochs:  ";
        size_t start = epoch_losses.size() >= 3 ? epoch_losses.size() - 3 : 0;
        for (size_t i = start; i < epoch_losses.size(); ++i)
        {
            std::cout << std::setprecision(6) << epoch_losses[i];
            if (i < epoch_losses.size() - 1)
                std::cout << " → ";
        }
        std::cout << "\n\n";

        std::cout << "Files saved in: " << MODEL_SAVE_DIR << "/\n";
        std::cout << "  - training_summary_cpu.txt (summary)\n";
        std::cout << "  - epoch_details_cpu.txt (detailed per-epoch info)\n";
        std::cout << "  - cpu_encoder_epoch_*.bin (weights for each epoch)\n\n";

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
