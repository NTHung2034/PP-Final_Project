// CIFAR-10 Autoencoder Training - CPU v2 (Compact)
#include "models/autoencoder_cpu.h"
#include "data/cifar10_loader.h"
#include "data/data_types.h"
#include "utils/image_utils.h"
#include "utils/memory_tracker.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cstring>

constexpr int TRAIN_IMAGES = 100;
constexpr int SUB_EPOCHS = 5;

using std::cout;

int main()
{
    cout << "\n=== CIFAR-10 Autoencoder (CPU) ===\n";
    cout << "Config: " << TRAIN_IMAGES << " images, " << SUB_EPOCHS << " epochs, batch size="
         << BATCH_SIZE << ", learning rate=" << LEARNING_RATE << "\n\n";

    try
    {
        // Initialize memory tracking
        MemoryTracker::init();

        // Load data
        cout << "Loading CIFAR-10 dataset...\n";
        auto data_load_start = std::chrono::high_resolution_clock::now();

        CIFAR10Loader loader(CIFAR_BIN_DIR);
        loader.load_train_data();

        auto data_load_end = std::chrono::high_resolution_clock::now();
        double data_load_time = std::chrono::duration<double>(data_load_end - data_load_start).count();

        cout << "Data loading completed in " << std::fixed << std::setprecision(2) << data_load_time << "s\n";
        cout << "Memory after data load: " << MemoryTracker::format_bytes(MemoryTracker::get_current_usage()) << "\n";

        // Initialize model
        AutoencoderCPU model;
        cout << "Memory after model init: " << MemoryTracker::format_bytes(MemoryTracker::get_current_usage());

                // Training
        std::vector<float> losses;
        std::vector<double> epoch_times;
        std::vector<size_t> epoch_memory;
        auto train_start = std::chrono::high_resolution_clock::now();

        for (int ep = 0; ep < SUB_EPOCHS; ++ep)
        {
            cout << "\n\n\n--- Epoch " << (ep + 1) << "/" << SUB_EPOCHS << " ---\n";

            auto ep_start = std::chrono::high_resolution_clock::now();
            loader.shuffle();

            int batches = (TRAIN_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;
            double ep_loss = 0.0;

            for (int b = 0; b < batches; ++b)
            {
                // Get batch data as float*
                float *batch_data = loader.get_batch(BATCH_SIZE);

                // Create tensor from float* (NCHW format)
                Tensor batch_tensor({BATCH_SIZE, 3, 32, 32}, false);
                std::memcpy(batch_tensor.raw_data(), batch_data, BATCH_SIZE * 3 * 32 * 32 * sizeof(float));

                // Forward pass
                Tensor output = model.forward(batch_tensor);
                float batch_loss = model.compute_loss(output, batch_tensor);
                ep_loss += batch_loss;

                // Backward pass
                model.backward(batch_tensor, LEARNING_RATE);

                // Display batch loss
                cout << "  Batch " << std::setw(2) << (b + 1) << "/" << batches
                     << " - Loss: " << std::fixed << std::setprecision(6) << batch_loss << "\n";
            }

            ep_loss /= batches;
            losses.push_back(static_cast<float>(ep_loss));

            auto ep_end = std::chrono::high_resolution_clock::now();
            double ep_time = std::chrono::duration<double>(ep_end - ep_start).count();
            epoch_times.push_back(ep_time);

            size_t current_mem = MemoryTracker::get_current_usage();
            epoch_memory.push_back(current_mem);

            cout << "Epoch " << (ep + 1) << " Summary:\n";
            cout << "  Average Loss: " << std::setprecision(6) << ep_loss << "\n";
            cout << "  Time: " << std::setprecision(2) << ep_time << "s";
            cout << " (" << static_cast<int>(TRAIN_IMAGES / ep_time) << " img/s)\n";
            cout << "  Memory: " << MemoryTracker::format_bytes(current_mem)
                 << " (Peak: " << MemoryTracker::format_bytes(MemoryTracker::get_peak_usage()) << ")\n";

            // Save original and reconstruction images

            float *sample_data = loader.get_batch(BATCH_SIZE);
            Tensor sample_tensor({BATCH_SIZE, 3, 32, 32}, false);
            std::memcpy(sample_tensor.raw_data(), sample_data, BATCH_SIZE * 3 * 32 * 32 * sizeof(float));

            Tensor reconstructed = model.forward(sample_tensor);

            ImageUtils::save_reconstruction_samples(
                sample_tensor, reconstructed,
                std::string(MODEL_SAVE_DIR),
                "epoch_" + std::to_string(ep + 1),
                1);
        }

        auto train_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(train_end - train_start).count();

        // Summary
        cout << "\n=== Results ===\n";
        cout << "Time: " << std::setprecision(1) << total_time << "s | "
             << "Loss: " << std::setprecision(6) << losses[0] << " -> " << losses.back()
             << " (-" << std::setprecision(1) << ((losses[0] - losses.back()) / losses[0] * 100) << "%)\n";

        // Save epoch metrics (for Colab parsing)
        std::ofstream metrics(std::string(MODEL_SAVE_DIR) + "/epoch_metrics_cpu.txt");
        metrics << "Epoch,Loss,Time,MemoryMB\n";
        for (int i = 0; i < SUB_EPOCHS; ++i)
        {
            metrics << (i + 1) << ","
                    << std::fixed << std::setprecision(6) << losses[i] << ","
                    << std::setprecision(2) << epoch_times[i] << ","
                    << std::setprecision(2) << (epoch_memory[i] / (1024.0 * 1024.0)) << "\n";
        }
        metrics.close();

        // Save final trained weights
        model.save_weights(std::string(MODEL_SAVE_DIR) + "/cpu_final.bin");

        cout << "Final weights saved to: " << MODEL_SAVE_DIR << "/cpu_final.bin\n";
        cout << "Metrics saved to: " << MODEL_SAVE_DIR << "/epoch_metrics_cpu.txt\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
