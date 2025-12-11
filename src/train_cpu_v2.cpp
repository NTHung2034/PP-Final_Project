// CIFAR-10 Autoencoder Training - CPU v2 (Compact)
#include "models/autoencoder_cpu.h"
#include "data/cifar10_loader.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

constexpr int TRAIN_IMAGES = 500;
constexpr int SUB_EPOCHS = 10;

int main() {
    std::cout << "\n=== CIFAR-10 Autoencoder (CPU v2) ===\n";
    std::cout << "Config: " << TRAIN_IMAGES << " imgs, " << SUB_EPOCHS << " epochs, bs=" 
              << BATCH_SIZE << ", lr=" << LEARNING_RATE << "\n\n";

    try {
        // Load data
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        loader.load_train_data();
        std::cout << "Data: " << loader.train_size() << " images loaded\n";

        // Initialize model
        AutoencoderCPU model;
        size_t out_size = BATCH_SIZE * 3 * 32 * 32;
        
        // Initial loss check
        loader.reset();
        float* batch = loader.get_batch(BATCH_SIZE);
        float init_loss = AutoencoderCPU::compute_loss(model.forward(batch, BATCH_SIZE), batch, out_size);
        std::cout << "Initial loss: " << std::fixed << std::setprecision(6) << init_loss << "\n\n";

        // Training
        std::cout << "Epoch |   Loss   |  Time  | img/s\n";
        std::cout << "------|----------|--------|------\n";
        
        std::vector<float> losses;
        auto train_start = std::chrono::high_resolution_clock::now();

        for (int ep = 0; ep < SUB_EPOCHS; ++ep) {
            auto ep_start = std::chrono::high_resolution_clock::now();
            loader.shuffle();
            loader.reset();
            
            int batches = TRAIN_IMAGES / BATCH_SIZE;
            double ep_loss = 0.0;

            for (int b = 0; b < batches; ++b) {
                float* imgs = loader.get_batch(BATCH_SIZE);
                float* out = model.forward(imgs, BATCH_SIZE);
                ep_loss += AutoencoderCPU::compute_loss(out, imgs, out_size);
                model.backward(imgs, LEARNING_RATE);
            }

            ep_loss /= batches;
            losses.push_back(static_cast<float>(ep_loss));
            
            auto ep_end = std::chrono::high_resolution_clock::now();
            double ep_time = std::chrono::duration<double>(ep_end - ep_start).count();
            
            std::cout << std::setw(5) << (ep + 1) << " | " 
                      << std::setprecision(6) << ep_loss << " | "
                      << std::setprecision(1) << std::setw(5) << ep_time << "s | "
                      << std::setw(4) << static_cast<int>(TRAIN_IMAGES / ep_time) << "\n";

            model.save_weights(std::string(MODEL_SAVE_DIR) + "/cpu_v2_epoch_" + std::to_string(ep + 1) + ".bin");
        }

        auto train_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(train_end - train_start).count();

        // Summary
        std::cout << "\n=== Results ===\n";
        std::cout << "Time: " << std::setprecision(1) << total_time << "s | "
                  << "Loss: " << std::setprecision(6) << losses[0] << " -> " << losses.back() 
                  << " (-" << std::setprecision(1) << ((losses[0] - losses.back()) / losses[0] * 100) << "%)\n";

        // Save summary
        std::ofstream f(std::string(MODEL_SAVE_DIR) + "/training_summary_cpu_v2.txt");
        f << "CIFAR-10 Autoencoder CPU v2\n";
        f << "Epochs: " << SUB_EPOCHS << ", Images: " << TRAIN_IMAGES << ", Batch: " << BATCH_SIZE << "\n";
        f << "Total time: " << std::fixed << std::setprecision(1) << total_time << "s\n";
        for (int i = 0; i < SUB_EPOCHS; ++i)
            f << "Epoch " << (i + 1) << ": " << std::setprecision(6) << losses[i] << "\n";
        f.close();

        std::cout << "Weights saved to: " << MODEL_SAVE_DIR << "/cpu_v2_epoch_*.bin\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
