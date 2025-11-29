/**
 * @file train_cpu.cpp
 * @brief CPU baseline training for the CIFAR-10 Autoencoder
 * 
 * This file implements the Phase 1 CPU training loop for the autoencoder.
 * It serves as the baseline for performance comparison with GPU implementations.
 * 
 * Training process:
 * 1. Load CIFAR-10 training data (50,000 images)
 * 2. For each epoch:
 *    - Shuffle the dataset
 *    - For each batch:
 *      - Forward pass: input → encoder → decoder → output
 *      - Compute reconstruction loss (MSE)
 *      - Backward pass: compute gradients
 *      - Update weights using SGD
 * 3. Save trained model weights
 * 
 * Usage:
 *   ./train_cpu [--epochs N] [--batch-size N] [--learning-rate F]
 * 
 * Expected performance (CPU baseline):
 *   - Training time: ~15-20 minutes per epoch
 *   - Final loss: ~0.05-0.08 after 5 epochs
 *   - Memory usage: ~4-6 GB RAM
 * 
 * Reference:
 * - docs/PHASE_1_GUIDE.md for detailed implementation guide
 * - docs/PROJECT_PLAN.md for project timeline
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#include "models/autoencoder_cpu.h"
#include "data/cifar10_dataset.h"
#include "utils/logger.h"
#include "config.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstring>

/**
 * @brief Parse command line arguments
 */
struct TrainingConfig {
    int epochs = EPOCHS;
    int batch_size = BATCH_SIZE;
    float learning_rate = LEARNING_RATE;
    std::string data_dir = CIFAR_BIN_DIR;
    std::string save_dir = MODEL_SAVE_DIR;
    bool verbose = true;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --epochs N        Number of training epochs (default: " << EPOCHS << ")\n"
              << "  --batch-size N    Batch size (default: " << BATCH_SIZE << ")\n"
              << "  --learning-rate F Learning rate (default: " << LEARNING_RATE << ")\n"
              << "  --data-dir PATH   Path to CIFAR-10 data (default: " << CIFAR_BIN_DIR << ")\n"
              << "  --save-dir PATH   Path to save model weights (default: " << MODEL_SAVE_DIR << ")\n"
              << "  --quiet           Disable verbose output\n"
              << "  --help            Show this help message\n";
}

TrainingConfig parse_args(int argc, char* argv[]) {
    TrainingConfig config;
    
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            config.epochs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            config.batch_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            config.learning_rate = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            config.data_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--save-dir") == 0 && i + 1 < argc) {
            config.save_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--quiet") == 0) {
            config.verbose = false;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    
    return config;
}

/**
 * @brief Format duration in human-readable format
 */
std::string format_duration(double seconds) {
    if (seconds < 60) {
        return std::to_string(static_cast<int>(seconds)) + "s";
    } else if (seconds < 3600) {
        int mins = static_cast<int>(seconds / 60);
        int secs = static_cast<int>(seconds) % 60;
        return std::to_string(mins) + "m " + std::to_string(secs) + "s";
    } else {
        int hours = static_cast<int>(seconds / 3600);
        int mins = (static_cast<int>(seconds) % 3600) / 60;
        return std::to_string(hours) + "h " + std::to_string(mins) + "m";
    }
}

int main(int argc, char* argv[]) {
    // Initialize logging
    LOG_INIT();
    
    // Parse command line arguments
    TrainingConfig config = parse_args(argc, argv);
    
    LOG_INFO("========================================");
    LOG_INFO("CIFAR-10 Autoencoder - CPU Training");
    LOG_INFO("========================================");
    LOG_INFO("Configuration:");
    LOG_INFO("  Epochs:        %d", config.epochs);
    LOG_INFO("  Batch size:    %d", config.batch_size);
    LOG_INFO("  Learning rate: %.6f", config.learning_rate);
    LOG_INFO("  Data dir:      %s", config.data_dir.c_str());
    LOG_INFO("  Save dir:      %s", config.save_dir.c_str());
    LOG_INFO("========================================");
    
    try {
        // =========================================
        // Step 1: Load Dataset
        // =========================================
        
        LOG_INFO("Loading CIFAR-10 training data...");
        auto load_start = std::chrono::high_resolution_clock::now();
        
        CIFAR10Dataset train_dataset(config.data_dir, CIFAR10Dataset::Mode::TRAIN);
        train_dataset.load_data();
        
        auto load_end = std::chrono::high_resolution_clock::now();
        double load_time = std::chrono::duration<double>(load_end - load_start).count();
        
        LOG_INFO("Dataset loaded: %zu images in %.2f seconds", 
                 train_dataset.size(), load_time);
        
        // =========================================
        // Step 2: Create Model
        // =========================================
        
        LOG_INFO("Creating autoencoder model...");
        AutoencoderCPU model;
        LOG_INFO("Model created with %zu parameters", model.get_num_parameters());
        
        // =========================================
        // Step 3: Training Loop
        // =========================================
        
        const int batches_per_epoch = train_dataset.size() / config.batch_size;
        double total_training_time = 0.0;
        
        LOG_INFO("Starting training...");
        LOG_INFO("  Batches per epoch: %d", batches_per_epoch);
        
        auto training_start = std::chrono::high_resolution_clock::now();
        
        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            // Shuffle dataset at the beginning of each epoch
            train_dataset.shuffle();
            train_dataset.reset();
            
            double epoch_loss = 0.0;
            int batch_count = 0;
            
            for (int batch = 0; batch < batches_per_epoch; ++batch) {
                // Get batch of images
                Tensor images = train_dataset.get_batch(config.batch_size);
                
                // Forward pass
                Tensor output = model.forward(images);
                
                // Backward pass and weight update
                // Note: For autoencoder, target is the same as input
                float batch_loss = model.backward(images, config.learning_rate);
                
                epoch_loss += batch_loss;
                batch_count++;
                
                // Log progress
                if (config.verbose && (batch % 100 == 0 || batch == batches_per_epoch - 1)) {
                    std::cout << "\rEpoch " << std::setw(2) << (epoch + 1) << "/" << config.epochs
                              << " | Batch " << std::setw(4) << (batch + 1) << "/" << batches_per_epoch
                              << " | Loss: " << std::fixed << std::setprecision(6) << batch_loss
                              << std::flush;
                }
            }
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            double epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();
            total_training_time += epoch_time;
            
            double avg_loss = epoch_loss / batch_count;
            
            std::cout << std::endl;
            LOG_INFO("Epoch %d/%d completed in %s | Avg Loss: %.6f",
                     epoch + 1, config.epochs, 
                     format_duration(epoch_time).c_str(), avg_loss);
            
            // Estimate remaining time
            if (epoch < config.epochs - 1) {
                double avg_epoch_time = total_training_time / (epoch + 1);
                double remaining_time = avg_epoch_time * (config.epochs - epoch - 1);
                LOG_INFO("  Estimated remaining: %s", format_duration(remaining_time).c_str());
            }
        }
        
        auto training_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(training_end - training_start).count();
        
        // =========================================
        // Step 4: Save Model
        // =========================================
        
        std::string model_path = std::string(config.save_dir) + "/cpu_encoder_weights.bin";
        model.save_weights(model_path);
        
        // =========================================
        // Step 5: Print Summary
        // =========================================
        
        LOG_INFO("========================================");
        LOG_INFO("Training Summary");
        LOG_INFO("========================================");
        LOG_INFO("Total training time: %s", format_duration(total_time).c_str());
        LOG_INFO("Average time/epoch:  %s", format_duration(total_time / config.epochs).c_str());
        LOG_INFO("Model saved to:      %s", model_path.c_str());
        LOG_INFO("========================================");
        LOG_INFO("CPU Baseline Phase 1 Complete!");
        LOG_INFO("========================================");
        
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal error: %s", e.what());
        return 1;
    }
}
