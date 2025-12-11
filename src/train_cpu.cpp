#include "models/autoencoder_cpu.h"
#include "data/cifar10_dataset.h"
#include "data/data_utils.h"
#include "utils/image_utils.h"
#include "utils/memory_tracker.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <algorithm>

using std::cout;

// CPU Training Configuration (reduced for faster execution)
constexpr int CPU_TRAIN_IMAGES = 1000;
constexpr int CPU_TEST_IMAGES = 200;
constexpr int CPU_EPOCHS = 2;

// CIFAR-10 class names
const char *CIFAR10_CLASSES[] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"};

int main()
{
    MemoryTracker::init();

    cout << "CIFAR-10 Autoencoder Training\n";
    cout << "Phase 1: CPU Baseline (Reduced)\n";

    size_t initial_memory = MemoryTracker::get_current_usage();
    cout << "Initial memory usage: " << MemoryTracker::format_bytes(initial_memory);

    try
    {
        // Step 1: Load dataset
        auto load_start = std::chrono::high_resolution_clock::now();

        CIFAR10Dataset train_dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
        train_dataset.load_data();

        auto load_end = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration<double>(load_end - load_start).count();

        size_t after_load_memory = MemoryTracker::get_current_usage();

        cout << "Loaded " << train_dataset.size() << " training images in "
             << std::fixed << std::setprecision(2) << load_time << "s\n";
        cout << "\nMemory after loading: " << MemoryTracker::format_bytes(after_load_memory) << " bytes\n\n";

        // Step 2: Initialize autoencoder model
        AutoencoderCPU model;

        size_t after_model_memory = MemoryTracker::get_current_usage();
        cout << "Memory after model init: " << MemoryTracker::format_bytes(after_model_memory);

        // Step 3: Training loop
        cout << "\nTraining autoencoder...";
        cout << "\nConfiguration:";
        cout << "\nEpochs: " << CPU_EPOCHS;
        cout << "\nBatch size: " << BATCH_SIZE;

        const int BATCHES_PER_EPOCH = CPU_TRAIN_IMAGES / BATCH_SIZE;
        cout << "\nBatches per epoch: " << BATCHES_PER_EPOCH;
        cout << "\nLearning rate: " << LEARNING_RATE;
        cout << "\nTotal images per epoch: " << CPU_TRAIN_IMAGES;

        std::vector<float> mean_epoch_losses;
        std::vector<double> epoch_times;
        std::vector<size_t> epoch_memory;
        std::vector<float> epoch_psnr;
        std::vector<float> epoch_ssim;
        auto training_start = std::chrono::high_resolution_clock::now();

        // Open epoch details file
        std::string epoch_details_file = std::string(MODEL_SAVE_DIR) + "/epoch_details_cpu.txt";
        std::ofstream epoch_details(epoch_details_file);
        epoch_details << "CIFAR-10 Autoencoder - Epoch-by-Epoch Training Details (CPU)\n\n";
        epoch_details << "\nConfiguration:";
        epoch_details << "\nTotal epochs: " << CPU_EPOCHS;
        epoch_details << "\nImages per epoch: " << CPU_TRAIN_IMAGES;
        epoch_details << "\nBatch size: " << BATCH_SIZE;
        epoch_details << "\nLearning rate: " << LEARNING_RATE;
        epoch_details << "\nInitial memory: " << MemoryTracker::format_bytes(initial_memory);

        for (int epoch = 0; epoch < CPU_EPOCHS; ++epoch)
        {
            auto epoch_start = std::chrono::high_resolution_clock::now();

            train_dataset.shuffle();

            double epoch_loss = 0.0;
            int batch_count = 0;

            cout << "\nEpoch " << (epoch + 1) << "/" << CPU_EPOCHS; // 0+1 = 1, 2, 3...

            for (int batch = 0; batch < BATCHES_PER_EPOCH; ++batch)
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
                cout << "\nBatch " << (batch + 1) << "/" << BATCHES_PER_EPOCH
                     << "\nLoss: " << std::fixed << std::setprecision(6) << loss
                     << "\nAvg: " << (epoch_loss / batch_count);
            }

            epoch_loss /= BATCHES_PER_EPOCH; // mean loss in an epoch
            mean_epoch_losses.push_back(epoch_loss);

            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();
            epoch_times.push_back(epoch_time);

            // Get memory usage
            size_t current_memory = MemoryTracker::get_current_usage();
            epoch_memory.push_back(current_memory);

            // Calculate reconstruction quality metrics on a sample batch
            auto sample_batch = train_dataset.get_batch(8);
            auto sample_output = model.forward(sample_batch);
            float psnr = ImageUtils::calculate_psnr(sample_batch, sample_output);
            float ssim = ImageUtils::calculate_ssim(sample_batch, sample_output);
            epoch_psnr.push_back(psnr);
            epoch_ssim.push_back(ssim);

            // Save reconstruction samples every epoch
            std::string sample_prefix = "epoch_" + std::to_string(epoch + 1);
            ImageUtils::save_reconstruction_samples(sample_batch, sample_output,
                                                    MODEL_SAVE_DIR, sample_prefix, 4);

            // Save weights after each epoch
            std::string weights_file = std::string(MODEL_SAVE_DIR) + "/cpu_encoder_epoch_" +
                                       std::to_string(epoch + 1) + ".bin";
            model.save_weights(weights_file);
            cout << "Weights saved: " << weights_file;

            // Write to epoch details file
            epoch_details << "\nEpoch " << (epoch + 1) << "/" << CPU_EPOCHS;
            epoch_details << "\nTime: " << std::fixed << std::setprecision(2) << epoch_time << " seconds";
            epoch_details << "\nAverage Loss: " << std::setprecision(6) << epoch_loss << "\n";
            epoch_details << "\nPSNR: " << std::setprecision(2) << psnr << " dB";
            epoch_details << "\nSSIM: " << std::setprecision(4) << ssim;
            epoch_details << "\nMemory Usage: " << MemoryTracker::format_bytes(current_memory);
            if (epoch > 0)
            {
                float loss_reduction = ((mean_epoch_losses[epoch - 1] - epoch_loss) / mean_epoch_losses[epoch - 1]) * 100.0f;
                epoch_details << "\nLoss Reduction: " << std::setprecision(2) << loss_reduction << "%\n";
            }
            epoch_details << "\nWeights File: cpu_encoder_epoch_" << (epoch + 1) << ".bin\n";
            epoch_details << "\nSample Images: " << sample_prefix << "_sample_*.ppm\n"
                          << std::endl;
        }

        epoch_details.close();

        auto training_end = std::chrono::high_resolution_clock::now();
        auto training_time = std::chrono::duration<double>(training_end - training_start).count();

        cout << "Training completed!\n";
        cout << "Total time: " << std::fixed << std::setprecision(2)
             << training_time << "s (" << (training_time / 60.0) << " min)\n";

        // Step 4: Feature Extraction Phase
        // Extract features from ALL CIFAR-10 images (50k train + 10k test) using trained encoder
        cout << "\n\n\nFEATURE EXTRACTION PHASE\n\n\n";

        // Load trained encoder weights
        std::string final_weights = std::string(MODEL_SAVE_DIR) + "/cpu_encoder_epoch_" +
                                    std::to_string(CPU_EPOCHS) + ".bin";
        model.load_weights(final_weights);
        cout << "\nLoaded trained encoder: " << final_weights;

        CIFAR10Dataset test_dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TEST);
        test_dataset.load_data();

        train_dataset.reset();
        train_dataset.shuffle();

        const int FEATURE_DIM = 8192;

        cout << "Extracting training features from " << train_dataset.size() << " images...\n";
        auto extract_train_start = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<float>> train_features;
        train_features.reserve(train_dataset.size());

        train_dataset.reset();
        int train_batches = (train_dataset.size() + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int batch_idx = 0; batch_idx < train_batches; ++batch_idx)
        {
            // Get batch and extract features (encoder only, no decoder)
            int current_batch_size = std::min(BATCH_SIZE,
                                              (int)train_dataset.size() - batch_idx * BATCH_SIZE);
            auto batch = train_dataset.get_batch(current_batch_size);
            auto latent = model.extract_features(batch);

            // Flatten each image's features into 1D vector
            const float *latent_data = latent.raw_data();
            for (int i = 0; i < current_batch_size; ++i)
            {
                std::vector<float> feature_vec(FEATURE_DIM);
                std::copy(latent_data + i * FEATURE_DIM,
                          latent_data + (i + 1) * FEATURE_DIM,
                          feature_vec.begin());
                train_features.push_back(std::move(feature_vec));
            }

            if ((batch_idx + 1) % 100 == 0)
            {
                cout << "  Processed " << train_features.size() << " / "
                     << train_dataset.size() << " images\n";
            }
        }

        auto extract_train_end = std::chrono::high_resolution_clock::now();
        auto train_extract_time = std::chrono::duration<double>(extract_train_end - extract_train_start).count();
        cout << "Training features extracted in " << std::fixed << std::setprecision(2)
             << train_extract_time << "s\n\n";

        // Extract test features (10,000 images)
        cout << "Extracting test features from " << test_dataset.size() << " images...\n";
        auto extract_test_start = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<float>> test_features;
        test_features.reserve(test_dataset.size());

        test_dataset.reset();
        int test_batches = (test_dataset.size() + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int batch_idx = 0; batch_idx < test_batches; ++batch_idx)
        {
            // Get batch and extract features (encoder only)
            int current_batch_size = std::min(BATCH_SIZE,
                                              (int)test_dataset.size() - batch_idx * BATCH_SIZE);
            auto batch = test_dataset.get_batch(current_batch_size);
            auto latent = model.extract_features(batch);

            // Flatten each image's features into 1D vector
            const float *latent_data = latent.raw_data();
            for (int i = 0; i < current_batch_size; ++i)
            {
                std::vector<float> feature_vec(FEATURE_DIM);
                std::copy(latent_data + i * FEATURE_DIM,
                          latent_data + (i + 1) * FEATURE_DIM,
                          feature_vec.begin());
                test_features.push_back(std::move(feature_vec));
            }

            if ((batch_idx + 1) % 50 == 0 || batch_idx == test_batches - 1)
            {
                cout << "  Processed " << test_features.size() << " / "
                     << test_dataset.size() << " images\n";
            }
        }

        auto extract_test_end = std::chrono::high_resolution_clock::now();
        auto test_extract_time = std::chrono::duration<double>(extract_test_end - extract_test_start).count();
        cout << "✓ Test features extracted in " << std::fixed << std::setprecision(2)
             << test_extract_time << "s\n\n";

        cout << "Feature extraction complete:\n";
        cout << "  Training features: " << train_features.size() << " x " << FEATURE_DIM << "\n";
        cout << "  Test features: " << test_features.size() << " x " << FEATURE_DIM << "\n";
        cout << "  Total extraction time: " << (train_extract_time + test_extract_time) << "s\n\n";

        // Save training summary
        std::string summary_file = std::string(MODEL_SAVE_DIR) + "/training_summary_cpu.txt";
        std::ofstream summary(summary_file);
        summary << "CIFAR-10 Autoencoder Training Summary (CPU)\n";
        summary << "============================================\n\n";

        summary << "\nConfiguration:\n";
        summary << "\nCPU Baseline (Reduced)";
        summary << "\nEpochs: " << CPU_EPOCHS;
        summary << "\nImages per epoch: " << CPU_TRAIN_IMAGES << "\n";
        summary << "Test images: " << CPU_TEST_IMAGES;
        summary << "\nBatch size: " << BATCH_SIZE;
        summary << "\nLearning rate: " << LEARNING_RATE;

        summary << "\n\nPerformance:\n";
        summary << "Total training time: " << std::fixed << std::setprecision(2)
                << training_time << "s (" << (training_time / 60.0) << " min)\n";
        summary << "Average time per epoch: " << (training_time / CPU_EPOCHS) << "s\n";

        summary << "\n\nFeature Extraction:\n";
        summary << "Training features extracted: " << train_features.size() << " x " << FEATURE_DIM << "\n";
        summary << "Test features extracted: " << test_features.size() << " x " << FEATURE_DIM << "\n";
        summary << "Feature extraction time: " << (train_extract_time + test_extract_time) << "s\n";

        summary << "\n\nEpoch Details:\n";
        summary << "  Epoch |    Loss    |  PSNR(dB) |   SSIM   |  Time(s)  | Memory(MB) |\n";
        summary << "  ------|------------|-----------|----------|-----------|------------|\n";
        for (size_t i = 0; i < mean_epoch_losses.size(); ++i)
        {
            summary << "  " << std::setw(5) << (i + 1) << " | "
                    << std::fixed << std::setprecision(6) << std::setw(10) << mean_epoch_losses[i] << " | "
                    << std::setprecision(2) << std::setw(9) << epoch_psnr[i] << " | "
                    << std::setprecision(4) << std::setw(8) << epoch_ssim[i] << " | "
                    << std::setprecision(2) << std::setw(9) << epoch_times[i] << " | "
                    << std::setw(10) << (epoch_memory[i] / (1024 * 1024)) << " | "
                    << std::setprecision(1) << std::setw(16) << (CPU_TRAIN_IMAGES / epoch_times[i]) << "\n";
        }
        summary << std::endl;
        summary.close();

        cout << "\n\nTraining summary saved: " << summary_file;
        cout << "\nEpoch details saved: " << epoch_details_file << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 1;
    }

    return 0;
}
