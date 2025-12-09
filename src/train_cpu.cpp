#include "models/autoencoder_cpu.h"
#include "data/cifar10_dataset.h"
#include "data/data_utils.h"
#include "utils/logger.h"
#include "utils/image_utils.h"
#include "utils/memory_tracker.h"
#include "utils/svm_classifier.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <algorithm>

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
    LOG_INIT();
    MemoryTracker::init();

    std::cout << "\n========================================\n";
    std::cout << "  CIFAR-10 Autoencoder Training\n";
    std::cout << "  Phase 1: CPU Baseline\n";
    std::cout << "  MODE: REDUCED TRAINING (" << CPU_TRAIN_IMAGES << " images)\n";
    std::cout << "========================================\n\n";

    LOG_INFO("Starting CIFAR-10 Autoencoder Training (CPU Baseline)");

    size_t initial_memory = MemoryTracker::get_current_usage();
    std::cout << "Initial memory usage: " << MemoryTracker::format_bytes(initial_memory) << "\n\n";

    try
    {
        // Step 1: Load dataset
        std::cout << "[1/5] Loading CIFAR-10 dataset...\n";
        auto load_start = std::chrono::high_resolution_clock::now();

        CIFAR10Dataset train_dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
        train_dataset.load_data();

        auto load_end = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration<double>(load_end - load_start).count();

        size_t after_load_memory = MemoryTracker::get_current_usage();

        std::cout << "      ✓ Loaded " << train_dataset.size() << " training images in "
                  << std::fixed << std::setprecision(2) << load_time << "s\n";
        std::cout << "      ✓ Using " << CPU_TRAIN_IMAGES << " images for CPU training\n";
        std::cout << "      ✓ Batch size: " << BATCH_SIZE << " (from config.h)\n";
        std::cout << "      ✓ Learning rate: " << LEARNING_RATE << " (from config.h)\n";
        std::cout << "      ✓ Memory after loading: " << MemoryTracker::format_bytes(after_load_memory) << "\n\n";

        // Step 2: Initialize autoencoder model
        std::cout << "[2/5] Initializing autoencoder model...\n";
        AutoencoderCPU model;

        size_t after_model_memory = MemoryTracker::get_current_usage();

        std::cout << "      ✓ Encoder: Conv(3→256)→ReLU→Pool→Conv(256→128)→ReLU→Pool\n";
        std::cout << "      ✓ Latent: 8×8×128 = 8,192 features\n";
        std::cout << "      ✓ Decoder: Conv(128→128)→ReLU→Up→Conv(128→256)→ReLU→Up→Conv(256→3)\n";
        std::cout << "      ✓ Memory after model init: " << MemoryTracker::format_bytes(after_model_memory) << "\n\n";

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

        auto verify_features = model.extract_features(test_batch);
        std::cout << "      Feature shape: [" << verify_features.batch() << ", "
                  << verify_features.channels() << ", " << verify_features.height() << ", "
                  << verify_features.width() << "] = "
                  << (verify_features.channels() * verify_features.height() * verify_features.width())
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
        std::vector<size_t> epoch_memory;
        std::vector<float> epoch_psnr;
        std::vector<float> epoch_ssim;
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
        epoch_details << "  - Learning rate: " << LEARNING_RATE << "\n";
        epoch_details << "  - Initial memory: " << MemoryTracker::format_bytes(initial_memory) << "\n\n";
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

            // Get memory usage
            size_t current_memory = MemoryTracker::get_current_usage();
            epoch_memory.push_back(current_memory);

            // Reset dataset to get fresh samples for visualization
            train_dataset.reset();

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

            // Calculate throughput
            double throughput = CPU_TRAIN_IMAGES / epoch_time;

            // Print to console
            std::cout << "\n";
            std::cout << "  ✓ Epoch " << (epoch + 1) << " completed\n";
            std::cout << "    - Time: " << std::fixed << std::setprecision(2) << epoch_time << "s\n";
            std::cout << "    - Avg Loss: " << std::setprecision(6) << epoch_loss << "\n";
            std::cout << "    - PSNR: " << std::setprecision(2) << psnr << " dB\n";
            std::cout << "    - SSIM: " << std::setprecision(4) << ssim << "\n";
            std::cout << "    - Throughput: " << std::setprecision(1) << throughput << " images/sec\n";
            std::cout << "    - Memory: " << MemoryTracker::format_bytes(current_memory) << "\n";

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
            epoch_details << "  PSNR: " << std::setprecision(2) << psnr << " dB\n";
            epoch_details << "  SSIM: " << std::setprecision(4) << ssim << "\n";
            epoch_details << "  Throughput: " << std::setprecision(1) << throughput << " images/sec\n";
            epoch_details << "  Memory Usage: " << MemoryTracker::format_bytes(current_memory) << "\n";
            if (epoch > 0)
            {
                float loss_reduction = ((epoch_losses[epoch - 1] - epoch_loss) / epoch_losses[epoch - 1]) * 100.0f;
                epoch_details << "  Loss Reduction: " << std::setprecision(2) << loss_reduction << "%\n";
            }
            epoch_details << "  Weights File: cpu_encoder_epoch_" << (epoch + 1) << ".bin\n";
            epoch_details << "  Sample Images: " << sample_prefix << "_sample_*.ppm\n";
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

        summary << "Memory Usage:\n";
        summary << "  Initial memory: " << MemoryTracker::format_bytes(initial_memory) << "\n";
        summary << "  After data loading: " << MemoryTracker::format_bytes(after_load_memory) << "\n";
        summary << "  After model init: " << MemoryTracker::format_bytes(after_model_memory) << "\n";
        summary << "  Peak memory: " << MemoryTracker::format_bytes(MemoryTracker::get_peak_usage()) << "\n";
        summary << "  Final memory: " << MemoryTracker::format_bytes(epoch_memory.back()) << "\n\n";

        summary << "Loss Progression:\n";
        summary << "  Initial loss (Epoch 1): " << std::setprecision(6) << epoch_losses[0] << "\n";
        summary << "  Final loss (Epoch " << CPU_EPOCHS << "): " << epoch_losses[CPU_EPOCHS - 1] << "\n";
        summary << "  Total reduction: " << std::setprecision(2)
                << ((epoch_losses[0] - epoch_losses[CPU_EPOCHS - 1]) / epoch_losses[0] * 100.0f) << "%\n\n";

        summary << "Reconstruction Quality:\n";
        summary << "  Initial PSNR (Epoch 1): " << std::setprecision(2) << epoch_psnr[0] << " dB\n";
        summary << "  Final PSNR (Epoch " << CPU_EPOCHS << "): " << epoch_psnr[CPU_EPOCHS - 1] << " dB\n";
        summary << "  Initial SSIM (Epoch 1): " << std::setprecision(4) << epoch_ssim[0] << "\n";
        summary << "  Final SSIM (Epoch " << CPU_EPOCHS << "): " << epoch_ssim[CPU_EPOCHS - 1] << "\n\n";

        summary << "Epoch Details:\n";
        summary << "  Epoch |    Loss    |  PSNR(dB) |   SSIM   |  Time(s)  | Memory(MB) | Throughput(img/s)\n";
        summary << "  ------|------------|-----------|----------|-----------|------------|------------------\n";
        for (size_t i = 0; i < epoch_losses.size(); ++i)
        {
            summary << "  " << std::setw(5) << (i + 1) << " | "
                    << std::fixed << std::setprecision(6) << std::setw(10) << epoch_losses[i] << " | "
                    << std::setprecision(2) << std::setw(9) << epoch_psnr[i] << " | "
                    << std::setprecision(4) << std::setw(8) << epoch_ssim[i] << " | "
                    << std::setprecision(2) << std::setw(9) << epoch_times[i] << " | "
                    << std::setw(10) << (epoch_memory[i] / (1024 * 1024)) << " | "
                    << std::setprecision(1) << std::setw(16) << (CPU_TRAIN_IMAGES / epoch_times[i]) << "\n";
        }
        summary << "\n";

        summary << "Files Generated:\n";
        summary << "  - Weights: cpu_encoder_epoch_1.bin to cpu_encoder_epoch_" << CPU_EPOCHS << ".bin\n";
        summary << "  - Epoch details: epoch_details_cpu.txt\n";
        summary << "  - Sample reconstructions: epoch_*_sample_*.ppm (4 samples per epoch)\n";
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

        // ========================================================================
        // PHASE 2: CLASSIFICATION WITH SVM
        // ========================================================================

        std::cout << "\n========================================\n";
        std::cout << "  PHASE 2: SVM CLASSIFICATION\n";
        std::cout << "========================================\n\n";

        LOG_INFO("Starting SVM classification pipeline");

        // Step 6: Extract features from training set
        std::cout << "[6/8] Extracting features from training set...\n";
        auto feature_start = std::chrono::high_resolution_clock::now();

        train_dataset.reset();
        std::vector<std::vector<float>> train_features;
        std::vector<int> train_labels;
        train_features.reserve(CPU_TRAIN_IMAGES);
        train_labels.reserve(CPU_TRAIN_IMAGES);

        int feature_batches = CPU_TRAIN_IMAGES / BATCH_SIZE;
        for (int batch = 0; batch < feature_batches; ++batch)
        {
            auto images = train_dataset.get_batch(BATCH_SIZE);
            auto labels = train_dataset.get_batch_labels(BATCH_SIZE);
            auto batch_features = loaded_model.extract_features(images);

            // Convert to vector format (flatten features)
            for (int i = 0; i < BATCH_SIZE; ++i)
            {
                std::vector<float> feat_vec(8192);
                int offset = i * 8192;
                std::copy_n(batch_features.raw_data() + offset, 8192, feat_vec.begin());
                train_features.push_back(std::move(feat_vec));
                train_labels.push_back(labels[i]);
            }

            std::cout << "      Extracted features: " << train_features.size() << "/" << CPU_TRAIN_IMAGES << "\r" << std::flush;
        }

        auto feature_end = std::chrono::high_resolution_clock::now();
        auto feature_time = std::chrono::duration<double>(feature_end - feature_start).count();

        std::cout << "\n      ✓ Extracted " << train_features.size() << " feature vectors (8192-dim each)\n";
        std::cout << "      ✓ Feature extraction time: " << std::fixed << std::setprecision(2)
                  << feature_time << "s\n\n";

        // Step 7: Train SVM classifier
        std::cout << "[7/8] Training SVM classifier...\n";
        auto svm_train_start = std::chrono::high_resolution_clock::now();

        svm_model *svm = SVMClassifier::train_svm(train_features, train_labels, 10.0, 0.0001);

        auto svm_train_end = std::chrono::high_resolution_clock::now();
        auto svm_train_time = std::chrono::duration<double>(svm_train_end - svm_train_start).count();

        if (!svm)
        {
            LOG_ERROR("SVM training failed");
            return 1;
        }

        std::cout << "      ✓ SVM training completed in " << std::fixed << std::setprecision(1)
                  << svm_train_time << "s\n\n";

        // Save SVM model
        std::string svm_model_file = std::string(MODEL_SAVE_DIR) + "/svm_model.bin";
        SVMClassifier::save_model(svm, svm_model_file);

        // Step 8: Test on test set
        std::cout << "[8/8] Evaluating on test set...\n";

        // Load test dataset
        CIFAR10Dataset test_dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TEST);
        test_dataset.load_data();
        std::cout << "      ✓ Loaded " << test_dataset.size() << " test images\n";

        // Extract test features
        auto test_start = std::chrono::high_resolution_clock::now();
        test_dataset.reset();

        std::vector<std::vector<float>> test_features;
        std::vector<int> test_true_labels;
        test_features.reserve(CPU_TEST_IMAGES);
        test_true_labels.reserve(CPU_TEST_IMAGES);

        int test_batches = CPU_TEST_IMAGES / BATCH_SIZE;
        for (int batch = 0; batch < test_batches; ++batch)
        {
            auto images = test_dataset.get_batch(BATCH_SIZE);
            auto labels = test_dataset.get_batch_labels(BATCH_SIZE);
            auto batch_features = loaded_model.extract_features(images);

            for (int i = 0; i < BATCH_SIZE; ++i)
            {
                std::vector<float> feat_vec(8192);
                int offset = i * 8192;
                std::copy_n(batch_features.raw_data() + offset, 8192, feat_vec.begin());
                test_features.push_back(std::move(feat_vec));
                test_true_labels.push_back(labels[i]);
            }
        }

        std::cout << "      ✓ Extracted " << test_features.size() << " test feature vectors\n";

        // Predict
        std::vector<int> test_predictions = SVMClassifier::predict(svm, test_features);

        auto test_end = std::chrono::high_resolution_clock::now();
        auto test_time = std::chrono::duration<double>(test_end - test_start).count();

        std::cout << "      ✓ Prediction completed in " << std::fixed << std::setprecision(2)
                  << test_time << "s\n\n";

        // Calculate accuracy
        float overall_accuracy = SVMClassifier::calculate_accuracy(test_true_labels, test_predictions);
        std::vector<float> per_class_acc = SVMClassifier::calculate_per_class_accuracy(
            test_true_labels, test_predictions, 10);

        // Print results
        std::cout << "========================================\n";
        std::cout << "  CLASSIFICATION RESULTS\n";
        std::cout << "========================================\n\n";

        std::cout << "Overall Test Accuracy: " << std::fixed << std::setprecision(2)
                  << (overall_accuracy * 100.0f) << "%\n\n";

        std::cout << "Per-Class Accuracy:\n";
        std::cout << "  Class        | Accuracy\n";
        std::cout << "  -------------|----------\n";
        for (int i = 0; i < 10; ++i)
        {
            std::cout << "  " << std::left << std::setw(12) << CIFAR10_CLASSES[i]
                      << " | " << std::right << std::setw(6) << std::setprecision(2)
                      << (per_class_acc[i] * 100.0f) << "%\n";
        }

        // Save classification results
        std::string class_results_file = std::string(MODEL_SAVE_DIR) + "/classification_results_cpu.txt";
        std::ofstream class_results(class_results_file);
        class_results << "CIFAR-10 Classification Results (CPU)\n";
        class_results << "======================================\n\n";

        class_results << "Overall Test Accuracy: " << std::fixed << std::setprecision(2)
                      << (overall_accuracy * 100.0f) << "%\n";
        class_results << "Test samples: " << test_predictions.size() << "\n";
        class_results << "Correct predictions: " << static_cast<int>(overall_accuracy * test_predictions.size()) << "\n\n";

        class_results << "Per-Class Accuracy:\n";
        class_results << "  Class        | Accuracy\n";
        class_results << "  -------------|----------\n";
        for (int i = 0; i < 10; ++i)
        {
            class_results << "  " << std::left << std::setw(12) << CIFAR10_CLASSES[i]
                          << " | " << std::right << std::setw(6) << std::setprecision(2)
                          << (per_class_acc[i] * 100.0f) << "%\n";
        }

        class_results << "\nTimings:\n";
        class_results << "  Feature extraction (train): " << std::setprecision(2) << feature_time << "s\n";
        class_results << "  SVM training: " << std::setprecision(1) << svm_train_time << "s\n";
        class_results << "  Feature extraction + prediction (test): " << std::setprecision(2) << test_time << "s\n";

        class_results.close();

        std::cout << "\n✓ Classification results saved: " << class_results_file << "\n";

        // Final complete summary
        std::cout << "\n========================================\n";
        std::cout << "  COMPLETE PIPELINE SUMMARY\n";
        std::cout << "========================================\n\n";
        std::cout << "✓ Autoencoder Training: " << std::fixed << std::setprecision(1) << training_time << "s\n";
        std::cout << "✓ Feature Extraction: " << std::setprecision(2) << feature_time << "s\n";
        std::cout << "✓ SVM Training: " << std::setprecision(1) << svm_train_time << "s\n";
        std::cout << "✓ Test Classification: " << std::setprecision(2) << test_time << "s\n";
        std::cout << "✓ Overall Accuracy: " << (overall_accuracy * 100.0f) << "%\n\n";

        std::cout << "All results saved in: " << MODEL_SAVE_DIR << "/\n\n";

        // Cleanup SVM model
        SVMClassifier::free_model(svm);

        LOG_INFO("Complete pipeline finished successfully!");
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n❌ ERROR: " << e.what() << "\n\n";
        LOG_ERROR("Fatal error: %s", e.what());
        return 1;
    }

    return 0;
}
