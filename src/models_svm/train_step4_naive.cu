#include "models/autoencoder_gpu_naive.cuh"
#include "data/cifar10_loader.h"
#include "config.h"
#include "../../external/svm.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>  

// Pretty printing helpers
void print_confusion_matrix(const std::vector<std::vector<int>>& cm,
                            const std::vector<std::string>& class_names) {
    std::cout << "\n=== Confusion Matrix ===" << std::endl;
    std::cout << std::setw(12) << " ";
    for (const auto& name : class_names) {
        std::cout << std::setw(12) << name.substr(0, 10);
    }
    std::cout << std::endl;
    
    for (size_t i = 0; i < cm.size(); ++i) {
        std::cout << std::setw(12) << class_names[i].substr(0, 10);
        for (size_t j = 0; j < cm[i].size(); ++j) {
            std::cout << std::setw(12) << cm[i][j];
        }
        std::cout << std::endl;
    }
}

void print_per_class_accuracy(const std::vector<std::vector<int>>& cm,
                              const std::vector<std::string>& class_names) {
    std::cout << "\n=== Per-Class Accuracy ===" << std::endl;
    for (size_t i = 0; i < cm.size(); ++i) {
        int total = 0;
        for (int val : cm[i]) total += val;
        float acc = (total > 0) ? (100.0f * cm[i][i] / total) : 0.0f;
        std::cout << std::setw(15) << class_names[i] << ": "
                  << std::fixed << std::setprecision(2) << acc << "%" << std::endl;
    }
}

int main(int argc, char** argv) {
    try {
        // Configuration
        std::string data_dir = CIFAR_BIN_DIR;
        std::string weights_dir = MODEL_SAVE_DIR_GPU_NAIVE;
        if (argc >= 2) data_dir = argv[1];

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║   CIFAR-10 AUTOENCODER - NAIVE GPU + SVM (PHASE 4)            ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        std::vector<std::string> class_names = {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        };

        // ====== Step 1: Load Dataset ======
        std::cout << "=== Step 1: Loading Dataset ===" << std::endl;
        
        CIFAR10Loader train_loader(data_dir);
        train_loader.load_train_data();
        train_loader.load_test_data();

        int num_train_samples = train_loader.train_size();
        int num_test_samples = train_loader.test_size();

        std::cout << "Training images: " << num_train_samples << std::endl;
        std::cout << "Test images: " << num_test_samples << std::endl;

        // ====== Step 2: Load Encoder Weights ======
        std::cout << "\n=== Step 2: Loading Encoder Weights ===" << std::endl;

        GPUAutoencoderNaive autoencoder;
        
        // Load weights helper (same style as train_gpu_naive.cu)
        auto load_conv_weights = [&](GPUConvWeights* weights, const std::string& name) {
            std::string weight_file = weights_dir + "/" + name + "_weights.bin";
            std::string bias_file = weights_dir + "/" + name + "_bias.bin";
            
            std::ifstream wf(weight_file, std::ios::binary);
            std::ifstream bf(bias_file, std::ios::binary);
            
            if (!wf.is_open() || !bf.is_open()) {
                std::cerr << "Failed to load weights for " << name << std::endl;
                return false;
            }
            
            std::vector<float> h_weights(weights->weight_size);
            std::vector<float> h_bias(weights->bias_size);
            
            wf.read(reinterpret_cast<char*>(h_weights.data()), weights->weight_size * sizeof(float));
            bf.read(reinterpret_cast<char*>(h_bias.data()), weights->bias_size * sizeof(float));
            
            CUDA_CHECK(cudaMemcpy(weights->d_weights, h_weights.data(), 
                                 weights->weight_size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(weights->d_bias, h_bias.data(), 
                                 weights->bias_size * sizeof(float), cudaMemcpyHostToDevice));
            
            wf.close();
            bf.close();
            return true;
        };
        
        // Load encoder weights (conv1 and conv2 only for feature extraction)
        if (!load_conv_weights(autoencoder.conv1, "conv1") ||
            !load_conv_weights(autoencoder.conv2, "conv2")) {
            std::cerr << "Failed to load weights from: " << weights_dir << std::endl;
            return 1;
        }
        
        std::cout << "Encoder weights loaded" << std::endl;

        // ====== Step 3: Extract Features ======
        std::cout << "\n=== Step 3: Extracting Features ===" << std::endl;
        
        const int feature_batch_size = 100;
        const int feature_dim = 8192;  // 128 * 8 * 8
        
        std::vector<float> train_features;
        std::vector<float> test_features;
        train_features.reserve(num_train_samples * feature_dim);
        test_features.reserve(num_test_samples * feature_dim);
        
        auto extract_start = std::chrono::high_resolution_clock::now();
        
        // Extract training features
        train_loader.reset();
        int num_train_batches = (num_train_samples + feature_batch_size - 1) / feature_batch_size;
        
        for (int batch_idx = 0; batch_idx < num_train_batches; batch_idx++) {
            int start_idx = batch_idx * feature_batch_size;
            int actual_batch_size = std::min(feature_batch_size, num_train_samples - start_idx);
            
            float* batch_data = train_loader.get_batch_at(start_idx, actual_batch_size);
            
            GPUTensor input(actual_batch_size, 3, 32, 32, false);
            GPUTensor features(actual_batch_size, 128, 8, 8, false);
            
            memcpy(input.h_data, batch_data, actual_batch_size * 3 * 32 * 32 * sizeof(float));
            input.copyToDevice();
            
            autoencoder.extract_features(input, features);
            
            features.copyToHost();
            
            for (int i = 0; i < actual_batch_size; i++) {
                float* feature_ptr = features.h_data + i * feature_dim;
                train_features.insert(train_features.end(), feature_ptr, feature_ptr + feature_dim);
            }
            
            if ((batch_idx + 1) % 50 == 0 || batch_idx == num_train_batches - 1) {
                std::cout << "  Train batch [" << (batch_idx + 1) << "/" << num_train_batches << "]" << std::endl;
            }
        }
        
        // Extract test features
        int num_test_batches = (num_test_samples + feature_batch_size - 1) / feature_batch_size;
        
        for (int batch_idx = 0; batch_idx < num_test_batches; batch_idx++) {
            int start_idx = batch_idx * feature_batch_size;
            int actual_batch_size = std::min(feature_batch_size, num_test_samples - start_idx);
            
            float* batch_data = train_loader.get_batch_at(start_idx + num_train_samples, actual_batch_size);
            if (batch_data == nullptr) {
                // Use test data directly
                batch_data = train_loader.test_images() + start_idx * 3 * 32 * 32;
            }
            
            GPUTensor input(actual_batch_size, 3, 32, 32, false);
            GPUTensor features(actual_batch_size, 128, 8, 8, false);
            
            memcpy(input.h_data, batch_data, actual_batch_size * 3 * 32 * 32 * sizeof(float));
            input.copyToDevice();
            
            autoencoder.extract_features(input, features);
            
            features.copyToHost();
            
            for (int i = 0; i < actual_batch_size; i++) {
                float* feature_ptr = features.h_data + i * feature_dim;
                test_features.insert(test_features.end(), feature_ptr, feature_ptr + feature_dim);
            }
            
            if ((batch_idx + 1) % 20 == 0 || batch_idx == num_test_batches - 1) {
                std::cout << "  Test batch [" << (batch_idx + 1) << "/" << num_test_batches << "]" << std::endl;
            }
        }

        auto extract_end = std::chrono::high_resolution_clock::now();
        float extract_time = std::chrono::duration<float>(extract_end - extract_start).count();

        std::cout << "Feature extraction completed: " << std::fixed << std::setprecision(2) 
                  << extract_time << "s" << std::endl;
        
        // Get labels
        std::vector<int> train_labels_int(train_loader.train_labels(), 
                                          train_loader.train_labels() + num_train_samples);
        std::vector<int> test_labels_int(train_loader.test_labels(), 
                                         train_loader.test_labels() + num_test_samples);
        
        // Normalize features
        std::cout << "Normalizing features..." << std::endl;
        std::vector<float> feature_mean(feature_dim, 0.0f);
        std::vector<float> feature_std(feature_dim, 0.0f);
        
        for (int i = 0; i < num_train_samples; i++) {
            for (int j = 0; j < feature_dim; j++) {
                feature_mean[j] += train_features[i * feature_dim + j];
            }
        }
        for (int j = 0; j < feature_dim; j++) {
            feature_mean[j] /= num_train_samples;
        }
        
        for (int i = 0; i < num_train_samples; i++) {
            for (int j = 0; j < feature_dim; j++) {
                float diff = train_features[i * feature_dim + j] - feature_mean[j];
                feature_std[j] += diff * diff;
            }
        }
        for (int j = 0; j < feature_dim; j++) {
            feature_std[j] = sqrt(feature_std[j] / num_train_samples);
            if (feature_std[j] < 1e-8f) feature_std[j] = 1.0f;
        }
        
        for (int i = 0; i < num_train_samples; i++) {
            for (int j = 0; j < feature_dim; j++) {
                int idx = i * feature_dim + j;
                train_features[idx] = (train_features[idx] - feature_mean[j]) / feature_std[j];
            }
        }
        
        for (int i = 0; i < num_test_samples; i++) {
            for (int j = 0; j < feature_dim; j++) {
                int idx = i * feature_dim + j;
                test_features[idx] = (test_features[idx] - feature_mean[j]) / feature_std[j];
            }
        }
        
        std::vector<uint8_t> train_labels(train_labels_int.begin(), train_labels_int.end());
        std::vector<uint8_t> test_labels(test_labels_int.begin(), test_labels_int.end());

        // ====== Step 4: Train SVM ======
        std::cout << "\n=== Step 4: Training SVM ===" << std::endl;

        SVMClassifier svm;
        auto svm_start = std::chrono::high_resolution_clock::now();

        svm.train(train_features, train_labels, train_labels.size(), feature_dim);

        auto svm_end = std::chrono::high_resolution_clock::now();
        float svm_time = std::chrono::duration<float>(svm_end - svm_start).count();

        std::cout << "SVM training completed: " << std::fixed << std::setprecision(2)
                  << svm_time << "s" << std::endl;

        // ====== Step 5: Evaluate ======
        std::cout << "\n=== Step 5: Evaluation ===" << std::endl;

        std::vector<uint8_t> predictions;
        svm.predict(test_features, test_labels.size(), feature_dim, predictions);
        
        float accuracy = svm.evaluate(predictions, test_labels);

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) 
                  << (accuracy * 100.0f) << "%" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::vector<std::vector<int>> confusion_matrix;
        svm.compute_confusion_matrix(predictions, test_labels, 10, confusion_matrix);

        print_confusion_matrix(confusion_matrix, class_names);
        print_per_class_accuracy(confusion_matrix, class_names);

        std::cout << "\n=== Timing Summary ===" << std::endl;
        std::cout << "Feature extraction: " << std::fixed << std::setprecision(2) 
                  << extract_time << "s" << std::endl;
        std::cout << "SVM training: " << svm_time << "s" << std::endl;
        std::cout << "Total: " << (extract_time + svm_time) << "s" << std::endl;
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    TRAINING COMPLETED!                         ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
