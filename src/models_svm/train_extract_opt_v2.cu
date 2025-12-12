#include "models/autoencoder_gpu_opt_v2.cuh"
#include "data/cifar10_loader.h"
#include "config.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>

int main(int argc, char** argv) {
    try {
        // Configuration
        std::string data_dir = CIFAR_BIN_DIR;
        std::string weights_dir = MODEL_SAVE_DIR_GPU_OPT_V2;
        if (argc >= 2) data_dir = argv[1];

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║   CIFAR-10 FEATURE EXTRACTION - GPU OPT V2 (PHASE 4)          ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

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

        // Create autoencoder with batch size for feature extraction
        const int feature_batch_size = 100;
        AutoencoderGPUOptV2 autoencoder(feature_batch_size);
        
        // Load weights helper (adapted for opt v2)
        auto load_conv_weights = [&](GPUConvWeightsOpt* weights, const std::string& name) {
            std::string weight_file = weights_dir + "/" + name + "_w.bin";
            std::string bias_file = weights_dir + "/" + name + "_b.bin";
            
            std::ifstream wf(weight_file, std::ios::binary);
            std::ifstream bf(bias_file, std::ios::binary);
            
            if (!wf.is_open() || !bf.is_open()) {
                std::cerr << "Failed to load weights for " << name << std::endl;
                std::cerr << "  Weight file: " << weight_file << std::endl;
                std::cerr << "  Bias file: " << bias_file << std::endl;
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
        
        std::cout << "Encoder weights loaded successfully" << std::endl;

        // ====== Step 3: Extract Features ======
        std::cout << "\n=== Step 3: Extracting Features ===" << std::endl;
        
        const int feature_dim = 8192;  // 128 * 8 * 8
        
        std::vector<float> train_features;
        std::vector<float> test_features;
        train_features.reserve(num_train_samples * feature_dim);
        test_features.reserve(num_test_samples * feature_dim);
        
        // Allocate host buffer for extracted features
        std::vector<float> h_features(feature_batch_size * feature_dim);
        
        auto extract_start = std::chrono::high_resolution_clock::now();
        
        // Extract training features
        train_loader.reset();
        int num_train_batches = (num_train_samples + feature_batch_size - 1) / feature_batch_size;
        
        for (int batch_idx = 0; batch_idx < num_train_batches; batch_idx++) {
            int start_idx = batch_idx * feature_batch_size;
            int actual_batch_size = std::min(feature_batch_size, num_train_samples - start_idx);
            
            float* batch_data = train_loader.get_batch_at(start_idx, actual_batch_size);
            
            // Copy input to device
            CUDA_CHECK(cudaMemcpy(autoencoder.input_buffer.d_data, batch_data, 
                                  actual_batch_size * 3 * 32 * 32 * sizeof(float), 
                                  cudaMemcpyHostToDevice));
            
            // Run encoder forward pass (using fused kernels)
            conv2d_forward_opt_v2(autoencoder.input_buffer, *autoencoder.conv1, 
                                 autoencoder.pool.act1, true);  // Conv1+ReLU (fused)
            maxpool2d_forward_opt_v2(autoencoder.pool.act1, autoencoder.pool.act2, 
                                    autoencoder.pool.pool1_idx); // Pool1
            conv2d_forward_opt_v2(autoencoder.pool.act2, *autoencoder.conv2, 
                                 autoencoder.pool.act3, true);  // Conv2+ReLU (fused)
            maxpool2d_forward_opt_v2(autoencoder.pool.act3, autoencoder.pool.act4, 
                                    autoencoder.pool.pool2_idx); // Pool2 (features)
            
            // Copy features back to host
            int feature_size = actual_batch_size * feature_dim;
            CUDA_CHECK(cudaMemcpy(h_features.data(), autoencoder.pool.act4.d_data,
                                  feature_size * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Append to train features
            train_features.insert(train_features.end(), h_features.begin(), 
                                 h_features.begin() + feature_size);
            
            if ((batch_idx + 1) % 50 == 0 || batch_idx == num_train_batches - 1) {
                std::cout << "  Train batch [" << (batch_idx + 1) << "/" << num_train_batches << "]" << std::endl;
            }
        }
        
        // Extract test features
        int num_test_batches = (num_test_samples + feature_batch_size - 1) / feature_batch_size;

        for (int batch_idx = 0; batch_idx < num_test_batches; batch_idx++) {
            int start_idx = batch_idx * feature_batch_size;
            int actual_batch_size = std::min(feature_batch_size, num_test_samples - start_idx);
            
            // Access test data directly
            float* batch_data = train_loader.test_images() + start_idx * 3 * 32 * 32;
            
            // Copy input to device
            CUDA_CHECK(cudaMemcpy(autoencoder.input_buffer.d_data, batch_data, 
                                  actual_batch_size * 3 * 32 * 32 * sizeof(float), 
                                  cudaMemcpyHostToDevice));
            
            // Run encoder forward pass (using fused kernels)
            conv2d_forward_opt_v2(autoencoder.input_buffer, *autoencoder.conv1, 
                                 autoencoder.pool.act1, true);  // Conv1+ReLU (fused)
            maxpool2d_forward_opt_v2(autoencoder.pool.act1, autoencoder.pool.act2, 
                                    autoencoder.pool.pool1_idx); // Pool1
            conv2d_forward_opt_v2(autoencoder.pool.act2, *autoencoder.conv2, 
                                 autoencoder.pool.act3, true);  // Conv2+ReLU (fused)
            maxpool2d_forward_opt_v2(autoencoder.pool.act3, autoencoder.pool.act4, 
                                    autoencoder.pool.pool2_idx); // Pool2 (features)
            
            // Copy features back to host
            int feature_size = actual_batch_size * feature_dim;
            CUDA_CHECK(cudaMemcpy(h_features.data(), autoencoder.pool.act4.d_data,
                                  feature_size * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Append to test features
            test_features.insert(test_features.end(), h_features.begin(), 
                                h_features.begin() + feature_size);
            
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
        
        std::vector<uint8_t> train_labels(train_labels_int.begin(), train_labels_int.end());
        std::vector<uint8_t> test_labels(test_labels_int.begin(), test_labels_int.end());

        // ====== Step 4: Save Features to .bin Files ======
        std::cout << "\n=== Step 4: Saving Features to .bin Files ===" << std::endl;

        std::string output_dir = weights_dir + "/svm_features";
        std::string mkdir_cmd = "mkdir -p " + output_dir;
        system(mkdir_cmd.c_str());

        // Save train features
        std::string train_features_file = output_dir + "/train_features.bin";
        std::ofstream train_features_stream(train_features_file, std::ios::binary);
        if (!train_features_stream.is_open()) {
            std::cerr << "Failed to open " << train_features_file << std::endl;
            return 1;
        }
        train_features_stream.write(reinterpret_cast<const char*>(train_features.data()), 
                                     train_features.size() * sizeof(float));
        train_features_stream.close();
        std::cout << "Saved train features: " << train_features_file << std::endl;
        std::cout << "  Shape: [" << num_train_samples << ", " << feature_dim << "]" << std::endl;

        // Save train labels
        std::string train_labels_file = output_dir + "/train_labels.bin";
        std::ofstream train_labels_stream(train_labels_file, std::ios::binary);
        if (!train_labels_stream.is_open()) {
            std::cerr << "Failed to open " << train_labels_file << std::endl;
            return 1;
        }
        train_labels_stream.write(reinterpret_cast<const char*>(train_labels.data()), 
                                   train_labels.size() * sizeof(uint8_t));
        train_labels_stream.close();
        std::cout << "Saved train labels: " << train_labels_file << std::endl;
        std::cout << "  Shape: [" << num_train_samples << "]" << std::endl;

        // Save test features
        std::string test_features_file = output_dir + "/test_features.bin";
        std::ofstream test_features_stream(test_features_file, std::ios::binary);
        if (!test_features_stream.is_open()) {
            std::cerr << "Failed to open " << test_features_file << std::endl;
            return 1;
        }
        test_features_stream.write(reinterpret_cast<const char*>(test_features.data()), 
                                    test_features.size() * sizeof(float));
        test_features_stream.close();
        std::cout << "Saved test features: " << test_features_file << std::endl;
        std::cout << "  Shape: [" << num_test_samples << ", " << feature_dim << "]" << std::endl;

        // Save test labels
        std::string test_labels_file = output_dir + "/test_labels.bin";
        std::ofstream test_labels_stream(test_labels_file, std::ios::binary);
        if (!test_labels_stream.is_open()) {
            std::cerr << "Failed to open " << test_labels_file << std::endl;
            return 1;
        }
        test_labels_stream.write(reinterpret_cast<const char*>(test_labels.data()), 
                                  test_labels.size() * sizeof(uint8_t));
        test_labels_stream.close();
        std::cout << "Saved test labels: " << test_labels_file << std::endl;
        std::cout << "  Shape: [" << num_test_samples << "]" << std::endl;

        // Save metadata
        std::string metadata_file = output_dir + "/metadata.txt";
        std::ofstream metadata_stream(metadata_file);
        if (metadata_stream.is_open()) {
            metadata_stream << "num_train_samples: " << num_train_samples << std::endl;
            metadata_stream << "num_test_samples: " << num_test_samples << std::endl;
            metadata_stream << "feature_dim: " << feature_dim << std::endl;
            metadata_stream << "num_classes: 10" << std::endl;
            metadata_stream << "data_type: float32 (features), uint8 (labels)" << std::endl;
            metadata_stream.close();
            std::cout << "Saved metadata: " << metadata_file << std::endl;
        }

        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Feature extraction time: " << std::fixed << std::setprecision(2) 
                  << extract_time << "s" << std::endl;
        std::cout << "All features saved to: " << output_dir << std::endl;
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              FEATURE EXTRACTION COMPLETED!                     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
