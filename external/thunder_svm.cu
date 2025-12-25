#include "svm.h"
#include "./thundersvm/svm_interface_api.h"  // ThunderSVM C API
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <sstream>

SVMClassifier::SVMClassifier() {
    svm_model_ = nullptr;
    svm_params_ = nullptr;
}

SVMClassifier::~SVMClassifier() {
    // ThunderSVM cleanup handled by library
    if (svm_model_) {
        // Model cleanup
        svm_model_ = nullptr;
    }
}

void SVMClassifier::train(const std::vector<float>& features,
                          const std::vector<uint8_t>& labels,
                          int num_samples,
                          int feature_dim) {
    
    std::cout << "Training SVM classifier with ThunderSVM (GPU)..." << std::endl;
    std::cout << "Samples: " << num_samples << ", Features: " << feature_dim << std::endl;
    
    // Prepare data in ThunderSVM format
    std::cout << "Preparing training data..." << std::endl;
    auto prep_start = std::chrono::high_resolution_clock::now();
    
    // Write data to temporary LIBSVM format file
    std::string temp_train_file = "/tmp/thundersvm_train.txt";
    std::ofstream train_file(temp_train_file);
    
    if (!train_file.is_open()) {
        std::cerr << "Error: Cannot create temporary training file!" << std::endl;
        return;
    }
    
    for (int i = 0; i < num_samples; ++i) {
        train_file << static_cast<int>(labels[i]);
        
        for (int j = 0; j < feature_dim; ++j) {
            float value = features[i * feature_dim + j];
            if (value != 0.0f) { 
                train_file << " " << (j + 1) << ":" << value;
            }
        }
        train_file << "\n";
        
        if ((i + 1) % 10000 == 0) {
            std::cout << "  Written " << (i + 1) << "/" << num_samples 
                      << " samples\r" << std::flush;
        }
    }
    
    train_file.close();
    
    auto prep_end = std::chrono::high_resolution_clock::now();
    float prep_time = std::chrono::duration<float>(prep_end - prep_start).count();
    std::cout << "\nData preparation completed in " << prep_time << " seconds" << std::endl;
    
    // Configure ThunderSVM parameters
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting ThunderSVM GPU training..." << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    auto train_start = std::chrono::high_resolution_clock::now();
    
    // Build command line arguments for ThunderSVM
    std::vector<std::string> args_str;
    args_str.push_back("thundersvm-train");
    args_str.push_back("-s");
    args_str.push_back("0");              // C-SVC
    args_str.push_back("-t");
    args_str.push_back("2");              // RBF kernel 
    args_str.push_back("-c");
    args_str.push_back("10");             // C parameter
    args_str.push_back("-g");
    args_str.push_back(std::to_string(1.0f / feature_dim));  // gamma
    args_str.push_back("-e");
    args_str.push_back("0.001");          // epsilon
    args_str.push_back("-o");
    args_str.push_back("1");              // GPU device ID
    args_str.push_back(temp_train_file);
    args_str.push_back("/tmp/thundersvm_model.txt");
    std::vector<char*> args;
    for (auto& s : args_str) {
        args.push_back(const_cast<char*>(s.c_str()));
    }
    // Train using ThunderSVM C API
    thundersvm_train(args.size(), args.data());
    
    auto train_end = std::chrono::high_resolution_clock::now();
    float train_time = std::chrono::duration<float>(train_end - train_start).count();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "ThunderSVM training completed in " << train_time 
              << " seconds (" << (train_time / 60.0f) << " minutes)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    svm_model_ = reinterpret_cast<void*>(1); 
}

void SVMClassifier::predict(const std::vector<float>& features,
                            int num_samples,
                            int feature_dim,
                            std::vector<uint8_t>& predictions) {
    if (!svm_model_) {
        std::cerr << "Error: SVM model not trained or loaded!" << std::endl;
        return;
    }
    
    predictions.resize(num_samples);
    
    std::cout << "Making predictions with ThunderSVM..." << std::endl;
    
    // Write test data to file
    std::string temp_test_file = "/tmp/thundersvm_test.txt";
    std::ofstream test_file(temp_test_file);
    
    for (int i = 0; i < num_samples; ++i) {
        test_file << "0";  // Dummy label
        
        for (int j = 0; j < feature_dim; ++j) {
            float value = features[i * feature_dim + j];
            if (value != 0.0f) {
                test_file << " " << (j + 1) << ":" << value;
            }
        }
        test_file << "\n";
    }
    
    test_file.close();
    
    // Build prediction arguments
    std::vector<std::string> args_str;
    args_str.push_back("thundersvm-predict");
    args_str.push_back(temp_test_file);
    args_str.push_back("/tmp/thundersvm_model.txt");
    args_str.push_back("/tmp/thundersvm_predictions.txt");
    
    std::vector<char*> args;
    for (auto& s : args_str) {
        args.push_back(const_cast<char*>(s.c_str()));
    }
    
    // Predict
    thundersvm_predict(args.size(), args.data());
    
    // Read predictions
    std::ifstream pred_file("/tmp/thundersvm_predictions.txt");
    for (int i = 0; i < num_samples; ++i) {
        double pred;
        pred_file >> pred;
        predictions[i] = static_cast<uint8_t>(pred);
    }
    pred_file.close();
    
    std::cout << "Predictions complete!" << std::endl;
}

float SVMClassifier::evaluate(const std::vector<uint8_t>& predictions,
                              const std::vector<uint8_t>& ground_truth) {
    if (predictions.size() != ground_truth.size()) {
        std::cerr << "Error: predictions and ground_truth size mismatch!" << std::endl;
        return 0.0f;
    }
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == ground_truth[i]) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / predictions.size();
}

void SVMClassifier::compute_confusion_matrix(const std::vector<uint8_t>& predictions,
                                             const std::vector<uint8_t>& ground_truth,
                                             int num_classes,
                                             std::vector<std::vector<int>>& confusion_matrix) {
    confusion_matrix.assign(num_classes, std::vector<int>(num_classes, 0));
    
    for (size_t i = 0; i < predictions.size(); i++) {
        int true_label = ground_truth[i];
        int pred_label = predictions[i];
        if (true_label < num_classes && pred_label < num_classes) {
            confusion_matrix[true_label][pred_label]++;
        }
    }
}

bool SVMClassifier::save_model(const std::string& filepath) {
    if (!svm_model_) {
        std::cerr << "Error: No model to save!" << std::endl;
        return false;
    }
    
    // Copy the ThunderSVM model file
    std::ifstream src("/tmp/thundersvm_model.txt", std::ios::binary);
    std::ofstream dst(filepath, std::ios::binary);
    
    if (!src.is_open() || !dst.is_open()) {
        std::cerr << "Error saving model!" << std::endl;
        return false;
    }
    
    dst << src.rdbuf();
    src.close();
    dst.close();
    
    std::cout << "Model saved successfully to: " << filepath << std::endl;
    return true;
}

bool SVMClassifier::load_model(const std::string& filepath) {
    // Copy model to temporary location
    std::ifstream src(filepath, std::ios::binary);
    std::ofstream dst("/tmp/thundersvm_model.txt", std::ios::binary);
    
    if (!src.is_open() || !dst.is_open()) {
        std::cerr << "Error loading model from: " << filepath << std::endl;
        return false;
    }
    
    dst << src.rdbuf();
    src.close();
    dst.close();
    
    svm_model_ = reinterpret_cast<void*>(1);  // Mark as loaded
    std::cout << "Model loaded successfully from: " << filepath << std::endl;
    return true;
}