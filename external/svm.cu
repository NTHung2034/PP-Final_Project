#include "svm.h"
#include "libsvm/svm.h"
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <chrono>

static void SVMPrint(const char* s) {
    std::cout << s;
    std::cout.flush();
}

SVMClassifier::SVMClassifier() {
    svm_model_ = nullptr;
    // Enable LIBSVM's progress output 
    svm_set_print_string_function(&SVMPrint);
    // Initialize SVM parameters (optimized for speed)
    svm_parameter* params = new svm_parameter();
    params->svm_type = C_SVC;
    params->kernel_type = RBF;
    params->degree = 3;
    params->gamma = 0; 
    params->coef0 = 0;
    params->nu = 0.5;
    params->cache_size = 10000;  
    params->C = 10.0;
    params->eps = 1e-3;  
    params->p = 0.1;
    params->shrinking = 1;  
    params->probability = 0;
    params->nr_weight = 0;
    params->weight_label = nullptr;
    params->weight = nullptr;
    
    svm_params_ = params;
}

SVMClassifier::~SVMClassifier() {
    if (svm_model_) {
        svm_free_and_destroy_model(&svm_model_);
    }
    if (svm_params_) {
        svm_destroy_param(svm_params_);
        delete svm_params_;
    }
}

void SVMClassifier::train(const std::vector<float>& features,
                          const std::vector<uint8_t>& labels,
                          int num_samples,
                          int feature_dim) {

    std::cout << "Training SVM classifier..." << std::endl;
    std::cout << "Samples: " << num_samples << ", Features: " << feature_dim << std::endl;

    // Set gamma if not specified
    if (svm_params_->gamma == 0) {
        svm_params_->gamma = 1.0 / feature_dim;
    }

    // Prepare LIBSVM problem
    std::cout << "Preparing training data for LIBSVM..." << std::endl;
    auto prep_start = std::chrono::high_resolution_clock::now();

    svm_problem prob;
    prob.l = num_samples;

    // Allocate y
    prob.y = new double[num_samples];

    // Allocate x pointer list
    prob.x = new svm_node*[num_samples];

    // Allocate all svm_node in one contiguous block
    svm_node* nodes = new svm_node[num_samples * (feature_dim + 1)];

    // Fill
    for (int i = 0; i < num_samples; ++i) {
        prob.y[i] = labels[i];

        svm_node* row = &nodes[i * (feature_dim + 1)];
        prob.x[i] = row;

        const float* sample = &features[i * feature_dim];

        for (int j = 0; j < feature_dim; ++j) {
            row[j].index = j + 1;
            row[j].value = sample[j];
        }

        row[feature_dim].index = -1;  
    }

    auto prep_end = std::chrono::high_resolution_clock::now();
    float prep_time = std::chrono::duration<float>(prep_end - prep_start).count();
    std::cout << "\nData preparation completed in " << prep_time << " seconds" << std::endl;

    // Check parameters
    const char* error_msg = svm_check_parameter(&prob, svm_params_);
    if (error_msg) {
        std::cerr << "SVM parameter error: " << error_msg << std::endl;
        delete[] nodes;
        delete[] prob.x;
        delete[] prob.y;
        return;
    }

    // Train model
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting SVM training..." << std::endl;
    std::cout << "Configuration: C=" << svm_params_->C 
              << ", gamma=" << svm_params_->gamma 
              << ", cache_size=" << svm_params_->cache_size << "MB" << std::endl;
    std::cout << "========================================\n" << std::endl;

    auto train_start = std::chrono::high_resolution_clock::now();
    svm_model* model = svm_train(&prob, svm_params_);
    auto train_end = std::chrono::high_resolution_clock::now();

    float train_time = std::chrono::duration<float>(train_end - train_start).count();

    if (svm_model_) {
        svm_free_and_destroy_model(&svm_model_);
    }
    svm_model_ = model;

    std::cout << "\n========================================" << std::endl;
    std::cout << "SVM training completed in " << train_time 
              << " seconds (" << (train_time / 60.0f) << " minutes)" << std::endl;
    std::cout << "========================================" << std::endl;

    // Free memory
    delete[] nodes;
    delete[] prob.x;
    delete[] prob.y;

    std::cout << "SVM training completed" << std::endl;
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
    
    std::cout << "Making predictions..." << std::endl;
    
    // Allocate temporary node array for prediction
    svm_node* nodes = new svm_node[feature_dim + 1];
    
    for (int i = 0; i < num_samples; i++) {
        const float* sample = &features[i * feature_dim];
        
        // Fill node array
        for (int j = 0; j < feature_dim; j++) {
            nodes[j].index = j + 1;
            nodes[j].value = sample[j];
        }
        nodes[feature_dim].index = -1;
        
        // Predict
        double pred = svm_predict(svm_model_, nodes);
        predictions[i] = static_cast<uint8_t>(pred);
        
        if ((i + 1) % 2000 == 0) {
            std::cout << "  Predicted " << (i + 1) << "/" << num_samples << " samples" << std::endl;
        }
    }
    
    delete[] nodes;
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
    
    int result = svm_save_model(filepath.c_str(), svm_model_);
    
    if (result == 0) {
        std::cout << "Model saved successfully to: " << filepath << std::endl;
        return true;
    } else {
        std::cerr << "Error saving model to: " << filepath << std::endl;
        return false;
    }
}

bool SVMClassifier::load_model(const std::string& filepath) {
    if (svm_model_) {
        svm_free_and_destroy_model(&svm_model_);
    }
    
    svm_model_ = svm_load_model(filepath.c_str());
    
    if (svm_model_) {
        std::cout << "Model loaded successfully from: " << filepath << std::endl;
        return true;
    } else {
        std::cerr << "Error loading model from: " << filepath << std::endl;
        return false;
    }
}