#include "data/cifar10_loader.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <cstring>

CIFAR10Loader::CIFAR10Loader(const std::string& data_dir) : data_dir_(data_dir), rng_(RANDOM_SEED) {
    
    // Initialize shuffled indices
    shuffled_indices_.resize(CIFAR_TRAIN_IMAGES);
    std::iota(shuffled_indices_.begin(), shuffled_indices_.end(), 0);
}

CIFAR10Loader::~CIFAR10Loader() {
    delete[] train_images_;
    delete[] train_labels_;
    delete[] test_images_;
    delete[] test_labels_;
    delete[] batch_buffer_;
    delete[] label_buffer_;
}

bool CIFAR10Loader::load_train_data() {
    if (train_loaded_) return true;
    
    std::cout << "Loading CIFAR-10 training data..." << std::endl;
    
    // Allocate memory
    train_images_ = new float[CIFAR_TRAIN_IMAGES * CIFAR_PIXELS];
    train_labels_ = new int[CIFAR_TRAIN_IMAGES];
    
    // Load 5 batch files
    const char* batch_files[] = {
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"
    };
    
    constexpr int IMAGES_PER_BATCH = CIFAR_TRAIN_IMAGES / 5;  // 10000 images per file
    for (int i = 0; i < 5; i++) {
        std::string filepath = data_dir_ + "/" + batch_files[i];
        load_batch_file(filepath, train_images_, train_labels_, i * IMAGES_PER_BATCH);
        std::cout << "  Loaded " << batch_files[i] << std::endl;
    }
    
    train_loaded_ = true;
    std::cout << "Training data loaded: " << CIFAR_TRAIN_IMAGES << " images" << std::endl;
    return true;
}

bool CIFAR10Loader::load_test_data() {
    if (test_loaded_) return true;
    
    std::cout << "Loading CIFAR-10 test data..." << std::endl;
    
    // Allocate memory
    test_images_ = new float[CIFAR_TEST_IMAGES * CIFAR_PIXELS];
    test_labels_ = new int[CIFAR_TEST_IMAGES];
    
    // Load test batch
    std::string filepath = data_dir_ + "/" TEST_BATCH_FILE;
    load_batch_file(filepath, test_images_, test_labels_, 0);
    
    test_loaded_ = true;
    std::cout << "Test data loaded: " << CIFAR_TEST_IMAGES << " images" << std::endl;
    return true;
}

void CIFAR10Loader::load_batch_file(const std::string& filepath, float* images, int* labels, int start_idx) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open: " + filepath);
    }
    
    // Each record: 1 byte label + CIFAR_PIXELS bytes image (RGB planar format)
    constexpr int RECORD_SIZE = 1 + CIFAR_PIXELS;
    constexpr int IMAGES_PER_BATCH = CIFAR_TRAIN_IMAGES / 5;  // 10000 images per file
    std::vector<uint8_t> buffer(RECORD_SIZE);
    
    for (int i = 0; i < IMAGES_PER_BATCH; i++) {
        file.read(reinterpret_cast<char*>(buffer.data()), RECORD_SIZE);
        if (!file) break;
        
        // Extract label (first byte)
        labels[start_idx + i] = buffer[0];
        
        // Extract and normalize image - CHW format
        float* img_ptr = images + (start_idx + i) * CIFAR_PIXELS;   
        
        for (int j = 0; j < CIFAR_PIXELS; j++) {
            img_ptr[j] = static_cast<float>(buffer[1 + j]) / 255.0f;
        }
    }
    
    file.close();
}

void CIFAR10Loader::shuffle() {
    std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), rng_);
    current_idx_ = 0;
}

void CIFAR10Loader::reset() {
    current_idx_ = 0;
}

float* CIFAR10Loader::get_batch(int batch_size) {
    if (!train_loaded_) {
        throw std::runtime_error("Training data not loaded. Call load_train_data() first.");
    }
    
    // Calculate actual batch size (handle end of dataset)
    int actual_size = std::min(batch_size, CIFAR_TRAIN_IMAGES - current_idx_);
    if (actual_size <= 0) return nullptr;
    
    // Reallocate buffer if needed
    if (batch_size > batch_buffer_size_) {
        delete[] batch_buffer_;
        batch_buffer_ = new float[batch_size * CIFAR_PIXELS];
        batch_buffer_size_ = batch_size;
    }
    
    // Copy images using shuffled indices
    for (int i = 0; i < actual_size; i++) {
        int img_idx = shuffled_indices_[current_idx_ + i];
        std::memcpy(batch_buffer_ + i * CIFAR_PIXELS,
               train_images_ + img_idx * CIFAR_PIXELS,
               CIFAR_PIXELS * sizeof(float));
    }
    
    current_idx_ += actual_size;
    return batch_buffer_;
}

int* CIFAR10Loader::get_batch_labels(int batch_size) {
    if (!train_loaded_) {
        throw std::runtime_error("Training data not loaded. Call load_train_data() first.");
    }
    
    int actual_size = std::min(batch_size, CIFAR_TRAIN_IMAGES - current_idx_);
    if (actual_size <= 0) return nullptr;
    
    // Reallocate buffer if needed
    if (batch_size > batch_buffer_size_) {
        delete[] label_buffer_;
        label_buffer_ = new int[batch_size];
    }
    
    // Copy labels using shuffled indices (note: uses current_idx_ BEFORE get_batch increments it)
    // So call get_batch_labels BEFORE get_batch, or use get_labels_at
    for (int i = 0; i < actual_size; i++) {
        int img_idx = shuffled_indices_[current_idx_ + i];
        label_buffer_[i] = train_labels_[img_idx];
    }
    
    return label_buffer_;
}

// EXTENDED FUNCTIONS
// Get specific batch by index (for deterministic access)
float* CIFAR10Loader::get_batch_at(int start_idx, int batch_size) {
    if (!train_loaded_) {
        throw std::runtime_error("Training data not loaded. Call load_train_data() first.");
    }
    
    if (start_idx + batch_size > CIFAR_TRAIN_IMAGES) {
        batch_size = CIFAR_TRAIN_IMAGES - start_idx;
    }
    if (batch_size <= 0) return nullptr;
    
    // Reallocate buffer if needed
    if (batch_size > batch_buffer_size_) {
        delete[] batch_buffer_;
        batch_buffer_ = new float[batch_size * CIFAR_PIXELS];
        batch_buffer_size_ = batch_size;
    }
    
    // Copy images using shuffled indices
    for (int i = 0; i < batch_size; i++) {
        int img_idx = shuffled_indices_[start_idx + i];
        std::memcpy(batch_buffer_ + i * CIFAR_PIXELS,
               train_images_ + img_idx * CIFAR_PIXELS,
               CIFAR_PIXELS * sizeof(float));
    }
    
    return batch_buffer_;
}

// Get specific batch labels by index (for deterministic access)
int* CIFAR10Loader::get_labels_at(int start_idx, int batch_size) {
    if (!train_loaded_) {
        throw std::runtime_error("Training data not loaded. Call load_train_data() first.");
    }
    
    if (start_idx + batch_size > CIFAR_TRAIN_IMAGES) {
        batch_size = CIFAR_TRAIN_IMAGES - start_idx;
    }
    if (batch_size <= 0) return nullptr;
    
    // Reallocate buffer if needed  
    if (batch_size > batch_buffer_size_) {
        delete[] label_buffer_;
        label_buffer_ = new int[batch_size];
    }
    
    for (int i = 0; i < batch_size; i++) {
        int img_idx = shuffled_indices_[start_idx + i];
        label_buffer_[i] = train_labels_[img_idx];
    }
    
    return label_buffer_;
}
