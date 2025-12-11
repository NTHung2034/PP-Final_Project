#pragma once

#include "config.h"
#include <string>
#include <vector>
#include <random>
#include <cstdint>

/**
 * Simple CIFAR-10 Data Loader
 * 
 * - Loads binary files directly into contiguous float arrays
 * - Images stored as NCHW format [N, 3, 32, 32]
 * - Normalized to [0, 1] range
 * - Supports batch generation and shuffling
 * - Easy integration with GPU tensors (just copy raw pointer)
 */
class CIFAR10Loader {
private:
    // Load single binary file
    void load_batch_file(const std::string& filepath, float* images, int* labels, int start_idx);

    // Data directory
    std::string data_dir_;

    // Raw data storage (contiguous arrays)
    float* train_images_ = nullptr;   // [CIFAR_TRAIN_IMAGES * CIFAR_PIXELS] floats
    int* train_labels_ = nullptr;     // [CIFAR_TRAIN_IMAGES] ints
    float* test_images_ = nullptr;    // [CIFAR_TEST_IMAGES * CIFAR_PIXELS] floats
    int* test_labels_ = nullptr;      // [CIFAR_TEST_IMAGES] ints

    // Shuffled indices for training
    std::vector<int> shuffled_indices_;
    int current_idx_ = 0;

    // Temporary batch buffer (reused)
    float* batch_buffer_ = nullptr;
    int* label_buffer_ = nullptr;
    int batch_buffer_size_ = 0;

    // Random generator
    std::mt19937 rng_;
    
    // Load flags
    bool train_loaded_ = false;
    bool test_loaded_ = false;

public:
    // Constructor & Destructor
    explicit CIFAR10Loader(const std::string& data_dir);
    ~CIFAR10Loader();

    // Disable copy
    CIFAR10Loader(const CIFAR10Loader&) = delete;
    CIFAR10Loader& operator=(const CIFAR10Loader&) = delete;

    // Load data from binary files
    bool load_train_data();
    bool load_test_data();

    // Batch generation - returns pointer to batch data (caller should NOT free)
    // The returned pointer is valid until next get_batch() call or shuffle
    float* get_batch(int batch_size);
    int* get_batch_labels(int batch_size);
    
    // Get specific batch by index (for deterministic access)
    float* get_batch_at(int start_idx, int batch_size);
    int* get_labels_at(int start_idx, int batch_size);

    // Shuffling and reset
    void shuffle();
    void reset();

    // Direct data access (for GPU copy)
    float* train_images() { return train_images_; }
    int* train_labels() { return train_labels_; }
    float* test_images() { return test_images_; }
    int* test_labels() { return test_labels_; }
    
    const float* train_images() const { return train_images_; }
    const int* train_labels() const { return train_labels_; }
    const float* test_images() const { return test_images_; }
    const int* test_labels() const { return test_labels_; }

    // Info
    int train_size() const { return CIFAR_TRAIN_IMAGES; }
    int test_size() const { return CIFAR_TEST_IMAGES; }
    int current_index() const { return current_idx_; }
    bool has_more_batches(int batch_size) const { return current_idx_ + batch_size <= CIFAR_TRAIN_IMAGES; }

};
