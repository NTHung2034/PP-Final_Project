#include "data/cifar10_dataset.h"
#include "utils/logger.h"
#include "config.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>

CIFAR10Dataset::CIFAR10Dataset(const std::string& data_root, Mode mode) 
    : data_root_(data_root), mode_(mode), rng_(RANDOM_SEED) {
    
    // Set number of images based on mode
    num_images_ = (mode == Mode::TRAIN) ? CIFAR_TRAIN_IMAGES : CIFAR_TEST_IMAGES;
    
    // Pre-allocate memory
    std::vector<int> shape = {static_cast<int>(num_images_), 
                              CIFAR_CHANNELS, 
                              CIFAR_IMAGE_SIZE, 
                              CIFAR_IMAGE_SIZE};
    images_ = std::make_unique<Tensor>(shape);
    
    labels_.resize(num_images_);
    create_shuffled_indices();
}

CIFAR10Dataset::~CIFAR10Dataset() = default;

void CIFAR10Dataset::create_shuffled_indices() {
    shuffled_indices_.resize(num_images_);
    std::iota(shuffled_indices_.begin(), shuffled_indices_.end(), 0);
}

void CIFAR10Dataset::shuffle() {
    std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), rng_);
    current_index_ = 0;
    LOG_INFO("Dataset shuffled. New order generated.");
}

void CIFAR10Dataset::load_data() {
    LOG_INFO("Loading CIFAR-10 %s data from %s...", 
             mode_ == Mode::TRAIN ? "training" : "test", data_root_.c_str());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (mode_ == Mode::TRAIN) {
        // Load 5 training batches
        const std::vector<std::string> batch_files = TRAIN_BATCH_FILES;
        const int images_per_batch = CIFAR_TRAIN_IMAGES / batch_files.size();
        
        for (size_t i = 0; i < batch_files.size(); ++i) {
            std::string filepath = data_root_ + "/" + batch_files[i];
            load_batch(filepath, i * images_per_batch);
        }
    } else {
        // Load test batch
        std::string filepath = data_root_ + "/" + TEST_BATCH_FILE;
        load_batch(filepath, 0);
    }
    
    is_loaded_ = true;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time).count();
    
    LOG_INFO("Dataset loaded successfully in %.2f seconds", duration);
    LOG_INFO("Total images: %zu, Image shape: [%d, %d, %d]", 
             num_images_, CIFAR_CHANNELS, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE);
}

void CIFAR10Dataset::load_batch(const std::string& filename, int start_idx) {
    /**
     * CIFAR-10 Binary Format:
     * - Each image record: 1 byte label + 3072 bytes pixel data
     * - Pixel data is stored in PLANAR format: 1024 red, then 1024 green, then 1024 blue
     * - Each color channel is stored row-by-row (row-major)
     * - Pixel values are uint8 in range [0, 255]
     * 
     * Reference: https://www.cs.toronto.edu/~kriz/cifar.html
     * "The first byte is the label of the first image... The next 3072 bytes 
     *  are the values of the pixels of the image. The first 1024 bytes are 
     *  the red channel values, the next 1024 the green, and the final 1024 the blue."
     */
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    const int record_size = 1 + CIFAR_PIXELS; // 1 byte label + 3072 bytes image
    const int batch_size = 10000; // Each CIFAR-10 file contains exactly 10,000 images
    
    std::vector<uint8_t> buffer(record_size);
    
    for (int i = 0; i < batch_size; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), record_size);
        if (!file) break;
        
        // Extract label (first byte)
        labels_[start_idx + i] = buffer[0];
        
        // Extract and normalize image data
        // CIFAR-10 binary is already in NCHW planar format (channel-first)
        // Buffer layout: [label, R0..R1023, G0..G1023, B0..B1023]
        float* img_ptr = images_->data->data() + (start_idx + i) * CIFAR_PIXELS;
        
        for (int c = 0; c < CIFAR_CHANNELS; ++c) {
            for (int h = 0; h < CIFAR_IMAGE_SIZE; ++h) {
                for (int w = 0; w < CIFAR_IMAGE_SIZE; ++w) {
                    // Source: buffer offset = 1 (skip label) + channel*1024 + h*32 + w
                    int src_idx = 1 + c * (CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE) + h * CIFAR_IMAGE_SIZE + w;
                    // Destination: NCHW format = c*1024 + h*32 + w
                    int dst_idx = c * (CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE) + h * CIFAR_IMAGE_SIZE + w;
                    
                    // Normalize [0, 255] to [0, 1]
                    img_ptr[dst_idx] = static_cast<float>(buffer[src_idx]) / 255.0f;
                }
            }
        }
    }
    
    file.close();
}

void CIFAR10Dataset::normalize_image(float* image) {
    // In-place normalization [0, 255] -> [0, 1]
    // Note: This is already done during load_batch, kept for backward compatibility
    for (int i = 0; i < CIFAR_PIXELS; ++i) {
        image[i] = image[i] / 255.0f;
    }
}

Tensor CIFAR10Dataset::get_batch(int batch_size) {
    if (!is_loaded_) {
        throw std::runtime_error("Dataset not loaded. Call load_data() first.");
    }
    
    // Adjust batch size for end of dataset
    int actual_batch_size = std::min(batch_size, 
                                     static_cast<int>(num_images_ - current_index_));
    
    // Create batch tensor [N, C, H, W]
    std::vector<int> batch_shape = {actual_batch_size, 
                                    CIFAR_CHANNELS, 
                                    CIFAR_IMAGE_SIZE, 
                                    CIFAR_IMAGE_SIZE};
    Tensor batch(batch_shape);
    
    // Copy images using shuffled indices
    float* batch_ptr = batch.data->data();
    const float* full_ptr = images_->data->data();
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < actual_batch_size; ++i) {
        size_t img_idx = shuffled_indices_[current_index_ + i];
        size_t src_offset = img_idx * CIFAR_PIXELS;
        size_t dst_offset = i * CIFAR_PIXELS;
        
        std::copy(full_ptr + src_offset, 
                  full_ptr + src_offset + CIFAR_PIXELS,
                  batch_ptr + dst_offset);
    }
    
    current_index_ += actual_batch_size;
    return batch;
}

std::vector<int> CIFAR10Dataset::get_batch_labels(int batch_size) {
    if (!is_loaded_) {
        throw std::runtime_error("Dataset not loaded. Call load_data() first.");
    }
    
    // Get labels for the PREVIOUS batch (after current_index_ was advanced)
    int actual_batch_size = std::min(batch_size, 
                                     static_cast<int>(current_index_));
    
    std::vector<int> batch_labels(actual_batch_size);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < actual_batch_size; ++i) {
        size_t img_idx = shuffled_indices_[current_index_ - actual_batch_size + i];
        batch_labels[i] = labels_[img_idx];
    }
    
    return batch_labels;
}

void CIFAR10Dataset::reset() {
    current_index_ = 0;
}