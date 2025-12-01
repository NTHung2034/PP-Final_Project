#pragma once
#include "config.h"
#include "data_types.h"
#include <string>
#include <vector>
#include <random>
#include <memory>
#include "config.h"

class CIFAR10Dataset
{
public:
    // Modes
    enum class Mode
    {
        TRAIN,
        TEST
    };

    // Constructors
    explicit CIFAR10Dataset(const std::string &data_root, Mode mode);
    ~CIFAR10Dataset();

    // Disable copy, enable move
    CIFAR10Dataset(const CIFAR10Dataset &) = delete;
    CIFAR10Dataset &operator=(const CIFAR10Dataset &) = delete;
    CIFAR10Dataset(CIFAR10Dataset &&) noexcept = default;
    CIFAR10Dataset &operator=(CIFAR10Dataset &&) noexcept = default;

    // Core Methods
    void load_data();
    Tensor get_batch(int batch_size);
    std::vector<int> get_batch_labels(int batch_size);

    // Shuffle dataset (call between epochs)
    void shuffle();

    // Reset pointer to beginning
    void reset();

    // Accessors
    size_t size() const { return num_images_; }
    int get_image_size() const { return CIFAR_IMAGE_SIZE; }
    int get_num_channels() const { return CIFAR_CHANNELS; }

private:
    // Internal helpers
    void load_batch(const std::string &filename, int start_idx);
    void normalize_image(float *image);
    void create_shuffled_indices();

    // Configuration (order matters for initialization)
    Mode mode_;
    std::string data_root_;

    // Data storage
    std::unique_ptr<Tensor> images_;       // All images [N, C, H, W]
    std::vector<int> labels_;              // Corresponding labels
    std::vector<size_t> shuffled_indices_; // For shuffling
    size_t current_index_ = 0;
    size_t num_images_ = 0;
    bool is_loaded_ = false;

    // Random generator
    std::mt19937 rng_;
};