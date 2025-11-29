#include "data/cifar10_dataset.h"
#include "config.h"
#include "utils/logger.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>

CIFAR10Dataset::CIFAR10Dataset(const std::string &data_root, Mode mode)
    : mode_(mode), data_root_(data_root), rng_(RANDOM_SEED)
{

    // Set number of images based on mode
    num_images_ = (mode == Mode::TRAIN) ? CIFAR_TRAIN_IMAGES : CIFAR_TEST_IMAGES;

    // Pre-allocate memory
    std::vector<int> shape = {static_cast<int>(num_images_),
                              CIFAR_CHANNELS,
                              CIFAR_IMAGE_SIZE,
                              CIFAR_IMAGE_SIZE};
    images_.reset(new Tensor(shape));

    labels_.resize(num_images_);
    create_shuffled_indices();
}

CIFAR10Dataset::~CIFAR10Dataset() = default;

void CIFAR10Dataset::create_shuffled_indices()
{
    shuffled_indices_.resize(num_images_);
    std::iota(shuffled_indices_.begin(), shuffled_indices_.end(), 0);
}

void CIFAR10Dataset::shuffle()
{
    std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), rng_);
    current_index_ = 0;
    LOG_INFO("Dataset shuffled. New order generated.");
}

void CIFAR10Dataset::load_data()
{
    LOG_INFO("Loading CIFAR-10 %s data from %s...",
             mode_ == Mode::TRAIN ? "training" : "test", data_root_.c_str());

    auto start_time = std::chrono::high_resolution_clock::now();

    if (mode_ == Mode::TRAIN)
    {
        // Load 5 training batches
        const std::vector<std::string> batch_files = TRAIN_BATCH_FILES;
        const int images_per_batch = CIFAR_TRAIN_IMAGES / batch_files.size();

        for (int i = 0; i < batch_files.size(); ++i)
        {
            std::string filepath = data_root_ + "/" + batch_files[i];
            load_batch(filepath, i * images_per_batch);
        }
    }
    else
    {
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

void CIFAR10Dataset::load_batch(const std::string &filename, int start_idx)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    const int record_size = 1 + CIFAR_PIXELS;                      // 1 byte label + 3072 bytes image
    const int batch_size = (mode_ == Mode::TRAIN) ? 10000 : 10000; // Each file has 10K images

    std::vector<uint8_t> buffer(record_size);

    for (int i = 0; i < batch_size; ++i)
    {
        file.read(reinterpret_cast<char *>(buffer.data()), record_size);
        if (!file)
            break;

        // Extract label (first byte)
        labels_[start_idx + i] = buffer[0];

        // Extract and normalize image
        float *img_ptr = images_->data->data() + (start_idx + i) * CIFAR_PIXELS;

        // Convert from NHWC (file format) to NCHW (our format)
        for (int c = 0; c < CIFAR_CHANNELS; ++c)
        {
            for (int h = 0; h < CIFAR_IMAGE_SIZE; ++h)
            {
                for (int w = 0; w < CIFAR_IMAGE_SIZE; ++w)
                {
                    int src_idx = 1 + (h * CIFAR_IMAGE_SIZE + w) * CIFAR_CHANNELS + c;
                    int dst_idx = (c * CIFAR_IMAGE_SIZE + h) * CIFAR_IMAGE_SIZE + w;
                    img_ptr[dst_idx] = static_cast<float>(buffer[src_idx]) / 255.0f;
                }
            }
        }
    }

    file.close();
}

void CIFAR10Dataset::normalize_image(float *image)
{
    // In-place normalization [0, 255] -> [0, 1]
    for (int i = 0; i < CIFAR_PIXELS; ++i)
    {
        image[i] = image[i] / 255.0f;
    }
}

Tensor CIFAR10Dataset::get_batch(int batch_size)
{
    if (!is_loaded_)
    {
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
    float *batch_ptr = batch.data->data();
    const float *full_ptr = images_->data->data();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < actual_batch_size; ++i)
    {
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

std::vector<int> CIFAR10Dataset::get_batch_labels(int batch_size)
{
    int actual_batch_size = std::min(batch_size,
                                     static_cast<int>(num_images_ - current_index_));

    std::vector<int> batch_labels(actual_batch_size);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < actual_batch_size; ++i)
    {
        size_t img_idx = shuffled_indices_[current_index_ + i - actual_batch_size];
        batch_labels[i] = labels_[img_idx];
    }

    return batch_labels;
}

void CIFAR10Dataset::reset()
{
    current_index_ = 0;
}