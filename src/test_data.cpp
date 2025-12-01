#include "data/cifar10_dataset.h"
#include "data/data_utils.h"
#include "utils/logger.h"
#include "config.h"
#include <iostream>
#include <cassert>
#include <cmath>

// Test helper function
bool float_equal(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

// Test 1: Tensor creation and basic operations
void test_tensor_basics() {
    LOG_INFO("Testing Tensor basics...");
    
    // Create tensor [2, 3, 4, 4] (batch=2, channels=3, height=4, width=4)
    std::vector<int> shape = {2, 3, 4, 4};
    Tensor tensor(shape);
    
    // Check shape
    assert(tensor.shape.size() == 4);
    assert(tensor.batch() == 2);
    assert(tensor.channels() == 3);
    assert(tensor.height() == 4);
    assert(tensor.width() == 4);
    assert(tensor.size() == 96);
    
    // Test element access
    tensor({0, 0, 0, 0}) = 1.5f;
    tensor({1, 2, 3, 3}) = 2.5f;
    
    assert(float_equal(tensor({0, 0, 0, 0}), 1.5f));
    assert(float_equal(tensor({1, 2, 3, 3}), 2.5f));
    
    LOG_INFO("Tensor basics test passed");
}

// Test 2: Normalize tensor
void test_normalize_tensor() {
    LOG_INFO("Testing normalize_tensor...");
    
    std::vector<int> shape = {1, 1, 2, 2};
    Tensor tensor(shape, false);
    
    // Fill with values [0, 255, 127.5, 63.75]
    tensor({0, 0, 0, 0}) = 0.0f;
    tensor({0, 0, 0, 1}) = 255.0f;
    tensor({0, 0, 1, 0}) = 127.5f;
    tensor({0, 0, 1, 1}) = 63.75f;
    
    DataUtils::normalize_tensor(tensor);
    
    // Check normalized values [0, 1, 0.5, 0.25]
    assert(float_equal(tensor({0, 0, 0, 0}), 0.0f));
    assert(float_equal(tensor({0, 0, 0, 1}), 1.0f));
    assert(float_equal(tensor({0, 0, 1, 0}), 0.5f));
    assert(float_equal(tensor({0, 0, 1, 1}), 0.25f));
    
    LOG_INFO("Normalize tensor test passed");
}

// Test 3: Standardize tensor
void test_standardize_tensor() {
    LOG_INFO("Testing standardize_tensor...");
    
    std::vector<int> shape = {1, 1, 2, 2};
    Tensor tensor(shape, false);
    
    // Fill with values [1, 2, 3, 4] - mean=2.5, std=1.118
    tensor({0, 0, 0, 0}) = 1.0f;
    tensor({0, 0, 0, 1}) = 2.0f;
    tensor({0, 0, 1, 0}) = 3.0f;
    tensor({0, 0, 1, 1}) = 4.0f;
    
    DataUtils::standardize_tensor(tensor);
    
    // After standardization, mean should be ~0 and std should be ~1
    float sum = 0.0f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            sum += tensor({0, 0, i, j});
        }
    }
    float mean = sum / 4.0f;
    
    assert(float_equal(mean, 0.0f, 1e-4f));
    LOG_INFO("Standardize tensor test passed");
}

// Test 4: Save and load tensor
void test_save_load_tensor() {
    LOG_INFO("Testing save/load tensor...");
    
    // Create a tensor with specific values
    std::vector<int> shape = {2, 2, 3, 3};
    Tensor original(shape, false);
    
    // Fill with sequential values
    for (int i = 0; i < original.size(); ++i) {
        original.data->data()[i] = static_cast<float>(i);
    }
    
    // Save to file
    std::string filepath = "test_tensor.bin";
    DataUtils::save_tensor(original, filepath);
    
    // Load from file
    Tensor loaded = DataUtils::load_tensor(filepath);
    
    // Verify shape
    assert(loaded.shape.size() == original.shape.size());
    for (size_t i = 0; i < original.shape.size(); ++i) {
        assert(loaded.shape[i] == original.shape[i]);
    }
    
    // Verify data
    for (size_t i = 0; i < original.size(); ++i) {
        assert(float_equal(loaded.data->data()[i], original.data->data()[i]));
    }
    
    LOG_INFO("Save/load tensor test passed");
}

// Test 5: CIFAR10Dataset initialization
void test_cifar10_dataset_init() {
    LOG_INFO("Testing CIFAR10Dataset initialization...");
    
    // Create dataset (without loading data)
    CIFAR10Dataset train_dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
    CIFAR10Dataset test_dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TEST);
    
    // Check sizes
    assert(train_dataset.size() == CIFAR_TRAIN_IMAGES);
    assert(test_dataset.size() == CIFAR_TEST_IMAGES);
    
    // Check image properties
    assert(train_dataset.get_image_size() == CIFAR_IMAGE_SIZE);
    assert(train_dataset.get_num_channels() == CIFAR_CHANNELS);
    
    LOG_INFO("CIFAR10Dataset initialization test passed");
}

// Test 6: CIFAR10Dataset shuffle
void test_cifar10_dataset_shuffle() {
    LOG_INFO("Testing CIFAR10Dataset shuffle...");
    
    CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TEST);
    
    // Shuffle multiple times
    dataset.shuffle();
    dataset.shuffle();
    
    // Reset
    dataset.reset();
    
    LOG_INFO("CIFAR10Dataset shuffle test passed");
}

// Test 7: CIFAR10Dataset load and batch (requires actual data files)
void test_cifar10_dataset_load() {
    LOG_INFO("Testing CIFAR10Dataset load and batch...");
    
    try {
        CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TEST);
        dataset.load_data();
        
        // Get a batch
        int batch_size = 16;
        Tensor batch = dataset.get_batch(batch_size);
        std::vector<int> labels = dataset.get_batch_labels(batch_size);
        
        // Check batch shape
        assert(batch.batch() == batch_size);
        assert(batch.channels() == CIFAR_CHANNELS);
        assert(batch.height() == CIFAR_IMAGE_SIZE);
        assert(batch.width() == CIFAR_IMAGE_SIZE);
        
        // Check labels
        assert(labels.size() == batch_size);
        for (int label : labels) {
            assert(label >= 0 && label < CIFAR_CLASSES);
        }
        
        // Check pixel values are normalized [0, 1]
        float* data = batch.data->data();
        for (int i = 0; i < 100; ++i) {  // Sample first 100 pixels
            assert(data[i] >= 0.0f && data[i] <= 1.0f);
        }
        
        LOG_INFO("CIFAR10Dataset load and batch test passed");
        
    } catch (const std::exception& e) {
        LOG_WARNING("Could not load CIFAR-10 data: %s", e.what());
        LOG_WARNING("Skipping data loading test (data files may not be available)");
    }
}

// Test 9: Edge cases
void test_edge_cases() {
    LOG_INFO("Testing edge cases...");
    
    // Small tensor
    Tensor small({1, 1, 1, 1});
    assert(small.size() == 1);
    small({0, 0, 0, 0}) = 42.0f;
    assert(float_equal(small({0, 0, 0, 0}), 42.0f));
    
    // Large tensor
    Tensor large({10, 3, 64, 64});
    assert(large.size() == 10 * 3 * 64 * 64);
    
    LOG_INFO("Edge cases test passed");
}

int main() {
    LOG_INIT();
    LOG_SET_LEVEL(LogLevel::INFO);
    
    LOG_INFO("=== Starting Data Functions Tests ===\n");
    
    try {
        // Run all tests
        test_tensor_basics();
        test_normalize_tensor();
        test_standardize_tensor();
        test_save_load_tensor();
        test_cifar10_dataset_init();
        test_cifar10_dataset_shuffle();
        test_cifar10_dataset_load();
        test_edge_cases();
        
        LOG_INFO("\n=== All Tests Passed! ===");
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Test failed with exception: %s", e.what());
        return 1;
    }
}
