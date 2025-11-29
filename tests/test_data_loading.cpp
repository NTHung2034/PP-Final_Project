/**
 * @file test_data_loading.cpp
 * @brief Test suite for CIFAR-10 data loading module
 * 
 * This test verifies:
 * - Dataset loading from binary files
 * - Correct normalization [0, 1]
 * - Batch generation
 * - Shuffling functionality
 * - Data integrity
 * 
 * Usage:
 *   ./test_data_loading
 * 
 * Expected output:
 *   All tests should pass with "✓" markers
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#include "data/cifar10_dataset.h"
#include "data/data_utils.h"
#include "utils/logger.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <cmath>

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

void test_pass(const std::string& name) {
    std::cout << "  ✓ " << name << std::endl;
    tests_passed++;
}

void test_fail(const std::string& name, const std::string& reason) {
    std::cout << "  ✗ " << name << " - " << reason << std::endl;
    tests_failed++;
}

/**
 * @brief Test 1: Dataset loading
 */
void test_dataset_loading() {
    std::cout << "\n=== Test 1: Dataset Loading ===" << std::endl;
    
    try {
        CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
        dataset.load_data();
        
        // Check size
        if (dataset.size() == CIFAR_TRAIN_IMAGES) {
            test_pass("Training dataset has correct size (50000)");
        } else {
            test_fail("Dataset size check", 
                      "Expected 50000, got " + std::to_string(dataset.size()));
        }
        
    } catch (const std::exception& e) {
        test_fail("Dataset loading", e.what());
    }
}

/**
 * @brief Test 2: Batch generation
 */
void test_batch_generation() {
    std::cout << "\n=== Test 2: Batch Generation ===" << std::endl;
    
    try {
        CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
        dataset.load_data();
        
        // Get a batch
        const int batch_size = 32;
        Tensor batch = dataset.get_batch(batch_size);
        
        // Check shape
        if (batch.batch() == batch_size && 
            batch.channels() == CIFAR_CHANNELS &&
            batch.height() == CIFAR_IMAGE_SIZE &&
            batch.width() == CIFAR_IMAGE_SIZE) {
            test_pass("Batch shape is correct [32, 3, 32, 32]");
        } else {
            test_fail("Batch shape check",
                      "Shape mismatch");
        }
        
        // Check total size
        size_t expected_size = batch_size * CIFAR_CHANNELS * CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE;
        if (batch.size() == expected_size) {
            test_pass("Batch total size correct (" + std::to_string(expected_size) + ")");
        } else {
            test_fail("Batch size check",
                      "Expected " + std::to_string(expected_size) + 
                      ", got " + std::to_string(batch.size()));
        }
        
    } catch (const std::exception& e) {
        test_fail("Batch generation", e.what());
    }
}

/**
 * @brief Test 3: Normalization
 */
void test_normalization() {
    std::cout << "\n=== Test 3: Normalization ===" << std::endl;
    
    try {
        CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
        dataset.load_data();
        
        Tensor batch = dataset.get_batch(100);
        const float* data = batch.data->data();
        size_t size = batch.size();
        
        // Find min and max values
        float min_val = *std::min_element(data, data + size);
        float max_val = *std::max_element(data, data + size);
        
        // Values should be in [0, 1] range
        if (min_val >= 0.0f && max_val <= 1.0f) {
            test_pass("Pixel values in [0, 1] range");
        } else {
            test_fail("Normalization check",
                      "Values outside [0,1]: min=" + std::to_string(min_val) +
                      ", max=" + std::to_string(max_val));
        }
        
        // Check that we have variety (not all zeros or ones)
        if (min_val < 0.1f && max_val > 0.9f) {
            test_pass("Pixel values have good variety");
        } else {
            test_fail("Value variety check",
                      "Limited value range");
        }
        
    } catch (const std::exception& e) {
        test_fail("Normalization test", e.what());
    }
}

/**
 * @brief Test 4: Labels
 */
void test_labels() {
    std::cout << "\n=== Test 4: Labels ===" << std::endl;
    
    try {
        CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
        dataset.load_data();
        
        // Get batch and labels
        Tensor batch = dataset.get_batch(100);
        std::vector<int> labels = dataset.get_batch_labels(100);
        
        // Check labels size
        if (labels.size() == 100) {
            test_pass("Labels size matches batch size");
        } else {
            test_fail("Labels size", 
                      "Expected 100, got " + std::to_string(labels.size()));
        }
        
        // Check labels are in valid range [0, 9]
        bool valid_range = true;
        for (int label : labels) {
            if (label < 0 || label > 9) {
                valid_range = false;
                break;
            }
        }
        
        if (valid_range) {
            test_pass("All labels in valid range [0, 9]");
        } else {
            test_fail("Labels range", "Found labels outside [0, 9]");
        }
        
    } catch (const std::exception& e) {
        test_fail("Labels test", e.what());
    }
}

/**
 * @brief Test 5: Shuffling
 */
void test_shuffling() {
    std::cout << "\n=== Test 5: Shuffling ===" << std::endl;
    
    try {
        CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
        dataset.load_data();
        
        // Get first batch before shuffling
        Tensor batch1 = dataset.get_batch(10);
        std::vector<float> data1(batch1.data->data(), 
                                  batch1.data->data() + batch1.size());
        
        // Reset and shuffle
        dataset.reset();
        dataset.shuffle();
        
        // Get first batch after shuffling
        Tensor batch2 = dataset.get_batch(10);
        std::vector<float> data2(batch2.data->data(),
                                  batch2.data->data() + batch2.size());
        
        // Data should be different after shuffling
        bool same = (data1 == data2);
        
        if (!same) {
            test_pass("Shuffling produces different batch order");
        } else {
            test_fail("Shuffling", "Batch order unchanged after shuffle");
        }
        
    } catch (const std::exception& e) {
        test_fail("Shuffling test", e.what());
    }
}

/**
 * @brief Test 6: Multiple batches
 */
void test_multiple_batches() {
    std::cout << "\n=== Test 6: Multiple Batches ===" << std::endl;
    
    try {
        CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
        dataset.load_data();
        
        // Get multiple batches
        const int batch_size = 100;
        const int num_batches = 10;
        
        for (int i = 0; i < num_batches; ++i) {
            Tensor batch = dataset.get_batch(batch_size);
            
            if (batch.batch() != batch_size) {
                test_fail("Multiple batches", 
                          "Batch " + std::to_string(i) + " has wrong size");
                return;
            }
        }
        
        test_pass("Successfully retrieved " + std::to_string(num_batches) + " batches");
        
    } catch (const std::exception& e) {
        test_fail("Multiple batches test", e.what());
    }
}

/**
 * @brief Test 7: Test dataset loading
 */
void test_test_dataset() {
    std::cout << "\n=== Test 7: Test Dataset ===" << std::endl;
    
    try {
        CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TEST);
        dataset.load_data();
        
        // Check size
        if (dataset.size() == CIFAR_TEST_IMAGES) {
            test_pass("Test dataset has correct size (10000)");
        } else {
            test_fail("Test dataset size",
                      "Expected 10000, got " + std::to_string(dataset.size()));
        }
        
        // Get a batch
        Tensor batch = dataset.get_batch(32);
        if (batch.batch() == 32) {
            test_pass("Test batch generation works");
        } else {
            test_fail("Test batch", "Wrong batch size");
        }
        
    } catch (const std::exception& e) {
        test_fail("Test dataset loading", e.what());
    }
}

int main() {
    LOG_INIT();
    
    std::cout << "========================================" << std::endl;
    std::cout << "CIFAR-10 Data Loading Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Run all tests
    test_dataset_loading();
    test_batch_generation();
    test_normalization();
    test_labels();
    test_shuffling();
    test_multiple_batches();
    test_test_dataset();
    
    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    
    if (tests_failed == 0) {
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ Some tests failed!" << std::endl;
        return 1;
    }
}
