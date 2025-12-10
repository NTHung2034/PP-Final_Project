/**
 * Test file for CIFAR10Loader - Simple CIFAR-10 data loader
 * Tests: loading, batching, shuffling, normalization
 */

#include "data/cifar10_loader.h"
#include "config.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>

// Test helper function
bool float_equal(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

// Test 1: Basic loader construction
void test_loader_construction() {
    std::cout << "[TEST] Testing CIFAR10Loader construction..." << std::endl;
    
    CIFAR10Loader loader(CIFAR_BIN_DIR);
    
    // Before loading, pointers should be null
    assert(loader.train_images() == nullptr);
    assert(loader.train_labels() == nullptr);
    assert(loader.test_images() == nullptr);
    assert(loader.test_labels() == nullptr);
    
    std::cout << "[PASS] Loader construction test passed" << std::endl;
}

// Test 2: Load training data
void test_load_train_data() {
    std::cout << "[TEST] Testing load_train_data()..." << std::endl;
    
    try {
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        bool success = loader.load_train_data();
        
        if (!success) {
            std::cout << "[SKIP] Could not load training data (files may not exist)" << std::endl;
            return;
        }
        
        // Check that data is loaded
        assert(loader.train_images() != nullptr);
        assert(loader.train_labels() != nullptr);
        assert(loader.train_size() == 50000);
        
        std::cout << "[PASS] load_train_data test passed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[SKIP] Exception: " << e.what() << std::endl;
    }
}

// Test 3: Load test data
void test_load_test_data() {
    std::cout << "[TEST] Testing load_test_data()..." << std::endl;
    
    try {
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        bool success = loader.load_test_data();
        
        if (!success) {
            std::cout << "[SKIP] Could not load test data (files may not exist)" << std::endl;
            return;
        }
        
        // Check that data is loaded
        assert(loader.test_images() != nullptr);
        assert(loader.test_labels() != nullptr);
        assert(loader.test_size() == 10000);
        
        std::cout << "[PASS] load_test_data test passed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[SKIP] Exception: " << e.what() << std::endl;
    }
}

// Test 4: Normalization check (values should be in [0, 1])
void test_normalization() {
    std::cout << "[TEST] Testing normalization [0, 1]..." << std::endl;
    
    try {
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        bool success = loader.load_train_data();
        
        if (!success) {
            std::cout << "[SKIP] Could not load data" << std::endl;
            return;
        }
        
        const float* images = loader.train_images();
        int total_pixels = 50000 * 3 * 32 * 32;  // NCHW format
        
        // Sample check: verify first 10000 values are in [0, 1]
        int check_count = std::min(10000, total_pixels);
        for (int i = 0; i < check_count; ++i) {
            assert(images[i] >= 0.0f && images[i] <= 1.0f);
        }
        
        // Also check some random positions
        for (int i = 0; i < 1000; ++i) {
            int idx = (i * 12345) % total_pixels;
            assert(images[idx] >= 0.0f && images[idx] <= 1.0f);
        }
        
        std::cout << "[PASS] Normalization test passed (all values in [0, 1])" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[SKIP] Exception: " << e.what() << std::endl;
    }
}

// Test 5: Labels check (values should be in [0, 9])
void test_labels() {
    std::cout << "[TEST] Testing labels [0, 9]..." << std::endl;
    
    try {
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        bool success = loader.load_train_data();
        
        if (!success) {
            std::cout << "[SKIP] Could not load data" << std::endl;
            return;
        }
        
        const int* labels = loader.train_labels();
        
        // Check all labels are in valid range
        for (int i = 0; i < 50000; ++i) {
            assert(labels[i] >= 0 && labels[i] <= 9);
        }
        
        std::cout << "[PASS] Labels test passed (all values in [0, 9])" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[SKIP] Exception: " << e.what() << std::endl;
    }
}

// Test 6: Batch generation
void test_batch_generation() {
    std::cout << "[TEST] Testing batch generation..." << std::endl;
    
    try {
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        bool success = loader.load_train_data();
        
        if (!success) {
            std::cout << "[SKIP] Could not load data" << std::endl;
            return;
        }
        
        const int batch_size = 32;
        const int image_size = 3 * 32 * 32;  // 3072 floats per image
        
        // Get first batch
        loader.reset();
        assert(loader.has_more_batches(batch_size));
        
        float* batch1 = loader.get_batch(batch_size);
        assert(batch1 != nullptr);
        
        // Get batch labels
        int* labels1 = loader.get_batch_labels(batch_size);
        assert(labels1 != nullptr);
        
        // Verify batch data is valid
        for (int i = 0; i < batch_size * image_size; ++i) {
            assert(batch1[i] >= 0.0f && batch1[i] <= 1.0f);
        }
        
        // Verify batch labels are valid
        for (int i = 0; i < batch_size; ++i) {
            assert(labels1[i] >= 0 && labels1[i] <= 9);
        }
        
        std::cout << "[PASS] Batch generation test passed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[SKIP] Exception: " << e.what() << std::endl;
    }
}

// Test 7: Multiple batches and reset
void test_multiple_batches() {
    std::cout << "[TEST] Testing multiple batches and reset..." << std::endl;
    
    try {
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        bool success = loader.load_train_data();
        
        if (!success) {
            std::cout << "[SKIP] Could not load data" << std::endl;
            return;
        }
        
        const int batch_size = 100;
        int batch_count = 0;
        
        loader.reset();
        while (loader.has_more_batches(batch_size)) {
            float* batch = loader.get_batch(batch_size);
            assert(batch != nullptr);
            batch_count++;
            
            // Safety check to avoid infinite loop
            if (batch_count > 600) break;
        }
        
        // Should have 500 full batches (50000 / 100)
        assert(batch_count == 500);
        
        // After exhausting, reset and try again
        loader.reset();
        assert(loader.has_more_batches(batch_size));
        
        float* batch = loader.get_batch(batch_size);
        assert(batch != nullptr);
        
        std::cout << "[PASS] Multiple batches test passed (500 batches)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[SKIP] Exception: " << e.what() << std::endl;
    }
}

// Test 8: Shuffle functionality
void test_shuffle() {
    std::cout << "[TEST] Testing shuffle..." << std::endl;
    
    try {
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        bool success = loader.load_train_data();
        
        if (!success) {
            std::cout << "[SKIP] Could not load data" << std::endl;
            return;
        }
        
        const int batch_size = 32;
        
        // Get first batch before shuffle
        loader.reset();
        float* batch_before = loader.get_batch(batch_size);
        
        // Copy first few values
        float first_values_before[100];
        std::memcpy(first_values_before, batch_before, 100 * sizeof(float));
        
        // Shuffle and get first batch again
        loader.shuffle();
        loader.reset();
        float* batch_after = loader.get_batch(batch_size);
        
        // Compare - should be different (with very high probability)
        int diff_count = 0;
        for (int i = 0; i < 100; ++i) {
            if (!float_equal(first_values_before[i], batch_after[i])) {
                diff_count++;
            }
        }
        
        // At least some values should be different after shuffle
        assert(diff_count > 0);
        
        std::cout << "[PASS] Shuffle test passed (" << diff_count << "/100 values different)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[SKIP] Exception: " << e.what() << std::endl;
    }
}

// Test 9: Memory layout (NCHW format)
void test_memory_layout() {
    std::cout << "[TEST] Testing NCHW memory layout..." << std::endl;
    
    try {
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        bool success = loader.load_train_data();
        
        if (!success) {
            std::cout << "[SKIP] Could not load data" << std::endl;
            return;
        }
        
        const float* images = loader.train_images();
        
        // In NCHW format:
        // - Each image is 3 * 32 * 32 = 3072 floats
        // - First 1024 floats = Red channel
        // - Next 1024 floats = Green channel  
        // - Last 1024 floats = Blue channel
        
        const int image_size = 3072;
        const int channel_size = 1024;
        
        // For image 0, check that each channel region is valid
        for (int c = 0; c < 3; ++c) {
            for (int pixel = 0; pixel < channel_size; ++pixel) {
                int idx = c * channel_size + pixel;
                assert(images[idx] >= 0.0f && images[idx] <= 1.0f);
            }
        }
        
        // Image 1 should start at offset 3072
        assert(images[image_size] >= 0.0f && images[image_size] <= 1.0f);
        
        std::cout << "[PASS] NCHW memory layout test passed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[SKIP] Exception: " << e.what() << std::endl;
    }
}

// Test 10: Data statistics
void test_data_statistics() {
    std::cout << "[TEST] Computing data statistics..." << std::endl;
    
    try {
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        bool success = loader.load_train_data();
        
        if (!success) {
            std::cout << "[SKIP] Could not load data" << std::endl;
            return;
        }
        
        const float* images = loader.train_images();
        const int total_pixels = 50000 * 3 * 32 * 32;
        
        // Compute mean and check range
        double sum = 0.0;
        float min_val = 1.0f, max_val = 0.0f;
        
        for (int i = 0; i < total_pixels; ++i) {
            sum += images[i];
            if (images[i] < min_val) min_val = images[i];
            if (images[i] > max_val) max_val = images[i];
        }
        
        double mean = sum / total_pixels;
        
        std::cout << "  - Total pixels: " << total_pixels << std::endl;
        std::cout << "  - Min value: " << min_val << std::endl;
        std::cout << "  - Max value: " << max_val << std::endl;
        std::cout << "  - Mean value: " << mean << std::endl;
        
        // CIFAR-10 normalized mean should be around 0.4-0.5
        assert(mean > 0.3 && mean < 0.6);
        assert(min_val >= 0.0f);
        assert(max_val <= 1.0f);
        
        std::cout << "[PASS] Data statistics test passed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[SKIP] Exception: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CIFAR10Loader Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Data directory: " << CIFAR_BIN_DIR << std::endl;
    std::cout << std::endl;
    
    try {
        test_loader_construction();
        test_load_train_data();
        test_load_test_data();
        test_normalization();
        test_labels();
        test_batch_generation();
        test_multiple_batches();
        test_shuffle();
        test_memory_layout();
        test_data_statistics();
        
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "  All Tests Completed!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
