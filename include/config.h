#pragma once

// Data Configuration
#define CIFAR_BIN_DIR           "../data/cifar-10-batches-bin"
#define TRAIN_BATCH_FILES       { "data_batch_1.bin", "data_batch_2.bin", \
                                 "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin" }
#define TEST_BATCH_FILE         "test_batch.bin"
#define MODEL_SAVE_DIR          "../models/saved_weights"

// Dataset Constants
constexpr int CIFAR_IMAGE_SIZE   = 32;
constexpr int CIFAR_CHANNELS     = 3;
constexpr int CIFAR_PIXELS       = CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE * CIFAR_CHANNELS;
constexpr int CIFAR_TRAIN_IMAGES = 50000;
constexpr int CIFAR_TEST_IMAGES  = 10000;
constexpr int CIFAR_CLASSES      = 10;

// Training Configuration
constexpr int BATCH_SIZE         = 32;
constexpr int EPOCHS             = 20;
constexpr float LEARNING_RATE    = 0.001f;

// Data Format (NCHW for optimal CUDA performance)
constexpr int DATA_FORMAT_NCHW   = 0;  // [batch, channels, height, width]
constexpr int DATA_FORMAT_NHWC   = 1;  // [batch, height, width, channels]
constexpr int DATA_FORMAT        = DATA_FORMAT_NCHW;

// Performance
constexpr bool ENABLE_SHUFFLING  = true;
constexpr int RANDOM_SEED        = 42;