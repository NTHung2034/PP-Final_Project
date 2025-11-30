# Phase 1: CPU Baseline & Data Pipeline Implementation Guide

**Duration:** November 28 - December 1, 2025 (4 days)  
**Objective:** Establish working CPU implementation and data infrastructure  
**Prerequisite:** Development environment set up (see README.md)

---

## 🎯 Current Progress (November 30, 2025 - Day 3)

### ✅ Completed Tasks

**Task 1.1: Data Loading** ✅ COMPLETE

- ✅ CIFAR-10 binary format parser implemented
- ✅ NHWC→NCHW conversion working
- ✅ Batch generation (configurable size)
- ✅ Dataset shuffling between epochs
- ✅ Normalization to [0,1] range
- ✅ **Test Result:** 50,000 images loaded in 0.30 seconds

**Task 1.2: CPU Layer Implementation** ✅ COMPLETE

- ✅ **Conv2D Layer:**
  - ✅ Forward pass with padding and stride
  - ✅ Backward pass (gradients for input, weights, bias)
  - ✅ Xavier weight initialization
  - ✅ Weight management (set/get/update)
  - ✅ SGD optimizer integration
- ✅ **ReLU Layer:**
  - ✅ Forward pass (element-wise max(0, x))
  - ✅ Backward pass (gradient masking)
- ✅ **MaxPool Layer:**
  - ✅ Forward pass with index tracking
  - ✅ Backward pass (gradient routing to max positions)
- ✅ **Upsample Layer:**
  - ✅ Forward pass (nearest-neighbor interpolation)
  - ✅ Backward pass (gradient accumulation)

**Windows Compatibility** ✅ COMPLETE

- ✅ Aligned memory allocation (`_aligned_malloc`)
- ✅ Path resolution (project-root relative)
- ✅ Build system configured (CMake + MinGW)
- ✅ All compilation errors resolved
- ✅ Clean build: zero errors, zero warnings

### 🔄 Current Task: Autoencoder Architecture (Day 3)

**Task 1.3: Autoencoder Integration** - IN PROGRESS

You need to create the autoencoder class that combines all layers:

**Required Files:**

```
include/models/          # Create this directory
  └── autoencoder_cpu.h  # TODO: Create

src/models/              # Create this directory
  └── autoencoder_cpu.cpp # TODO: Create
```

**Architecture to Implement:**

- **Encoder:** Conv(3→256) + ReLU + MaxPool + Conv(256→128) + ReLU + MaxPool
- **Decoder:** Conv(128→128) + ReLU + Up + Conv(128→256) + ReLU + Up + Conv(256→3)
- **Loss:** MSE between input and reconstructed output

### 📋 Next Steps

**Immediate Actions (Today - November 30):**

1. Create `include/models/` and `src/models/` directories
2. Implement AutoencoderCPU class
3. Chain encoder layers (2 conv blocks with pooling)
4. Chain decoder layers (3 conv blocks with upsampling)
5. Implement full forward pass (input → encoder → decoder → output)
6. Implement full backward pass (MSE loss → gradients → weight updates)
7. Update CMakeLists.txt to build autoencoder

**Verification Test:**

```powershell
# Build autoencoder
cd build
cmake --build . --config Release

# Run shape verification
.\bin\Release\train_autoencoder.exe
# Expected: Input [32,32,3] → Latent [8,8,128] → Output [32,32,3]
```

### 📊 Progress Checklist

- ✅ **Day 1-2:** Data pipeline (COMPLETE)

  - ✅ Task 1.1.1: CIFAR-10 format understood
  - ✅ Task 1.1.2: Binary reader implemented
  - ✅ Task 1.1.3: Normalization working
  - ✅ Task 1.1.4: Data loading tested

- ✅ **Day 2-3:** Layer implementation (COMPLETE)

  - ✅ Task 1.2.1: Conv2D layer (forward + backward + optimizer)
  - ✅ Task 1.2.2: ReLU activation
  - ✅ Task 1.2.3: MaxPooling
  - ✅ Task 1.2.4: Upsampling

- 🔄 **Day 3:** Autoencoder architecture (IN PROGRESS)

  - [ ] Task 1.3.1: Autoencoder class
  - [ ] Task 1.3.2: Forward pass
  - [ ] Task 1.3.3: Backward pass

- ⏳ **Day 3-4:** Training loop (PENDING)
  - [ ] Task 1.4.1: Training function
  - [ ] Task 1.4.2: Loss tracking
  - [ ] Task 1.4.3: Weight save/load

---

## Table of Contents

1. [Overview](#1-overview)
2. [Step-by-Step Implementation](#2-step-by-step-implementation)
3. [Testing and Verification](#3-testing-and-verification)
4. [Common Pitfalls](#4-common-pitfalls)
5. [Optimization Tips](#5-optimization-tips)
6. [Google Colab Notes](#6-google-colab-notes)

---

## 1. Overview

### 1.1 Goals

By the end of Phase 1, you will have:

- ✅ CIFAR-10 dataset loaded and preprocessed
- ✅ All neural network layers working on CPU
- ✅ Complete autoencoder architecture (encoder + decoder)
- ✅ Training loop with SGD optimizer
- ✅ Baseline performance metrics
- ✅ Weight save/load functionality

### 1.2 Architecture Reminder

```
INPUT (32×32×3)
    ↓
ENCODER:
    Conv2D(256, 3×3, pad=1) + ReLU  →  (32×32×256)
    MaxPool(2×2)                     →  (16×16×256)
    Conv2D(128, 3×3, pad=1) + ReLU  →  (16×16×128)
    MaxPool(2×2)                     →  (8×8×128)
    ↓
LATENT (8×8×128) = 8192 features
    ↓
DECODER:
    Conv2D(128, 3×3, pad=1) + ReLU  →  (8×8×128)
    UpSample(2×2)                    →  (16×16×128)
    Conv2D(256, 3×3, pad=1) + ReLU  →  (16×16×256)
    UpSample(2×2)                    →  (32×32×256)
    Conv2D(3, 3×3, pad=1)            →  (32×32×3)
    ↓
OUTPUT (32×32×3)
```

### 1.3 Expected Performance

| Metric                    | Target Value           |
| ------------------------- | ---------------------- |
| Training time/epoch       | 15-20 minutes          |
| Final reconstruction loss | <0.08 (after 5 epochs) |
| Memory usage              | ~4-6 GB RAM            |
| Batch size                | 32 images              |

---

## 2. Step-by-Step Implementation

### **Task 1.1: Data Loading (Day 1 - Morning)** ✅ COMPLETE

**Status:** All steps completed and verified. Dataset loading works successfully.

**Current Implementation Files:**

- `include/data/cifar10_dataset.h` - Dataset interface
- `src/data/cifar10_dataset.cpp` - Implementation with binary parsing
- `include/data/data_types.h` - Tensor class (Windows-compatible)
- `include/config.h` - Configuration constants

**Verification:**

```powershell
.\build\bin\train_autoencoder.exe
# ✅ Output: Dataset loaded successfully in 0.33 seconds
# ✅ Output: Total images: 50000, Image shape: [3, 32, 32]
```

#### Step 1.1.1: Understand CIFAR-10 Binary Format ✅

**File Structure:**

- Each batch file: 10,000 records
- Each record: 1 byte (label) + 3,072 bytes (image)
- Image format: 1,024 red + 1,024 green + 1,024 blue (row-major)
- Pixel values: uint8 [0, 255]

**Reference:** https://www.cs.toronto.edu/~kriz/cifar.html

#### Step 1.1.2: Implement Binary Reader

**File:** `src/data/cifar10_dataset.cpp`

```cpp
void CIFAR10Dataset::load_batch(const std::string& filename, int start_idx) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    const int record_size = 1 + CIFAR_PIXELS; // 1 label + 3072 pixels
    std::vector<uint8_t> buffer(record_size);

    for (int i = 0; i < 10000; ++i) {
        // Read record
        file.read(reinterpret_cast<char*>(buffer.data()), record_size);
        if (!file) break;

        // Extract label
        int label = static_cast<int>(buffer[0]);
        labels_[start_idx + i] = label;

        // Extract and convert image to NCHW format
        float* img_ptr = images_->data->data() + (start_idx + i) * CIFAR_PIXELS;

        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 32; ++h) {
                for (int w = 0; w < 32; ++w) {
                    int src_idx = 1 + c * 1024 + h * 32 + w;
                    int dst_idx = c * 1024 + h * 32 + w;
                    img_ptr[dst_idx] = static_cast<float>(buffer[src_idx]) / 255.0f;
                }
            }
        }
    }
}
```

**Key Points:**

- Convert uint8 [0,255] to float [0,1] immediately
- Store in NCHW format for optimal GPU performance later
- Use `start_idx` to load multiple batches into one contiguous array

#### Step 1.1.3: Implement Normalization

**File:** `src/data/data_utils.cpp`

```cpp
void normalize_cifar10(float* data, size_t num_images) {
    // CIFAR-10 is already normalized to [0,1] during loading
    // Optional: mean-std normalization
    // Mean: [0.4914, 0.4822, 0.4465]
    // Std:  [0.2470, 0.2435, 0.2616]

    const float mean[3] = {0.4914f, 0.4822f, 0.4465f};
    const float std[3]  = {0.2470f, 0.2435f, 0.2616f};

    #pragma omp parallel for
    for (size_t i = 0; i < num_images; ++i) {
        float* img = data + i * CIFAR_PIXELS;
        for (int c = 0; c < 3; ++c) {
            for (int pixel = 0; pixel < 1024; ++pixel) {
                int idx = c * 1024 + pixel;
                img[idx] = (img[idx] - mean[c]) / std[c];
            }
        }
    }
}
```

**Note:** For this project, simple [0,1] normalization is sufficient. Mean-std normalization is optional.

#### Step 1.1.4: Test Data Loading

**Create test file:** `tests/test_data_loading.cpp`

```cpp
#include "data/cifar10_dataset.h"
#include <iostream>

int main() {
    CIFAR10Dataset dataset("../data/cifar-10-batches-bin",
                           CIFAR10Dataset::Mode::TRAIN);
    dataset.load_data();

    std::cout << "Loaded " << dataset.size() << " images\n";

    // Test batch retrieval
    auto batch = dataset.get_batch(32);
    auto labels = dataset.get_batch_labels(32);

    // Verify shape
    assert(batch.batch() == 32);
    assert(batch.channels() == 3);
    assert(batch.height() == 32);
    assert(batch.width() == 32);

    // Verify normalization
    float* data = batch.data->data();
    float min_val = *std::min_element(data, data + batch.size());
    float max_val = *std::max_element(data, data + batch.size());

    std::cout << "Pixel range: [" << min_val << ", " << max_val << "]\n";
    assert(min_val >= 0.0f && max_val <= 1.0f);

    std::cout << "✓ Data loading test passed!\n";
    return 0;
}
```

**Run:**

**Linux/Ubuntu/macOS:**

```bash
mkdir build && cd build
cmake ..
make test_data_loading
./bin/test_data_loading
```

**Windows 11 PowerShell:**

```powershell
New-Item -ItemType Directory -Force -Path build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release --target test_data_loading
.\bin\Release\test_data_loading.exe
```

**Expected output:**

```
Loaded 50000 images
Pixel range: [0.0, 1.0]
✓ Data loading test passed!
```

---

### **Task 1.2: CPU Layer Implementation (Day 2-3)** 🔄 CURRENT TASK

**Status:** Not started - this is your immediate next step.

**Before You Start:**

1. **Create directories:**

```powershell
New-Item -ItemType Directory -Force -Path include/layers
New-Item -ItemType Directory -Force -Path src/layers
```

2. **Update CMakeLists.txt** to include layer files (add after you create them):

```cmake
# Add to your CMakeLists.txt
set(LAYER_SOURCES
    src/layers/conv2d_cpu.cpp
    src/layers/relu_cpu.cpp
    src/layers/maxpool_cpu.cpp
    src/layers/upsample_cpu.cpp
)
add_executable(train_autoencoder ${SOURCES} ${LAYER_SOURCES})
```

3. **Testing Strategy:** Create a test file for each layer as you implement it

#### Step 1.2.1: Implement Convolution Layer 🎯 START HERE

**File:** `include/layers/conv2d_cpu.h`

```cpp
#pragma once
#include "data/data_types.h"
#include <vector>

class Conv2DCPU {
public:
    Conv2DCPU(int in_channels, int out_channels, int kernel_size,
              int stride = 1, int padding = 0);

    // Forward pass
    Tensor forward(const Tensor& input);

    // Backward pass
    Tensor backward(const Tensor& grad_output);

    // Weight accessors
    void set_weights(const std::vector<float>& weights,
                     const std::vector<float>& bias);
    void get_gradients(std::vector<float>& grad_w, std::vector<float>& grad_b);
    void update_weights(float learning_rate);

private:
    int in_c_, out_c_, k_size_, stride_, pad_;
    std::vector<float> weights_;  // [out_c, in_c, k_size, k_size]
    std::vector<float> bias_;     // [out_c]
    std::vector<float> grad_w_, grad_b_;
    Tensor cached_input_;         // For backward pass
};
```

**Implementation:** `src/layers/conv2d_cpu.cpp`

```cpp
Tensor Conv2DCPU::forward(const Tensor& input) {
    cached_input_ = input; // Cache for backward pass

    int batch = input.batch();
    int in_h = input.height();
    int in_w = input.width();

    // Calculate output dimensions
    int out_h = (in_h + 2 * pad_ - k_size_) / stride_ + 1;
    int out_w = (in_w + 2 * pad_ - k_size_) / stride_ + 1;

    Tensor output({batch, out_c_, out_h, out_w});
    float* out_data = output.data->data();
    const float* in_data = input.data->data();

    // Parallelize over batch and output channels
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < out_c_; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = bias_[oc];

                    // Convolution operation
                    for (int ic = 0; ic < in_c_; ++ic) {
                        for (int kh = 0; kh < k_size_; ++kh) {
                            for (int kw = 0; kw < k_size_; ++kw) {
                                int ih = oh * stride_ - pad_ + kh;
                                int iw = ow * stride_ - pad_ + kw;

                                // Handle padding
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = ((n * in_c_ + ic) * in_h + ih) * in_w + iw;
                                    int w_idx = ((oc * in_c_ + ic) * k_size_ + kh) * k_size_ + kw;
                                    sum += in_data[in_idx] * weights_[w_idx];
                                }
                            }
                        }
                    }

                    int out_idx = ((n * out_c_ + oc) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = sum;
                }
            }
        }
    }

    return output;
}
```

**Key Implementation Details:**

- NCHW data layout: `[batch, channels, height, width]`
- Weight indexing: `[out_channels, in_channels, kernel_h, kernel_w]`
- Padding: Zero-padding (implicit by bounds checking)
- OpenMP: Parallelize outer loops for multi-core performance

**Constructor with Xavier Initialization:**

```cpp
Conv2DCPU::Conv2DCPU(int in_channels, int out_channels, int kernel_size, 
                     int stride, int padding)
    : in_c_(in_channels), out_c_(out_channels), k_size_(kernel_size), 
      stride_(stride), pad_(padding) 
{
    // Initialize weights and biases with Xavier/He initialization
    int weight_size = out_c_ * in_c_ * k_size_ * k_size_;
    weights_.resize(weight_size);
    bias_.resize(out_c_);
    grad_w_.resize(weight_size);
    grad_b_.resize(out_c_);

    // Xavier initialization: scale = sqrt(2.0 / (in_c * k_size * k_size))
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / (in_c_ * k_size_ * k_size_));
    std::normal_distribution<float> dist(0.0f, scale);

    for (int i = 0; i < weight_size; ++i) {
        weights_[i] = dist(gen);
    }

    // Initialize biases to zero
    std::fill(bias_.begin(), bias_.end(), 0.0f);
}
```

**Backward Pass Implementation:**

```cpp
Tensor Conv2DCPU::backward(const Tensor& grad_output) {
    // Initialize gradient tensors
    Tensor grad_input(cached_input_.shape);
    float* grad_in_data = grad_input.data->data();
    const float* grad_out_data = grad_output.data->data();
    const float* in_data = cached_input_.data->data();

    // Zero initialize gradients
    std::memset(grad_in_data, 0, grad_input.size() * sizeof(float));
    std::memset(grad_w_.data(), 0, grad_w_.size() * sizeof(float));
    std::memset(grad_b_.data(), 0, grad_b_.size() * sizeof(float));

    int batch = cached_input_.batch();
    int in_h = cached_input_.height();
    int in_w = cached_input_.width();
    int out_h = grad_output.height();
    int out_w = grad_output.width();

    // Compute gradients
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < out_c_; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx = ((n * out_c_ + oc) * out_h + oh) * out_w + ow;
                    float grad_out = grad_out_data[out_idx];

                    // Gradient w.r.t. bias
                    #pragma omp atomic
                    grad_b_[oc] += grad_out;

                    // Gradient w.r.t. weights and input
                    for (int ic = 0; ic < in_c_; ++ic) {
                        for (int kh = 0; kh < k_size_; ++kh) {
                            for (int kw = 0; kw < k_size_; ++kw) {
                                int ih = oh * stride_ - pad_ + kh;
                                int iw = ow * stride_ - pad_ + kw;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = ((n * in_c_ + ic) * in_h + ih) * in_w + iw;
                                    int w_idx = ((oc * in_c_ + ic) * k_size_ + kh) * k_size_ + kw;

                                    // Gradient w.r.t. weights: dL/dW = dL/dY * X
                                    #pragma omp atomic
                                    grad_w_[w_idx] += grad_out * in_data[in_idx];

                                    // Gradient w.r.t. input: dL/dX = dL/dY * W
                                    #pragma omp atomic
                                    grad_in_data[in_idx] += grad_out * weights_[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}
```

**Weight Management Functions:**

```cpp
void Conv2DCPU::set_weight(const std::vector<float>& weights, 
                           const std::vector<float>& bias) {
    if (weights.size() != weights_.size()) {
        throw std::runtime_error(\"Weight size mismatch\");
    }
    if (bias.size() != bias_.size()) {
        throw std::runtime_error(\"Bias size mismatch\");
    }
    weights_ = weights;
    bias_ = bias;
}

void Conv2DCPU::get_gradients(std::vector<float>& grad_w, 
                              std::vector<float>& grad_b) {
    grad_w = grad_w_;
    grad_b = grad_b_;
}

void Conv2DCPU::update_weights(float learning_rate) {
    // SGD update: weight -= learning_rate * gradient
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate * grad_w_[i];
    }
    
    for (size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] -= learning_rate * grad_b_[i];
    }
}
```

**Key Points:**
- **Forward:** Computes Y = W * X + b (convolution + bias)
- **Backward:** Computes three gradients:
  - dL/dW = dL/dY ⊗ X (gradient w.r.t. weights)
  - dL/db = sum(dL/dY) (gradient w.r.t. bias)
  - dL/dX = dL/dY ⊗ W (gradient w.r.t. input, for backprop)
- **Atomic operations** prevent race conditions in parallel regions
- **Xavier initialization** helps prevent vanishing/exploding gradients

#### Step 1.2.2: Implement ReLU Activation

**File:** `include/layers/relu_cpu.h`

```cpp
#pragma once
#include "data/data_types.h"

class ReLUCPU {
public:
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

private:
    Tensor cached_input_;
};
```

**Implementation:**

```cpp
Tensor ReLUCPU::forward(const Tensor& input) {
    cached_input_ = input;
    Tensor output = input; // Copy

    float* data = output.data->data();
    size_t size = output.size();

    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }

    return output;
}

Tensor ReLUCPU::backward(const Tensor& grad_output) {
    Tensor grad_input = grad_output; // Copy

    const float* input_data = cached_input_.data->data();
    float* grad_data = grad_input.data->data();
    size_t size = grad_input.size();

    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        grad_data[i] = (input_data[i] > 0.0f) ? grad_data[i] : 0.0f;
    }

    return grad_input;
}
```

#### Step 1.2.3: Implement MaxPooling

**File:** `include/layers/maxpool_cpu.h`

```cpp
Tensor MaxPoolCPU::forward(const Tensor& input) {
    cached_input_ = input;

    int batch = input.batch();
    int channels = input.channels();
    int in_h = input.height();
    int in_w = input.width();

    int out_h = in_h / pool_size_;
    int out_w = in_w / pool_size_;

    Tensor output({batch, channels, out_h, out_w});
    max_indices_.resize(output.size()); // Store indices for backward pass

    const float* in_data = input.data->data();
    float* out_data = output.data->data();

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = 0;

                    // Find max in pool_size × pool_size window
                    for (int ph = 0; ph < pool_size_; ++ph) {
                        for (int pw = 0; pw < pool_size_; ++pw) {
                            int ih = oh * pool_size_ + ph;
                            int iw = ow * pool_size_ + pw;
                            int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;

                            if (in_data[in_idx] > max_val) {
                                max_val = in_data[in_idx];
                                max_idx = in_idx;
                            }
                        }
                    }

                    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = max_val;
                    max_indices_[out_idx] = max_idx;
                }
            }
        }
    }

    return output;
}
```

**Backward Pass Implementation:**

```cpp
Tensor MaxPoolCPU::backward(const Tensor& grad_output) {
    // Create gradient tensor with same shape as input
    int batch = cached_input_.batch();
    int channels = cached_input_.channels();
    int in_h = cached_input_.height();
    int in_w = cached_input_.width();

    Tensor grad_input({batch, channels, in_h, in_w});
    float* grad_in_data = grad_input.data->data();
    const float* grad_out_data = grad_output.data->data();

    // Zero initialize gradient input
    std::memset(grad_in_data, 0, grad_input.size() * sizeof(float));

    int out_h = in_h / pool_size_;
    int out_w = in_w / pool_size_;

    // Distribute gradients only to max positions
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                    int max_idx = max_indices_[out_idx];
                    grad_in_data[max_idx] += grad_out_data[out_idx];
                }
            }
        }
    }

    return grad_input;
}
```

**Key Points:**

- Gradient flows only to the position that had the maximum value during forward pass
- Uses stored `max_indices_` from forward pass
- All other positions get zero gradient

#### Step 1.2.4: Implement Upsampling (Nearest Neighbor)

**File:** `include/layers/upsample_cpu.h`

```cpp
Tensor UpsampleCPU::forward(const Tensor& input) {
    cached_input_ = input;

    int batch = input.batch();
    int channels = input.channels();
    int in_h = input.height();
    int in_w = input.width();

    int out_h = in_h * scale_;
    int out_w = in_w * scale_;

    Tensor output({batch, channels, out_h, out_w});

    const float* in_data = input.data->data();
    float* out_data = output.data->data();

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    // Map output to input coordinates
                    int ih = oh / scale_;
                    int iw = ow / scale_;

                    int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;

                    out_data[out_idx] = in_data[in_idx];
                }
            }
        }
    }

    return output;
}
```

---

### **Task 1.3: Autoencoder Architecture (Day 2 Afternoon)**

#### Step 1.3.1: Create Autoencoder Class

**File:** `include/models/autoencoder_cpu.h`

```cpp
#pragma once
#include "layers/conv2d_cpu.h"
#include "layers/relu_cpu.h"
#include "layers/maxpool_cpu.h"
#include "layers/upsample_cpu.h"
#include <memory>

class AutoencoderCPU {
public:
    AutoencoderCPU();

    // Forward pass (training mode: returns reconstruction)
    Tensor forward(const Tensor& input);

    // Extract features (inference mode: returns latent representation)
    Tensor extract_features(const Tensor& input);

    // Backward pass + weight update
    void backward(const Tensor& target, float learning_rate);

    // Compute MSE loss
    float compute_loss(const Tensor& output, const Tensor& target);

    // Save/load weights
    void save_weights(const std::string& filename);
    void load_weights(const std::string& filename);

private:
    // Encoder layers
    std::unique_ptr<Conv2DCPU> conv1_;    // 3 → 256
    std::unique_ptr<ReLUCPU> relu1_;
    std::unique_ptr<MaxPoolCPU> pool1_;

    std::unique_ptr<Conv2DCPU> conv2_;    // 256 → 128
    std::unique_ptr<ReLUCPU> relu2_;
    std::unique_ptr<MaxPoolCPU> pool2_;

    // Decoder layers
    std::unique_ptr<Conv2DCPU> conv3_;    // 128 → 128
    std::unique_ptr<ReLUCPU> relu3_;
    std::unique_ptr<UpsampleCPU> up1_;

    std::unique_ptr<Conv2DCPU> conv4_;    // 128 → 256
    std::unique_ptr<ReLUCPU> relu4_;
    std::unique_ptr<UpsampleCPU> up2_;

    std::unique_ptr<Conv2DCPU> conv5_;    // 256 → 3

    // Cached tensors for backward pass
    Tensor latent_;
    Tensor output_;
};
```

#### Step 1.3.2: Implement Forward Pass

```cpp
Tensor AutoencoderCPU::forward(const Tensor& input) {
    // ENCODER
    auto x = conv1_->forward(input);      // (32,32,3) → (32,32,256)
    x = relu1_->forward(x);
    x = pool1_->forward(x);               // → (16,16,256)

    x = conv2_->forward(x);               // → (16,16,128)
    x = relu2_->forward(x);
    latent_ = pool2_->forward(x);         // → (8,8,128) = 8192 features

    // DECODER
    x = conv3_->forward(latent_);         // (8,8,128) → (8,8,128)
    x = relu3_->forward(x);
    x = up1_->forward(x);                 // → (16,16,128)

    x = conv4_->forward(x);               // → (16,16,256)
    x = relu4_->forward(x);
    x = up2_->forward(x);                 // → (32,32,256)

    output_ = conv5_->forward(x);         // → (32,32,3)

    return output_;
}
```

#### Step 1.3.3: Implement Backward Pass

```cpp
void AutoencoderCPU::backward(const Tensor& target, float learning_rate) {
    // Compute gradient of loss w.r.t. output
    Tensor grad = output_; // Copy
    float* grad_data = grad.data->data();
    const float* target_data = target.data->data();
    size_t size = grad.size();

    // MSE gradient: 2 * (output - target) / N
    float scale = 2.0f / size;
    for (size_t i = 0; i < size; ++i) {
        grad_data[i] = scale * (grad_data[i] - target_data[i]);
    }

    // Backpropagate through decoder
    grad = conv5_->backward(grad);
    grad = up2_->backward(grad);
    grad = relu4_->backward(grad);
    grad = conv4_->backward(grad);
    grad = up1_->backward(grad);
    grad = relu3_->backward(grad);
    grad = conv3_->backward(grad);

    // Backpropagate through encoder
    grad = pool2_->backward(grad);
    grad = relu2_->backward(grad);
    grad = conv2_->backward(grad);
    grad = pool1_->backward(grad);
    grad = relu1_->backward(grad);
    grad = conv1_->backward(grad);

    // Update weights
    conv1_->update_weights(learning_rate);
    conv2_->update_weights(learning_rate);
    conv3_->update_weights(learning_rate);
    conv4_->update_weights(learning_rate);
    conv5_->update_weights(learning_rate);
}
```

---

### **Task 1.4: Training Loop (Day 3)**

#### Step 1.4.1: Implement Training Function

**File:** `src/train_cpu.cpp`

```cpp
#include "models/autoencoder_cpu.h"
#include "data/cifar10_dataset.h"
#include "utils/logger.h"
#include "config.h"
#include <chrono>

int main() {
    LOG_INIT();
    LOG_INFO("Starting CPU Training");

    // Load dataset
    CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
    dataset.load_data();
    LOG_INFO("Loaded %zu training images", dataset.size());

    // Create model
    AutoencoderCPU model;

    // Training configuration
    const int batch_size = BATCH_SIZE;
    const int epochs = EPOCHS;
    const float lr = LEARNING_RATE;
    const int batches_per_epoch = dataset.size() / batch_size;

    LOG_INFO("Training config: epochs=%d, batch_size=%d, lr=%.4f",
             epochs, batch_size, lr);

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        dataset.shuffle();
        dataset.reset();

        double epoch_loss = 0.0;

        for (int batch = 0; batch < batches_per_epoch; ++batch) {
            // Get batch
            auto images = dataset.get_batch(batch_size);

            // Forward pass
            auto output = model.forward(images);

            // Compute loss
            float loss = model.compute_loss(output, images);
            epoch_loss += loss;

            // Backward pass + update
            model.backward(images, lr);

            // Log progress
            if (batch % 100 == 0) {
                LOG_INFO("Epoch %d/%d, Batch %d/%d, Loss: %.6f",
                         epoch + 1, epochs, batch, batches_per_epoch, loss);
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
            epoch_end - epoch_start).count();

        epoch_loss /= batches_per_epoch;

        LOG_INFO("Epoch %d/%d completed in %ld seconds, Avg Loss: %.6f",
                 epoch + 1, epochs, epoch_time, epoch_loss);
    }

    // Save trained model
    model.save_weights(MODEL_SAVE_DIR "/cpu_encoder_weights.bin");
    LOG_INFO("Training complete! Model saved.");

    return 0;
}
```

---

## 3. Testing and Verification

### 3.1 Unit Tests

**Test 1: Convolution Correctness**

```cpp
// Test against known output
void test_conv2d() {
    // Create 1×1×3×3 input
    Tensor input({1, 1, 3, 3});
    float* data = input.data->data();
    for (int i = 0; i < 9; ++i) data[i] = i + 1;

    // Create 1×1×3×3 kernel (identity-like)
    Conv2DCPU conv(1, 1, 3, 1, 0);
    std::vector<float> weights(9, 1.0f);
    std::vector<float> bias(1, 0.0f);
    conv.set_weights(weights, bias);

    auto output = conv.forward(input);

    // Output should be 1×1×1×1 with value 45 (sum of 1..9)
    assert(output.shape == std::vector<int>({1, 1, 1, 1}));
    assert(std::abs(output.data->data()[0] - 45.0f) < 1e-5);
}
```

**Test 2: Autoencoder Shape Consistency**

```cpp
void test_autoencoder_shapes() {
    AutoencoderCPU model;

    // Create dummy input
    Tensor input({4, 3, 32, 32}); // Batch of 4

    // Forward pass
    auto output = model.forward(input);

    // Check output shape
    assert(output.batch() == 4);
    assert(output.channels() == 3);
    assert(output.height() == 32);
    assert(output.width() == 32);

    // Check latent shape
    auto latent = model.extract_features(input);
    assert(latent.batch() == 4);
    assert(latent.channels() == 128);
    assert(latent.height() == 8);
    assert(latent.width() == 8);
}
```

### 3.2 Integration Tests

**Test 3: Loss Decreases**

```cpp
void test_loss_decreases() {
    CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
    dataset.load_data();

    AutoencoderCPU model;

    auto batch = dataset.get_batch(32);

    // Initial loss
    auto output1 = model.forward(batch);
    float loss1 = model.compute_loss(output1, batch);

    // Train for 10 iterations
    for (int i = 0; i < 10; ++i) {
        auto output = model.forward(batch);
        model.backward(batch, 0.001f);
    }

    // Final loss
    auto output2 = model.forward(batch);
    float loss2 = model.compute_loss(output2, batch);

    // Loss should decrease
    assert(loss2 < loss1);
    LOG_INFO("Loss: %.6f → %.6f ✓", loss1, loss2);
}
```

### 3.3 Visual Verification

**Save Reconstructed Images:**

```cpp
void save_reconstructions(AutoencoderCPU& model, CIFAR10Dataset& dataset) {
    auto batch = dataset.get_batch(10);
    auto output = model.forward(batch);

    // Save original and reconstructed images as PPM
    for (int i = 0; i < 10; ++i) {
        save_image_ppm(batch, i, "original_" + std::to_string(i) + ".ppm");
        save_image_ppm(output, i, "reconstructed_" + std::to_string(i) + ".ppm");
    }
}
```

---

## 4. Common Pitfalls

### 4.1 Data Loading Issues

❌ **Pitfall:** Reading CIFAR-10 in NHWC format instead of NCHW

```cpp
// WRONG: Channel-last format (NHWC)
int idx = (h * 32 + w) * 3 + c;

// CORRECT: Channel-first format (NCHW)
int idx = c * 1024 + h * 32 + w;
```

❌ **Pitfall:** Forgetting to normalize pixel values

```cpp
// WRONG: Using uint8 [0, 255]
float pixel = buffer[idx];

// CORRECT: Normalize to [0, 1]
float pixel = buffer[idx] / 255.0f;
```

### 4.2 Convolution Implementation

❌ **Pitfall:** Incorrect padding handling

```cpp
// WRONG: No bounds checking
int ih = oh * stride + kh;
sum += input[ih][iw] * kernel[kh][kw];

// CORRECT: Check bounds
int ih = oh * stride - pad + kh;
if (ih >= 0 && ih < input_height) {
    sum += input[ih][iw] * kernel[kh][kw];
}
```

❌ **Pitfall:** Weight indexing errors

```cpp
// WRONG: Confused channel ordering
int w_idx = ic * out_c * k * k + oc * k * k + kh * k + kw;

// CORRECT: [out_c, in_c, k, k]
int w_idx = ((oc * in_c + ic) * k + kh) * k + kw;
```

### 4.3 Training Issues

❌ **Pitfall:** Exploding/vanishing gradients

- **Solution:** Use gradient clipping, check weight initialization
- **Detection:** Monitor loss - if NaN or infinity, gradients exploded

❌ **Pitfall:** Loss not decreasing

- **Check:** Learning rate (too low or too high)
- **Check:** Gradient computation (verify with numerical gradients)
- **Check:** Weight initialization (use Xavier/He initialization)

❌ **Pitfall:** Memory leaks

- **Solution:** Use RAII, smart pointers, memory checking tools

**Linux/Ubuntu/macOS:**

```bash
valgrind --leak-check=full ./bin/train_cpu
```

**Windows 11 PowerShell:**

```powershell
# Use Visual Studio's built-in memory profiler
# Or use Dr. Memory (free alternative to Valgrind)
winget install DrMemory.DrMemory
drmemory -- .\bin\Release\train_cpu.exe

# Or use Windows Performance Analyzer
# Debug in Visual Studio with Memory Diagnostics enabled
```

---

## 5. Optimization Tips (CPU)

### 5.1 OpenMP Parallelization

```cpp
// Parallelize outer loops
#pragma omp parallel for collapse(2)
for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < out_channels; ++oc) {
        // Inner convolution loop
    }
}
```

### 5.2 Memory Layout

- **Use NCHW format:** Better cache locality for channel-wise operations
- **Align memory:** 64-byte alignment for cache lines
- **Preallocate buffers:** Avoid repeated malloc/free

### 5.3 Compiler Flags

```cmake
target_compile_options(train_cpu PRIVATE
    -O3                  # Maximum optimization
    -march=native        # Use CPU-specific instructions (AVX, AVX2)
    -fopenmp            # Enable OpenMP
    -ffast-math         # Aggressive math optimizations
)
```

---

## 6. Google Colab Notes

### 💡 For Intel Core i5 Users

**Phase 1 (CPU Baseline) can run on your local Windows 11 machine:**

- ✅ No NVIDIA GPU required for this phase
- ✅ Works on Intel Core i5 with integrated graphics
- ✅ Expected performance: 15-20 min/epoch (acceptable)
- ⚠️ For Phase 2-4 (GPU required), you MUST use Google Colab

**To run locally on Windows 11 (Intel Core i5):**

```powershell
# Follow the build instructions above, then:
cd build
.\bin\Release\train_cpu.exe --epochs 5
```

### 6.1 Setup on Colab (Optional for Phase 1)

**Note:** Phase 1 can run on CPU, so Colab is optional here. However, if you prefer cloud execution:

**Cell 1: Install dependencies**

```python
!apt-get update
!apt-get install -y cmake build-essential libomp-dev

# Clone your repository
!git clone https://github.com/YOUR_USERNAME/PP-Final_Project.git
%cd PP-Final_Project
```

**Cell 2: Download CIFAR-10**

```python
import urllib.request
import tarfile

url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
urllib.request.urlretrieve(url, "cifar-10-binary.tar.gz")

with tarfile.open("cifar-10-binary.tar.gz", "r:gz") as tar:
    tar.extractall("data/")

!ls data/cifar-10-batches-bin/
```

**Cell 3: Build and run**

```python
!mkdir -p build
%cd build
!cmake .. -DCMAKE_BUILD_TYPE=Release
!make -j4

# Run training
!./bin/train_cpu
```

### 6.2 Colab-Specific Considerations

**Runtime limits:**

- Free tier: 12 hours max session
- Save checkpoints every epoch to Google Drive

**Mount Google Drive:**

```python
from google.colab import drive
drive.mount('/content/drive')

# Save weights to Drive
!cp models/saved_weights/*.bin /content/drive/MyDrive/autoencoder_weights/
```

**Monitor resources:**

```python
import psutil
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"RAM usage: {psutil.virtual_memory().percent}%")
```

---

## 7. Deliverables Checklist

- [ ] Data loading works for all 50K training images
- [ ] All 5 convolutional layers implemented
- [ ] Forward pass produces correct output shape (32×32×3)
- [ ] Backward pass computes gradients correctly
- [ ] Training loop completes for 5 epochs
- [ ] Loss decreases from ~0.15 to <0.08
- [ ] Reconstructed images visually resemble inputs
- [ ] Weights saved to disk
- [ ] Training time recorded (~15-20 min/epoch)
- [ ] Code compiles with no warnings
- [ ] Unit tests pass

---

## 8. Next Steps

Once Phase 1 is complete and verified:

1. **Record baseline metrics:**

   - Training time per epoch
   - Final loss value
   - Memory usage
   - Visual quality of reconstructions

2. **Prepare for Phase 2:**

   - Install CUDA Toolkit
   - Verify GPU is detected (`nvidia-smi`)
   - Review CUDA programming basics

3. **Document lessons learned:**
   - What was challenging?
   - What would you optimize first on GPU?
   - Any architectural insights?

---

## 9. Reference Resources

**CIFAR-10 Dataset:**

- Official page: https://www.cs.toronto.edu/~kriz/cifar.html
- Binary format details: https://www.cs.toronto.edu/~kriz/cifar.html

**Autoencoder Theory:**

- Hinton & Salakhutdinov (2006): https://www.cs.toronto.edu/~hinton/science.pdf
- Deep Learning Book Ch 14: https://www.deeplearningbook.org/contents/autoencoders.html

**C++ Neural Networks:**

- turkdogan/autoencoder: https://github.com/turkdogan/autoencoder
- Tiny-DNN: https://github.com/tiny-dnn/tiny-dnn

**OpenMP Tutorial:**

- https://www.openmp.org/resources/tutorials-articles/

---

## 10. Quick Start for Current Task (November 29, 2025)

### 🎯 What You Should Do RIGHT NOW

**Current Status:** Data pipeline complete ✅, need to implement neural network layers.

**Step 1: Create Layer Directories**

```powershell
# Run from project root
New-Item -ItemType Directory -Force -Path include/layers
New-Item -ItemType Directory -Force -Path src/layers
New-Item -ItemType Directory -Force -Path tests
```

**Step 2: Create Conv2D Header File**

Create `include/layers/conv2d_cpu.h` - copy the code from Step 1.2.1 above.

**Step 3: Create Conv2D Implementation**

Create `src/layers/conv2d_cpu.cpp` - implement forward pass first, then backward.

**Step 4: Create Simple Test**

Create `tests/test_conv2d.cpp`:

```cpp
#include "layers/conv2d_cpu.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "Testing Conv2D layer...\n";

    // Create simple 1×1×3×3 input
    Tensor input({1, 1, 3, 3});
    float* data = input.data;
    for (int i = 0; i < 9; ++i) data[i] = static_cast<float>(i + 1);

    // Create Conv2D: 1 input channel → 1 output channel, 3×3 kernel
    Conv2DCPU conv(1, 1, 3, 1, 0);

    // Set weights to all 1.0 (simple test)
    std::vector<float> weights(9, 1.0f);
    std::vector<float> bias(1, 0.0f);
    conv.set_weights(weights, bias);

    // Forward pass
    auto output = conv.forward(input);

    // Output should be 1×1×1×1 with value 45 (sum of 1..9)
    std::cout << "Output shape: [" << output.batch() << ", "
              << output.channels() << ", " << output.height()
              << ", " << output.width() << "]\n";
    std::cout << "Output value: " << output.data[0] << "\n";

    assert(output.batch() == 1);
    assert(output.channels() == 1);
    assert(output.height() == 1);
    assert(output.width() == 1);
    assert(std::abs(output.data[0] - 45.0f) < 1e-4);

    std::cout << "✓ Conv2D test passed!\n";
    return 0;
}
```

**Step 5: Update CMakeLists.txt**

Add to your CMakeLists.txt:

```cmake
# After your existing source files
set(LAYER_SOURCES
    src/layers/conv2d_cpu.cpp
)

# Update train_autoencoder target
add_executable(train_autoencoder
    src/main_train.cpp
    src/data/cifar10_dataset.cpp
    src/data/data_utils.cpp
    src/utils/logger.cpp
    src/utils/memory_pool.cpp
    ${LAYER_SOURCES}
)

# Add test executable
add_executable(test_conv2d
    tests/test_conv2d.cpp
    src/layers/conv2d_cpu.cpp
    src/utils/logger.cpp
)
target_link_libraries(test_conv2d PRIVATE OpenMP::OpenMP_CXX)
```

**Step 6: Build and Test**

```powershell
cd build
cmake --build . --config Release

# Run test
.\bin\Release\test_conv2d.exe
# Expected: "✓ Conv2D test passed!"
```

**Step 7: Repeat for Other Layers**

Once Conv2D works:

1. Create `relu_cpu.h/.cpp` (easier - good practice)
2. Create `maxpool_cpu.h/.cpp`
3. Create `upsample_cpu.h/.cpp`
4. Test each one individually

### 📋 Verification Checklist

After implementing all layers, verify:

```powershell
# All layers compile
cmake --build . --config Release

# All tests pass
.\bin\Release\test_conv2d.exe
.\bin\Release\test_relu.exe
.\bin\Release\test_maxpool.exe
.\bin\Release\test_upsample.exe

# No memory leaks (optional but recommended)
# Use Dr. Memory on Windows
drmemory -- .\bin\Release\test_conv2d.exe
```

### 🎓 Learning Tips

1. **Start Simple:** Implement forward pass first, test it, then add backward pass
2. **Use Small Inputs:** Test with 1×1×3×3 tensors before full 32×32×3 images
3. **Check Shapes:** Print tensor shapes at each step to catch dimension errors early
4. **Verify Gradients:** Use numerical gradient checking for backward pass
5. **OpenMP Later:** Get correctness first, add `#pragma omp parallel for` after

### 🔗 Reference Implementation

See **Step 1.2.1** above for complete Conv2D implementation code.
See **Task 1.2.2-1.2.4** for ReLU, MaxPool, and Upsample implementations.

### ⏱️ Time Estimate

- Conv2D: 2-3 hours (most complex)
- ReLU: 30 minutes (very simple)
- MaxPool: 1 hour
- Upsample: 1 hour
- Testing: 1 hour

**Total: ~6 hours** (rest of Day 2 + morning of Day 3)

---

**Good luck with Phase 1! Take your time to get the foundation right - it will make GPU optimization much easier.**
