# Phase 1 Implementation Documentation

## Data Loading and Preprocessing + CPU Neural Network Layers

**Phase Duration:** November 28 - December 1, 2025  
**Status:** ✅ Complete  
**Date Completed:** 2025

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Loading Implementation](#2-data-loading-implementation)
3. [CPU Layer Implementation](#3-cpu-layer-implementation)
4. [Autoencoder Architecture](#4-autoencoder-architecture)
5. [Training Implementation](#5-training-implementation)
6. [Testing and Verification](#6-testing-and-verification)
7. [References](#7-references)

---

## 1. Overview

Phase 1 implements the complete CPU baseline for the CIFAR-10 autoencoder project. This serves as:

1. **Foundation** for GPU optimization in later phases
2. **Reference implementation** for correctness verification
3. **Performance baseline** for speedup measurements

### What Was Implemented

| Component | Files | Description |
|-----------|-------|-------------|
| Data Loading | `cifar10_dataset.cpp/h` | CIFAR-10 binary file parsing, normalization |
| Conv2D Layer | `conv2d_cpu.cpp/h` | 2D convolution with forward/backward pass |
| ReLU Layer | `relu_cpu.cpp/h` | Rectified Linear Unit activation |
| MaxPool Layer | `maxpool_cpu.cpp/h` | 2×2 max pooling with index tracking |
| Upsample Layer | `upsample_cpu.cpp/h` | Nearest neighbor 2× upsampling |
| Loss Functions | `loss_functions.cpp/h` | MSE loss and gradient computation |
| Autoencoder | `autoencoder_cpu.cpp/h` | Complete encoder-decoder model |
| Training | `train_cpu.cpp` | Full training loop with SGD |

### Architecture Summary

```
INPUT (32×32×3 = 3,072 values)
    ↓
ENCODER:
    Conv2D(256, 3×3, pad=1) + ReLU  →  (32×32×256)
    MaxPool(2×2)                     →  (16×16×256)
    Conv2D(128, 3×3, pad=1) + ReLU  →  (16×16×128)
    MaxPool(2×2)                     →  (8×8×128)
    ↓
LATENT (8×8×128 = 8,192 features)
    ↓
DECODER:
    Conv2D(128, 3×3, pad=1) + ReLU  →  (8×8×128)
    UpSample(2×2)                    →  (16×16×128)
    Conv2D(256, 3×3, pad=1) + ReLU  →  (16×16×256)
    UpSample(2×2)                    →  (32×32×256)
    Conv2D(3, 3×3, pad=1)            →  (32×32×3)
    ↓
OUTPUT (32×32×3 = 3,072 values)
```

**Total Parameters:** 751,875

---

## 2. Data Loading Implementation

### 2.1 CIFAR-10 Binary Format

CIFAR-10 stores images in a binary format with each record containing:

```
[1 byte: label] [3072 bytes: image data]
```

The image data is stored in **planar RGB format** (channel-first):
- Bytes 1-1024: Red channel (row-major)
- Bytes 1025-2048: Green channel (row-major)
- Bytes 2049-3072: Blue channel (row-major)

**Reference:** https://www.cs.toronto.edu/~kriz/cifar.html

### 2.2 Implementation Details

**File:** `src/data/cifar10_dataset.cpp`

```cpp
void CIFAR10Dataset::load_batch(const std::string& filename, int start_idx) {
    // CIFAR-10 binary is already in NCHW planar format (channel-first)
    // Buffer layout: [label, R0..R1023, G0..G1023, B0..B1023]
    
    for (int c = 0; c < CIFAR_CHANNELS; ++c) {
        for (int h = 0; h < CIFAR_IMAGE_SIZE; ++h) {
            for (int w = 0; w < CIFAR_IMAGE_SIZE; ++w) {
                // Source: buffer offset = 1 (skip label) + channel*1024 + h*32 + w
                int src_idx = 1 + c * (CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE) 
                            + h * CIFAR_IMAGE_SIZE + w;
                // Destination: NCHW format = c*1024 + h*32 + w
                int dst_idx = c * (CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE) 
                            + h * CIFAR_IMAGE_SIZE + w;
                
                // Normalize [0, 255] to [0, 1]
                img_ptr[dst_idx] = static_cast<float>(buffer[src_idx]) / 255.0f;
            }
        }
    }
}
```

### 2.3 Key Features

- **NCHW Format:** Data is stored in channel-first format for optimal GPU performance
- **Normalization:** Pixel values normalized to [0, 1] range during loading
- **Efficient Shuffling:** Uses index-based shuffling (O(n) indices, not O(n*3072) data)
- **OpenMP Parallelization:** Batch generation is parallelized

---

## 3. CPU Layer Implementation

### 3.1 Conv2D Layer

**Mathematical Operation:**

```
output[n,oc,oh,ow] = bias[oc] + 
    Σ_{ic,kh,kw} input[n,ic,ih,iw] * weight[oc,ic,kh,kw]

where:
    ih = oh * stride - padding + kh
    iw = ow * stride - padding + kw
```

**Implementation Features:**

1. **Weight Initialization:** He initialization for ReLU compatibility
   ```cpp
   float std_dev = sqrt(2.0f / fan_in);  // fan_in = in_channels * k * k
   ```

2. **Forward Pass:** Nested loops with OpenMP parallelization
   ```cpp
   #pragma omp parallel for collapse(2) schedule(static)
   for (int n = 0; n < batch; ++n) {
       for (int oc = 0; oc < out_channels_; ++oc) {
           // Convolution computation
       }
   }
   ```

3. **Backward Pass:** Computes three gradients:
   - Gradient w.r.t. input (for backpropagation)
   - Gradient w.r.t. weights (for weight update)
   - Gradient w.r.t. bias (for bias update)

**Reference Implementation:**
- https://github.com/turkdogan/autoencoder
- CS231n: https://cs231n.github.io/convolutional-networks/

### 3.2 ReLU Activation

**Mathematical Operation:**
```
ReLU(x) = max(0, x)
d_ReLU/dx = 1 if x > 0, else 0
```

**Implementation:**
```cpp
#pragma omp parallel for simd schedule(static)
for (size_t i = 0; i < size; ++i) {
    out_data[i] = std::max(0.0f, in_data[i]);
}
```

### 3.3 MaxPool Layer

**Operation:** Takes maximum value in each 2×2 window

**Key Implementation Detail:** Stores indices of maximum values for efficient backward pass:

```cpp
std::vector<int> max_indices_;  // For gradient routing

// Forward: store index of max
max_indices_[out_idx] = max_idx;

// Backward: route gradient to max position
grad_in_data[max_indices_[out_idx]] += grad_out_data[out_idx];
```

### 3.4 Upsample Layer (Nearest Neighbor)

**Operation:** Each input pixel is replicated to a 2×2 block

```cpp
// Forward
int ih = oh / scale_;
int iw = ow / scale_;
out_data[out_idx] = in_data[in_idx];

// Backward: sum gradients from replicated positions
for (int dh = 0; dh < scale_; ++dh) {
    for (int dw = 0; dw < scale_; ++dw) {
        sum += grad_out_data[out_idx];
    }
}
```

### 3.5 MSE Loss

**Loss:**
```
L = (1/N) * Σ(output[i] - target[i])²
```

**Gradient:**
```
dL/d_output = (2/N) * (output - target)
```

---

## 4. Autoencoder Architecture

### 4.1 Layer Configuration

| Layer | Type | Input Shape | Output Shape | Parameters |
|-------|------|-------------|--------------|------------|
| enc_conv1 | Conv2D(3→256, 3×3, pad=1) | (N,3,32,32) | (N,256,32,32) | 7,168 |
| enc_relu1 | ReLU | (N,256,32,32) | (N,256,32,32) | 0 |
| enc_pool1 | MaxPool(2×2) | (N,256,32,32) | (N,256,16,16) | 0 |
| enc_conv2 | Conv2D(256→128, 3×3, pad=1) | (N,256,16,16) | (N,128,16,16) | 295,040 |
| enc_relu2 | ReLU | (N,128,16,16) | (N,128,16,16) | 0 |
| enc_pool2 | MaxPool(2×2) | (N,128,16,16) | (N,128,8,8) | 0 |
| **LATENT** | - | - | **(N,128,8,8)** | - |
| dec_conv1 | Conv2D(128→128, 3×3, pad=1) | (N,128,8,8) | (N,128,8,8) | 147,584 |
| dec_relu1 | ReLU | (N,128,8,8) | (N,128,8,8) | 0 |
| dec_up1 | Upsample(2×) | (N,128,8,8) | (N,128,16,16) | 0 |
| dec_conv2 | Conv2D(128→256, 3×3, pad=1) | (N,128,16,16) | (N,256,16,16) | 295,168 |
| dec_relu2 | ReLU | (N,256,16,16) | (N,256,16,16) | 0 |
| dec_up2 | Upsample(2×) | (N,256,16,16) | (N,256,32,32) | 0 |
| dec_conv3 | Conv2D(256→3, 3×3, pad=1) | (N,256,32,32) | (N,3,32,32) | 6,915 |

**Total: 751,875 parameters** (matches Keras reference)

### 4.2 Parameter Calculation

```
Layer          | Weights                    | Bias  | Total
---------------|----------------------------|-------|--------
enc_conv1      | 3 × 256 × 3 × 3 = 6,912   | 256   | 7,168
enc_conv2      | 256 × 128 × 3 × 3 = 294,912| 128   | 295,040
dec_conv1      | 128 × 128 × 3 × 3 = 147,456| 128   | 147,584
dec_conv2      | 128 × 256 × 3 × 3 = 294,912| 256   | 295,168
dec_conv3      | 256 × 3 × 3 × 3 = 6,912   | 3     | 6,915
---------------|----------------------------|-------|--------
TOTAL          |                            |       | 751,875
```

---

## 5. Training Implementation

### 5.1 Training Loop

**File:** `src/train_cpu.cpp`

```cpp
for (int epoch = 0; epoch < config.epochs; ++epoch) {
    train_dataset.shuffle();
    train_dataset.reset();
    
    for (int batch = 0; batch < batches_per_epoch; ++batch) {
        // 1. Get batch of images
        Tensor images = train_dataset.get_batch(config.batch_size);
        
        // 2. Forward pass
        Tensor output = model.forward(images);
        
        // 3. Backward pass and weight update
        // Note: For autoencoder, target = input (reconstruction)
        float batch_loss = model.backward(images, config.learning_rate);
    }
}
```

### 5.2 SGD Optimizer

Weight update rule:
```
weight = weight - learning_rate * gradient
```

Default hyperparameters:
- **Batch size:** 32
- **Learning rate:** 0.001
- **Epochs:** 20

### 5.3 Weight Save/Load

Binary format for weights:
```
[4 bytes: magic number 0x41455743]
[4 bytes: num_layers]
For each layer:
    [4 bytes: weight_size]
    [weight_size × 4 bytes: weights]
    [4 bytes: bias_size]
    [bias_size × 4 bytes: biases]
```

---

## 6. Testing and Verification

### 6.1 Test Suite

**File:** `tests/test_cpu_layers.cpp`

| Test | Description | Result |
|------|-------------|--------|
| Conv2D forward | Check output shape [2,64,32,32] | ✓ Pass |
| Conv2D backward | Check gradient shape | ✓ Pass |
| ReLU forward | All outputs ≥ 0 | ✓ Pass |
| ReLU shape | Shape preserved | ✓ Pass |
| MaxPool forward | Output halved [2,3,8,8] | ✓ Pass |
| MaxPool backward | Gradient shape | ✓ Pass |
| Upsample forward | Output doubled [2,3,16,16] | ✓ Pass |
| Upsample values | Values replicated | ✓ Pass |
| MSE loss | Positive loss | ✓ Pass |
| MSE gradient | Shape match | ✓ Pass |
| Autoencoder params | ~751,875 | ✓ Pass |
| Autoencoder shape | [4,3,32,32] | ✓ Pass |
| Latent shape | [4,128,8,8] | ✓ Pass |
| Training step | No errors | ✓ Pass |
| Weight I/O | Save/load | ✓ Pass |

**Total: 23 tests passing**

### 6.2 Running Tests

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run layer tests
./bin/test_cpu_layers

# Run data loading tests (requires CIFAR-10 dataset)
./bin/test_data_loading
```

---

## 7. References

### Academic Papers

1. **Autoencoders:**
   - Hinton & Salakhutdinov (2006). "Reducing the Dimensionality of Data with Neural Networks"
   - https://www.cs.toronto.edu/~hinton/science.pdf

2. **He Initialization:**
   - He et al. (2015). "Delving Deep into Rectifiers"
   - https://arxiv.org/abs/1502.01852

### Reference Implementations

1. **turkdogan/autoencoder:** https://github.com/turkdogan/autoencoder
2. **tbennun/cudnn-training:** https://github.com/tbennun/cudnn-training

### Educational Resources

1. **CS231n Convolutional Neural Networks:**
   - https://cs231n.github.io/convolutional-networks/

2. **Deep Learning Book Chapter 14:**
   - https://www.deeplearningbook.org/contents/autoencoders.html

### Dataset

1. **CIFAR-10 Official Page:**
   - https://www.cs.toronto.edu/~kriz/cifar.html

---

## Next Steps: Phase 2

With Phase 1 complete, the next phase involves:

1. **GPU Memory Management:** Allocate device memory for weights and activations
2. **Naive GPU Kernels:** Port all layers to basic CUDA kernels
3. **Correctness Verification:** Ensure GPU outputs match CPU within tolerance
4. **Initial Speedup Measurement:** Target 5-10× speedup

See `docs/PHASE_2_GUIDE.md` for detailed implementation guide.

---

**Phase 1 Complete!** ✅
