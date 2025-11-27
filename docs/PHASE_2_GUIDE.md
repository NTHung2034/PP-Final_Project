# Phase 2: Naive GPU Implementation Guide

**Duration:** December 2-4, 2025 (3 days)  
**Objective:** Port CPU implementation to GPU with basic parallelization  
**Prerequisite:** Phase 1 complete, CUDA Toolkit installed

---

## Table of Contents

1. [Overview](#1-overview)
2. [Step-by-Step Implementation](#2-step-by-step-implementation)
3. [Testing and Verification](#3-testing-and-verification)
4. [Common Pitfalls](#4-common-pitfalls)
5. [Profiling Guide](#5-profiling-guide)
6. [Google Colab Notes](#6-google-colab-notes)

---

## 1. Overview

### 1.1 Goals

By the end of Phase 2, you will have:

- ✅ All layers ported to GPU (CUDA kernels)
- ✅ GPU memory management working
- ✅ Forward and backward passes running on GPU
- ✅ 5-10× speedup over CPU baseline
- ✅ Correctness verified (GPU matches CPU output)
- ✅ Profiling data collected

### 1.2 GPU Parallelization Strategy

**Key Principle:** Map each output element to one CUDA thread

| Operation | Thread Mapping            | Workload per Thread                   |
| --------- | ------------------------- | ------------------------------------- |
| Conv2D    | 1 thread = 1 output pixel | Loop over input channels + kernel     |
| ReLU      | 1 thread = 1 element      | `y = max(0, x)`                       |
| MaxPool   | 1 thread = 1 output pixel | Find max in 2×2 window                |
| Upsample  | 1 thread = 1 output pixel | Copy from corresponding input pixel   |
| MSE Loss  | Reduction (parallel sum)  | Compute partial sums in shared memory |

### 1.3 Expected Performance

| Metric              | Target Value |
| ------------------- | ------------ |
| Training time/epoch | 2-3 minutes  |
| Speedup vs CPU      | 5-10×        |
| GPU memory usage    | ~2-3 GB      |
| Batch size          | 64 images    |

---

## 2. Step-by-Step Implementation

### **Task 2.1: GPU Memory Management (Day 1 Morning)**

#### Step 2.1.1: Create GPU Tensor Class

**File:** `include/cuda/gpu_tensor.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

class GPUTensor {
public:
    std::vector<int> shape;
    float* d_data;  // Device pointer
    size_t size;

    // Constructor: allocate device memory
    GPUTensor(const std::vector<int>& dims) : shape(dims), d_data(nullptr) {
        size = 1;
        for (int d : dims) size *= d;

        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_data, 0, size * sizeof(float)));
    }

    // Destructor: free device memory
    ~GPUTensor() {
        if (d_data) cudaFree(d_data);
    }

    // Disable copy, enable move
    GPUTensor(const GPUTensor&) = delete;
    GPUTensor& operator=(const GPUTensor&) = delete;

    GPUTensor(GPUTensor&& other) noexcept
        : shape(std::move(other.shape)), d_data(other.d_data), size(other.size) {
        other.d_data = nullptr;
    }

    // Copy data from host to device
    void from_host(const float* h_data) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Copy data from device to host
    void to_host(float* h_data) const {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // Copy data from another GPU tensor
    void copy_from(const GPUTensor& other) {
        if (size != other.size) {
            throw std::runtime_error("Tensor size mismatch");
        }
        CUDA_CHECK(cudaMemcpy(d_data, other.d_data, size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    // Dimensions
    int batch() const { return shape[0]; }
    int channels() const { return shape[1]; }
    int height() const { return shape[2]; }
    int width() const { return shape[3]; }
};
```

#### Step 2.1.2: Test GPU Memory Operations

**File:** `tests/test_gpu_memory.cu`

```cpp
#include "cuda/gpu_tensor.cuh"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing GPU memory management...\n";

    // Test 1: Allocation
    GPUTensor tensor({2, 3, 4, 4});  // Batch=2, C=3, H=4, W=4
    std::cout << "✓ Allocated " << tensor.size << " floats on GPU\n";

    // Test 2: Host to Device
    std::vector<float> h_data(tensor.size);
    for (size_t i = 0; i < h_data.size(); ++i) {
        h_data[i] = i * 0.1f;
    }
    tensor.from_host(h_data.data());
    std::cout << "✓ Copied data to GPU\n";

    // Test 3: Device to Host
    std::vector<float> h_result(tensor.size);
    tensor.to_host(h_result.data());
    std::cout << "✓ Copied data from GPU\n";

    // Test 4: Verify correctness
    bool correct = true;
    for (size_t i = 0; i < h_data.size(); ++i) {
        if (std::abs(h_data[i] - h_result[i]) > 1e-6) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "✓ All tests passed!\n";
    } else {
        std::cerr << "✗ Data mismatch!\n";
        return 1;
    }

    return 0;
}
```

**Build and run:**

```bash
nvcc -o test_gpu_memory tests/test_gpu_memory.cu -I include
./test_gpu_memory
```

---

### **Task 2.2: Naive CUDA Kernels (Day 1 Afternoon - Day 2)**

#### Step 2.2.1: Convolution Kernel

**File:** `src/cuda/kernels/conv2d_kernel.cu`

```cpp
#include "cuda/kernels/conv2d_kernel.cuh"

__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,    // [batch, in_c, in_h, in_w]
    const float* __restrict__ weights,  // [out_c, in_c, k, k]
    const float* __restrict__ bias,     // [out_c]
    float* __restrict__ output,         // [batch, out_c, out_h, out_w]
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k_size, int stride, int pad
) {
    // Each thread computes one output element
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % out_c;
    int n = blockIdx.z / out_c;

    if (ow >= out_w || oh >= out_h || n >= batch) return;

    float sum = bias[oc];

    // Convolution: loop over input channels and kernel
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k_size; ++kh) {
            for (int kw = 0; kw < k_size; ++kw) {
                int ih = oh * stride - pad + kh;
                int iw = ow * stride - pad + kw;

                // Check bounds (zero padding)
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                    int w_idx = ((oc * in_c + ic) * k_size + kh) * k_size + kw;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }
    }

    int out_idx = ((n * out_c + oc) * out_h + oh) * out_w + ow;
    output[out_idx] = sum;
}

void launch_conv2d_forward(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k_size, int stride, int pad
) {
    // Configure threads and blocks
    dim3 threads(16, 16);  // 16×16 = 256 threads per block
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        batch * out_c
    );

    conv2d_forward_kernel<<<blocks, threads>>>(
        d_input, d_weights, d_bias, d_output,
        batch, in_c, in_h, in_w,
        out_c, out_h, out_w,
        k_size, stride, pad
    );

    CUDA_CHECK(cudaGetLastError());
}
```

**Key Points:**

- **Thread mapping:** 1 thread computes 1 output pixel
- **Memory access:** Global memory (naive approach)
- **Bounds checking:** Handle zero padding
- **Block size:** 16×16 = 256 threads (good occupancy)

#### Step 2.2.2: ReLU Kernel

**File:** `src/cuda/kernels/relu_kernel.cu`

```cpp
__global__ void relu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_backward_kernel(
    const float* __restrict__ input,      // Original input (cached)
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

void launch_relu_forward(const float* d_input, float* d_output, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    relu_forward_kernel<<<blocks, threads>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu_backward(const float* d_input, const float* d_grad_output,
                          float* d_grad_input, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(d_input, d_grad_output,
                                               d_grad_input, size);
    CUDA_CHECK(cudaGetLastError());
}
```

#### Step 2.2.3: MaxPooling Kernel

**File:** `src/cuda/kernels/maxpool_kernel.cu`

```cpp
__global__ void maxpool_forward_kernel(
    const float* __restrict__ input,   // [batch, channels, in_h, in_w]
    float* __restrict__ output,        // [batch, channels, out_h, out_w]
    int* __restrict__ max_indices,     // Store indices for backward pass
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w, int pool_size
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;

    if (ow >= out_w || oh >= out_h || n >= batch) return;

    float max_val = -1e38f;  // Very small number
    int max_idx = 0;

    // Find maximum in pool_size × pool_size window
    for (int ph = 0; ph < pool_size; ++ph) {
        for (int pw = 0; pw < pool_size; ++pw) {
            int ih = oh * pool_size + ph;
            int iw = ow * pool_size + pw;

            int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;

            if (input[in_idx] > max_val) {
                max_val = input[in_idx];
                max_idx = in_idx;
            }
        }
    }

    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
    output[out_idx] = max_val;
    max_indices[out_idx] = max_idx;
}

void launch_maxpool_forward(
    const float* d_input, float* d_output, int* d_max_indices,
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w, int pool_size
) {
    dim3 threads(16, 16);
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        batch * channels
    );

    maxpool_forward_kernel<<<blocks, threads>>>(
        d_input, d_output, d_max_indices,
        batch, channels, in_h, in_w,
        out_h, out_w, pool_size
    );

    CUDA_CHECK(cudaGetLastError());
}
```

#### Step 2.2.4: Upsampling Kernel (Nearest Neighbor)

**File:** `src/cuda/kernels/upsample_kernel.cu`

```cpp
__global__ void upsample_forward_kernel(
    const float* __restrict__ input,   // [batch, channels, in_h, in_w]
    float* __restrict__ output,        // [batch, channels, out_h, out_w]
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w, int scale
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;

    if (ow >= out_w || oh >= out_h || n >= batch) return;

    // Map output coordinate to input coordinate (nearest neighbor)
    int ih = oh / scale;
    int iw = ow / scale;

    int in_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;

    output[out_idx] = input[in_idx];
}

void launch_upsample_forward(
    const float* d_input, float* d_output,
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w, int scale
) {
    dim3 threads(16, 16);
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        batch * channels
    );

    upsample_forward_kernel<<<blocks, threads>>>(
        d_input, d_output,
        batch, channels, in_h, in_w,
        out_h, out_w, scale
    );

    CUDA_CHECK(cudaGetLastError());
}
```

#### Step 2.2.5: MSE Loss Kernel (Parallel Reduction)

**File:** `src/cuda/kernels/loss_kernel.cu`

```cpp
__global__ void mse_loss_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ partial_sums,  // Per-block partial sums
    int size
) {
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute squared error for this thread
    float error = 0.0f;
    if (idx < size) {
        float diff = output[idx] - target[idx];
        error = diff * diff;
    }
    shared_sum[tid] = error;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

float launch_mse_loss(const float* d_output, const float* d_target, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Allocate temporary buffer for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, blocks * sizeof(float)));

    // Launch kernel
    mse_loss_kernel<<<blocks, threads>>>(d_output, d_target, d_partial_sums, size);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums to host and compute final sum
    std::vector<float> h_partial_sums(blocks);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums,
                          blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (float s : h_partial_sums) {
        total_loss += s;
    }

    cudaFree(d_partial_sums);

    return total_loss / size;  // Mean squared error
}
```

---

### **Task 2.3: GPU Autoencoder Class (Day 2 Afternoon)**

#### Step 2.3.1: Create GPU Autoencoder

**File:** `include/models/autoencoder_gpu.cuh`

```cpp
#pragma once
#include "cuda/gpu_tensor.cuh"
#include "cuda/kernels/conv2d_kernel.cuh"
#include "cuda/kernels/relu_kernel.cuh"
#include "cuda/kernels/maxpool_kernel.cuh"
#include "cuda/kernels/upsample_kernel.cuh"
#include "cuda/kernels/loss_kernel.cuh"
#include <memory>

class AutoencoderGPU {
public:
    AutoencoderGPU();
    ~AutoencoderGPU();

    // Forward pass
    void forward(const float* h_input, int batch_size);

    // Get output
    void get_output(float* h_output);

    // Compute loss
    float compute_loss(const float* h_target, int batch_size);

    // Backward pass + weight update
    void backward(const float* h_target, float learning_rate, int batch_size);

    // Extract features (encoder only)
    void extract_features(const float* h_input, float* h_features, int batch_size);

    // Weight management
    void save_weights(const std::string& filename);
    void load_weights(const std::string& filename);

private:
    // Device memory for weights
    float *d_conv1_w_, *d_conv1_b_;  // Conv1: 3→256
    float *d_conv2_w_, *d_conv2_b_;  // Conv2: 256→128
    float *d_conv3_w_, *d_conv3_b_;  // Conv3: 128→128
    float *d_conv4_w_, *d_conv4_b_;  // Conv4: 128→256
    float *d_conv5_w_, *d_conv5_b_;  // Conv5: 256→3

    // Device memory for activations (cached for backward pass)
    std::unique_ptr<GPUTensor> input_;
    std::unique_ptr<GPUTensor> conv1_out_;
    std::unique_ptr<GPUTensor> relu1_out_;
    std::unique_ptr<GPUTensor> pool1_out_;
    std::unique_ptr<GPUTensor> conv2_out_;
    std::unique_ptr<GPUTensor> relu2_out_;
    std::unique_ptr<GPUTensor> latent_;      // (8×8×128)
    std::unique_ptr<GPUTensor> conv3_out_;
    std::unique_ptr<GPUTensor> relu3_out_;
    std::unique_ptr<GPUTensor> up1_out_;
    std::unique_ptr<GPUTensor> conv4_out_;
    std::unique_ptr<GPUTensor> relu4_out_;
    std::unique_ptr<GPUTensor> up2_out_;
    std::unique_ptr<GPUTensor> output_;

    // Device memory for gradients
    float *d_conv1_grad_w_, *d_conv1_grad_b_;
    float *d_conv2_grad_w_, *d_conv2_grad_b_;
    float *d_conv3_grad_w_, *d_conv3_grad_b_;
    float *d_conv4_grad_w_, *d_conv4_grad_b_;
    float *d_conv5_grad_w_, *d_conv5_grad_b_;

    // Helper: allocate weight memory
    void allocate_weights();
    void allocate_gradients();
    void initialize_weights();
};
```

#### Step 2.3.2: Implement Forward Pass

**File:** `src/models/autoencoder_gpu.cu`

```cpp
void AutoencoderGPU::forward(const float* h_input, int batch_size) {
    // Allocate activation tensors if needed
    if (!input_) {
        input_ = std::make_unique<GPUTensor>(
            std::vector<int>{batch_size, 3, 32, 32});
        conv1_out_ = std::make_unique<GPUTensor>(
            std::vector<int>{batch_size, 256, 32, 32});
        // ... allocate all other tensors ...
    }

    // Copy input to GPU
    input_->from_host(h_input);

    // ENCODER
    // Conv1: (32,32,3) → (32,32,256)
    launch_conv2d_forward(
        input_->d_data, d_conv1_w_, d_conv1_b_, conv1_out_->d_data,
        batch_size, 3, 32, 32,
        256, 32, 32,
        3, 1, 1  // kernel=3, stride=1, pad=1
    );

    // ReLU1
    launch_relu_forward(conv1_out_->d_data, relu1_out_->d_data,
                        relu1_out_->size);

    // MaxPool1: (32,32,256) → (16,16,256)
    launch_maxpool_forward(
        relu1_out_->d_data, pool1_out_->d_data, d_maxpool1_indices_,
        batch_size, 256, 32, 32,
        16, 16, 2  // pool_size=2
    );

    // Conv2: (16,16,256) → (16,16,128)
    launch_conv2d_forward(
        pool1_out_->d_data, d_conv2_w_, d_conv2_b_, conv2_out_->d_data,
        batch_size, 256, 16, 16,
        128, 16, 16,
        3, 1, 1
    );

    // ReLU2
    launch_relu_forward(conv2_out_->d_data, relu2_out_->d_data,
                        relu2_out_->size);

    // MaxPool2: (16,16,128) → (8,8,128) = LATENT
    launch_maxpool_forward(
        relu2_out_->d_data, latent_->d_data, d_maxpool2_indices_,
        batch_size, 128, 16, 16,
        8, 8, 2
    );

    // DECODER
    // Conv3: (8,8,128) → (8,8,128)
    launch_conv2d_forward(
        latent_->d_data, d_conv3_w_, d_conv3_b_, conv3_out_->d_data,
        batch_size, 128, 8, 8,
        128, 8, 8,
        3, 1, 1
    );

    // ReLU3
    launch_relu_forward(conv3_out_->d_data, relu3_out_->d_data,
                        relu3_out_->size);

    // Upsample1: (8,8,128) → (16,16,128)
    launch_upsample_forward(
        relu3_out_->d_data, up1_out_->d_data,
        batch_size, 128, 8, 8,
        16, 16, 2  // scale=2
    );

    // Conv4: (16,16,128) → (16,16,256)
    launch_conv2d_forward(
        up1_out_->d_data, d_conv4_w_, d_conv4_b_, conv4_out_->d_data,
        batch_size, 128, 16, 16,
        256, 16, 16,
        3, 1, 1
    );

    // ReLU4
    launch_relu_forward(conv4_out_->d_data, relu4_out_->d_data,
                        relu4_out_->size);

    // Upsample2: (16,16,256) → (32,32,256)
    launch_upsample_forward(
        relu4_out_->d_data, up2_out_->d_data,
        batch_size, 256, 16, 16,
        32, 32, 2
    );

    // Conv5: (32,32,256) → (32,32,3)
    launch_conv2d_forward(
        up2_out_->d_data, d_conv5_w_, d_conv5_b_, output_->d_data,
        batch_size, 256, 32, 32,
        3, 32, 32,
        3, 1, 1
    );
}
```

---

### **Task 2.4: GPU Training Loop (Day 3)**

#### Step 2.4.1: Implement Training Script

**File:** `src/train_gpu.cu`

```cpp
#include "models/autoencoder_gpu.cuh"
#include "data/cifar10_dataset.h"
#include "utils/logger.h"
#include "config.h"
#include <chrono>

int main() {
    LOG_INIT();
    LOG_INFO("Starting GPU Training");

    // Load dataset
    CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
    dataset.load_data();
    LOG_INFO("Loaded %zu training images", dataset.size());

    // Create GPU model
    AutoencoderGPU model;

    // Training configuration
    const int batch_size = 64;  // Larger batch for GPU
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
            // Get batch (on CPU)
            auto images = dataset.get_batch(batch_size);
            float* h_batch = images.data->data();

            // Forward pass (on GPU)
            model.forward(h_batch, batch_size);

            // Compute loss
            float loss = model.compute_loss(h_batch, batch_size);
            epoch_loss += loss;

            // Backward pass + update (on GPU)
            model.backward(h_batch, lr, batch_size);

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
    model.save_weights(MODEL_SAVE_DIR "/gpu_encoder_weights.bin");
    LOG_INFO("Training complete! Model saved.");

    // Print GPU memory usage
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    LOG_INFO("GPU Memory: %.2f GB used / %.2f GB total",
             (total_mem - free_mem) / 1e9, total_mem / 1e9);

    return 0;
}
```

---

## 3. Testing and Verification

### 3.1 Correctness Tests

**Test 1: GPU vs CPU Output Comparison**

```cpp
#include "models/autoencoder_cpu.h"
#include "models/autoencoder_gpu.cuh"
#include <cmath>

void test_gpu_correctness() {
    // Create dummy input
    const int batch_size = 4;
    Tensor cpu_input({batch_size, 3, 32, 32});
    float* data = cpu_input.data->data();
    for (size_t i = 0; i < cpu_input.size(); ++i) {
        data[i] = (rand() % 1000) / 1000.0f;
    }

    // CPU forward pass
    AutoencoderCPU cpu_model;
    auto cpu_output = cpu_model.forward(cpu_input);

    // GPU forward pass
    AutoencoderGPU gpu_model;
    gpu_model.load_weights("same_weights.bin");  // Use same weights
    gpu_model.forward(data, batch_size);

    std::vector<float> gpu_output(cpu_output.size());
    gpu_model.get_output(gpu_output.data());

    // Compare outputs
    float max_diff = 0.0f;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        float diff = std::abs(cpu_output.data->data()[i] - gpu_output[i]);
        max_diff = std::max(max_diff, diff);
    }

    printf("Max difference: %e\n", max_diff);
    assert(max_diff < 1e-4);  // Allow small floating-point differences
    printf("✓ GPU output matches CPU!\n");
}
```

### 3.2 Memory Leak Check

```bash
# Use cuda-memcheck to detect memory errors
cuda-memcheck --leak-check full ./bin/train_gpu
```

**Expected:** No memory leaks, no out-of-bounds accesses

### 3.3 Performance Benchmark

```cpp
void benchmark_gpu() {
    AutoencoderGPU model;

    const int batch_size = 64;
    std::vector<float> dummy_input(batch_size * 3 * 32 * 32);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        model.forward(dummy_input.data(), batch_size);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        model.forward(dummy_input.data(), batch_size);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    printf("Average forward pass time: %.2f ms\n", duration / 100.0);
}
```

---

## 4. Common Pitfalls

### 4.1 CUDA-Specific Issues

❌ **Pitfall:** Forgetting to synchronize before timing

```cpp
// WRONG
auto start = std::chrono::now();
my_kernel<<<...>>>();
auto end = std::chrono::now();  // Kernel may still be running!

// CORRECT
auto start = std::chrono::now();
my_kernel<<<...>>>();
cudaDeviceSynchronize();
auto end = std::chrono::now();
```

❌ **Pitfall:** Not checking CUDA errors

```cpp
// WRONG
cudaMalloc(&ptr, size);
cudaMemcpy(...);

// CORRECT
CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaMemcpy(...));
```

❌ **Pitfall:** Incorrect kernel launch configuration

```cpp
// WRONG: May launch too many threads
dim3 blocks(out_w / 16, out_h / 16);  // Integer division!

// CORRECT: Ceiling division
dim3 blocks((out_w + 15) / 16, (out_h + 15) / 16);
```

### 4.2 Memory Management

❌ **Pitfall:** Double-free or use-after-free

```cpp
// WRONG
GPUTensor tensor({...});
cudaFree(tensor.d_data);  // Destructor will free again!

// CORRECT: Let destructor handle it automatically
```

❌ **Pitfall:** Copying large data on every iteration

```cpp
// WRONG: Inefficient
for (int i = 0; i < 1000; ++i) {
    cudaMemcpy(d_data, h_data, large_size, H2D);  // Slow!
}

// BETTER: Copy once, process on GPU
cudaMemcpy(d_data, h_data, large_size, H2D);
for (int i = 0; i < 1000; ++i) {
    gpu_process_kernel<<<...>>>(d_data);
}
```

### 4.3 Numerical Issues

❌ **Pitfall:** Accumulation order differences

- GPU parallelism may sum in different order than CPU
- Result: Small floating-point differences (<1e-5)
- **Solution:** Use tolerance in tests, not exact equality

---

## 5. Profiling Guide

### 5.1 Using Nsight Compute

**Profile a specific kernel:**

```bash
ncu --set full -o profile_conv2d ./bin/train_gpu
```

**View profile:**

```bash
ncu-ui profile_conv2d.ncu-rep
```

**Key metrics to check:**

- **Memory throughput:** % of peak bandwidth used
- **Compute throughput:** % of peak FLOPS used
- **Occupancy:** % of theoretical maximum
- **Warp execution efficiency:** Thread divergence

### 5.2 Using Nsight Systems

**Timeline profiling:**

```bash
nsys profile -o timeline ./bin/train_gpu
nsys-ui timeline.qdrep
```

**Look for:**

- Kernel execution gaps (launch overhead)
- H2D/D2H transfer times
- CPU-GPU synchronization points

### 5.3 Simple Timing

**Add timing to each kernel:**

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
my_kernel<<<...>>>();
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);

printf("Kernel time: %.3f ms\n", ms);
```

---

## 6. Google Colab Notes

### 6.1 Enable GPU Runtime

1. **Runtime → Change runtime type**
2. **Hardware accelerator → GPU**
3. **GPU type → T4** (free tier) or **V100/A100** (Pro)

### 6.2 Check CUDA Installation

```python
!nvidia-smi
!nvcc --version
```

**Expected output:**

```
NVIDIA-SMI 525.x.x       Driver Version: 525.x.x       CUDA Version: 12.0
```

### 6.3 Build on Colab

```python
!git clone https://github.com/YOUR_USERNAME/PP-Final_Project.git
%cd PP-Final_Project

# Build with CUDA support
!mkdir -p build
%cd build
!cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
!make -j4

# Run GPU training
!./bin/train_gpu
```

**Note:** T4 GPU is Compute Capability 7.5, V100 is 7.0, A100 is 8.0

### 6.4 Monitor GPU Usage

```python
# In a separate cell, run periodically
!nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

---

## 7. Deliverables Checklist

- [ ] All CUDA kernels implemented (Conv2D, ReLU, MaxPool, Upsample, MSE)
- [ ] GPU forward pass working
- [ ] GPU backward pass working (gradients computed)
- [ ] GPU training loop completes
- [ ] Correctness verified (GPU matches CPU output)
- [ ] Speedup measured: 5-10× vs CPU
- [ ] No CUDA errors or memory leaks
- [ ] Profiling data collected
- [ ] Code runs on Google Colab with T4 GPU

---

## 8. Next Steps

After Phase 2 completion:

1. **Analyze profiling data:**

   - Which kernel is slowest?
   - What's the memory bandwidth utilization?
   - Where are optimization opportunities?

2. **Identify bottlenecks:**

   - Convolution likely dominates (80-90% of time)
   - Memory-bound or compute-bound?
   - Kernel launch overhead significant?

3. **Prepare for Phase 3:**
   - Read about shared memory tiling
   - Understand memory coalescing
   - Review kernel fusion techniques

---

## 9. Reference Resources

**CUDA Programming Guide:**

- https://docs.nvidia.com/cuda/cuda-c-programming-guide/

**CUDA Best Practices:**

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

**Nsight Compute:**

- https://docs.nvidia.com/nsight-compute/

**Reference Implementation:**

- tbennun/cudnn-training: https://github.com/tbennun/cudnn-training

---

**Well done on completing Phase 2! You now have a working GPU implementation. Phase 3 will focus on making it fast!**
