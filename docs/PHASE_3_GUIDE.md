# Phase 3: Advanced GPU Optimization Guide

**Duration:** December 5-7, 2025 (3 days)  
**Objective:** Apply systematic optimizations to achieve >50× speedup  
**Prerequisite:** Phase 2 complete, profiling data collected

---

## Table of Contents

1. [Overview](#1-overview)
2. [Optimization Version 1: Shared Memory Tiling](#2-optimization-version-1-shared-memory-tiling)
3. [Optimization Version 2: Kernel Fusion](#3-optimization-version-2-kernel-fusion)
4. [Optimization Version 3: Multi-Stream Pipelining](#4-optimization-version-3-multi-stream-pipelining)
5. [Testing and Verification](#5-testing-and-verification)
6. [Performance Analysis](#6-performance-analysis)
7. [Google Colab Notes](#7-google-colab-notes)

---

## 1. Overview

### 1.1 Goals

By the end of Phase 3, you will have:

- ✅ Shared memory optimization implemented (20-40× cumulative speedup)
- ✅ Kernel fusion implemented (40-60× cumulative speedup)
- ✅ Optional: Multi-stream pipelining (60-80× cumulative speedup)
- ✅ Complete performance analysis with profiling data
- ✅ Training time <1 minute for 20 epochs

### 1.2 Optimization Priority

Based on profiling from Phase 2, we focus on:

**1. Shared Memory Tiling (Highest Impact)**

- Convolution is 80-90% of runtime
- Naive version: 100+ GB/s memory traffic
- Optimization: Reduce by 4-10× using shared memory

**2. Kernel Fusion (Medium Impact)**

- 10+ kernel launches per forward pass
- Launch overhead: ~5-10 μs each
- Optimization: Fuse Conv+ReLU, reduce launches by 50%

**3. Multi-Stream Pipelining (Lower Impact)**

- CPU-GPU transfer time ~10-20ms per batch
- Optimization: Overlap with compute

### 1.3 Expected Performance Progression

| Version             | Training Time/Epoch | Speedup vs CPU | Cumulative Speedup |
| ------------------- | ------------------- | -------------- | ------------------ |
| CPU Baseline        | 18 min              | 1×             | 1×                 |
| GPU Naive (Phase 2) | 2.5 min             | 7.2×           | 7.2×               |
| **+ Shared Memory** | 45 sec              | 24×            | 24×                |
| **+ Kernel Fusion** | 28 sec              | 38.5×          | 38.5×              |
| **+ Multi-Stream**  | 20 sec              | 54×            | 54×                |

---

## 2. Optimization Version 1: Shared Memory Tiling

**Duration:** December 5-6 (1.5 days)

### 2.1 Theory

**Problem with Naive Convolution:**

- Each output pixel requires `in_channels × kernel_size²` reads
- Same input pixels read multiple times by different threads
- **Example:** For 3×3 conv, each input pixel reused 9 times

**Shared Memory Solution:**

- Load input tile into shared memory once
- All threads in block reuse cached data
- **Speedup:** 4-8× for convolution kernels

**Reference:** CUDA C Programming Guide - Shared Memory
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory

### 2.2 Implementation

#### Step 2.2.1: Tiled Convolution Kernel

**File:** `src/cuda/kernels/conv2d_tiled_kernel.cu`

```cpp
#include "cuda/kernels/conv2d_kernel.cuh"

// Tile size for shared memory (tunable)
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

__global__ void conv2d_tiled_kernel(
    const float* __restrict__ input,    // [batch, in_c, in_h, in_w]
    const float* __restrict__ weights,  // [out_c, in_c, k, k]
    const float* __restrict__ bias,     // [out_c]
    float* __restrict__ output,         // [batch, out_c, out_h, out_w]
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k_size, int stride, int pad
) {
    // Shared memory for input tile
    // Size: (TILE_HEIGHT + k_size - 1) × (TILE_WIDTH + k_size - 1) × in_c
    // For 3×3 kernel: (18 × 18) × in_c floats
    extern __shared__ float shared_input[];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output indices
    int ow = blockIdx.x * TILE_WIDTH + tx;
    int oh = blockIdx.y * TILE_HEIGHT + ty;
    int oc = blockIdx.z % out_c;
    int n = blockIdx.z / out_c;

    // Compute shared memory dimensions
    int shared_h = TILE_HEIGHT + k_size - 1;
    int shared_w = TILE_WIDTH + k_size - 1;

    float sum = 0.0f;

    // Process one input channel at a time (to limit shared memory usage)
    for (int ic = 0; ic < in_c; ++ic) {
        // Collaboratively load input tile into shared memory
        // Each thread loads multiple elements if necessary
        for (int i = ty; i < shared_h; i += TILE_HEIGHT) {
            for (int j = tx; j < shared_w; j += TILE_WIDTH) {
                int ih = (blockIdx.y * TILE_HEIGHT) * stride - pad + i;
                int iw = (blockIdx.x * TILE_WIDTH) * stride - pad + j;

                // Check bounds and load (zero padding)
                float val = 0.0f;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w && n < batch) {
                    int in_idx = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                    val = input[in_idx];
                }

                int shared_idx = i * shared_w + j;
                shared_input[shared_idx] = val;
            }
        }
        __syncthreads();

        // Compute convolution using shared memory
        if (ow < out_w && oh < out_h && n < batch) {
            for (int kh = 0; kh < k_size; ++kh) {
                for (int kw = 0; kw < k_size; ++kw) {
                    int shared_row = ty * stride + kh;
                    int shared_col = tx * stride + kw;
                    int shared_idx = shared_row * shared_w + shared_col;

                    int w_idx = ((oc * in_c + ic) * k_size + kh) * k_size + kw;
                    sum += shared_input[shared_idx] * weights[w_idx];
                }
            }
        }
        __syncthreads();
    }

    // Add bias and write output
    if (ow < out_w && oh < out_h && n < batch) {
        sum += bias[oc];
        int out_idx = ((n * out_c + oc) * out_h + oh) * out_w + ow;
        output[out_idx] = sum;
    }
}

void launch_conv2d_tiled(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k_size, int stride, int pad
) {
    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocks(
        (out_w + TILE_WIDTH - 1) / TILE_WIDTH,
        (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch * out_c
    );

    // Calculate shared memory size
    int shared_h = TILE_HEIGHT + k_size - 1;
    int shared_w = TILE_WIDTH + k_size - 1;
    size_t shared_mem_size = shared_h * shared_w * sizeof(float);

    conv2d_tiled_kernel<<<blocks, threads, shared_mem_size>>>(
        d_input, d_weights, d_bias, d_output,
        batch, in_c, in_h, in_w,
        out_c, out_h, out_w,
        k_size, stride, pad
    );

    CUDA_CHECK(cudaGetLastError());
}
```

**Key Implementation Details:**

1. **Shared memory allocation:** `extern __shared__ float shared_input[]`

   - Size calculated at kernel launch: `(TILE_H + k - 1) × (TILE_W + k - 1)`
   - For 16×16 tile + 3×3 kernel: 18×18 = 324 floats = 1.3 KB

2. **Cooperative loading:**

   - All threads collaborate to load tile
   - Each thread may load multiple elements

3. **Synchronization:**

   - `__syncthreads()` after loading tile
   - `__syncthreads()` before next channel (if reusing shared memory)

4. **Memory access pattern:**
   - Coalesced global memory reads (consecutive threads read consecutive addresses)
   - Fast shared memory reads (low latency, high bandwidth)

#### Step 2.2.2: Update Autoencoder to Use Tiled Convolution

**File:** `src/models/autoencoder_gpu.cu`

```cpp
void AutoencoderGPU::forward_v1(const float* h_input, int batch_size) {
    // ... same as before, but replace conv2d launches ...

    // Conv1: Use tiled version
    launch_conv2d_tiled(
        input_->d_data, d_conv1_w_, d_conv1_b_, conv1_out_->d_data,
        batch_size, 3, 32, 32,
        256, 32, 32,
        3, 1, 1
    );

    // ... same for all other conv layers ...
}
```

### 2.3 Testing Shared Memory Version

**Test correctness:**

```cpp
void test_tiled_conv() {
    // Create test input
    std::vector<float> h_input(1 * 3 * 32 * 32);
    std::fill(h_input.begin(), h_input.end(), 0.5f);

    // Run naive version
    AutoencoderGPU model_naive;
    model_naive.forward(h_input.data(), 1);
    std::vector<float> output_naive(1 * 3 * 32 * 32);
    model_naive.get_output(output_naive.data());

    // Run tiled version
    AutoencoderGPU model_tiled;
    model_tiled.forward_v1(h_input.data(), 1);
    std::vector<float> output_tiled(1 * 3 * 32 * 32);
    model_tiled.get_output(output_tiled.data());

    // Compare
    float max_diff = 0.0f;
    for (size_t i = 0; i < output_naive.size(); ++i) {
        float diff = std::abs(output_naive[i] - output_tiled[i]);
        max_diff = std::max(max_diff, diff);
    }

    printf("Max diff (naive vs tiled): %e\n", max_diff);
    assert(max_diff < 1e-4);
}
```

**Benchmark:**

```bash
./bin/benchmark_gpu --version naive
./bin/benchmark_gpu --version tiled

# Compare results
```

**Expected improvement:** 3-5× speedup for convolution kernels

### 2.4 Profiling with Nsight Compute

```bash
ncu --set full --kernel-name conv2d_tiled_kernel -o profile_tiled ./bin/train_gpu

# View metrics:
# - Memory Bandwidth: Should be 200-400 GB/s (higher than naive)
# - Shared Memory Efficiency: >80%
# - Occupancy: 50-75% (limited by shared memory usage)
```

---

## 3. Optimization Version 2: Kernel Fusion

**Duration:** December 6-7 (1 day)

### 3.1 Theory

**Problem:**

- Each kernel launch has overhead (~5-10 μs)
- Forward pass has 10+ kernel launches
- Total overhead: ~100 μs per batch

**Solution: Fuse adjacent operations**

- Conv + ReLU → Single kernel
- Conv + ReLU + Bias → Single kernel
- Reduce kernel launches by 50%

**Benefits:**

- Fewer kernel launches
- Reduced global memory traffic (intermediate results stay in registers)
- Better instruction-level parallelism

### 3.2 Implementation

#### Step 3.2.1: Fused Conv+ReLU+Bias Kernel

**File:** `src/cuda/kernels/conv_relu_kernel.cu`

```cpp
__global__ void conv2d_relu_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,  // Output already has ReLU applied
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k_size, int stride, int pad
) {
    extern __shared__ float shared_input[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow = blockIdx.x * TILE_WIDTH + tx;
    int oh = blockIdx.y * TILE_HEIGHT + ty;
    int oc = blockIdx.z % out_c;
    int n = blockIdx.z / out_c;

    int shared_h = TILE_HEIGHT + k_size - 1;
    int shared_w = TILE_WIDTH + k_size - 1;

    float sum = 0.0f;

    // Convolution (same as tiled version)
    for (int ic = 0; ic < in_c; ++ic) {
        // Load tile
        for (int i = ty; i < shared_h; i += TILE_HEIGHT) {
            for (int j = tx; j < shared_w; j += TILE_WIDTH) {
                int ih = (blockIdx.y * TILE_HEIGHT) * stride - pad + i;
                int iw = (blockIdx.x * TILE_WIDTH) * stride - pad + j;

                float val = 0.0f;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w && n < batch) {
                    int in_idx = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                    val = input[in_idx];
                }
                shared_input[i * shared_w + j] = val;
            }
        }
        __syncthreads();

        // Compute
        if (ow < out_w && oh < out_h && n < batch) {
            for (int kh = 0; kh < k_size; ++kh) {
                for (int kw = 0; kw < k_size; ++kw) {
                    int shared_idx = (ty * stride + kh) * shared_w + (tx * stride + kw);
                    int w_idx = ((oc * in_c + ic) * k_size + kh) * k_size + kw;
                    sum += shared_input[shared_idx] * weights[w_idx];
                }
            }
        }
        __syncthreads();
    }

    // Fused: Add bias + ReLU in one operation
    if (ow < out_w && oh < out_h && n < batch) {
        sum += bias[oc];
        sum = fmaxf(0.0f, sum);  // ReLU

        int out_idx = ((n * out_c + oc) * out_h + oh) * out_w + ow;
        output[out_idx] = sum;
    }
}

void launch_conv2d_relu_fused(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k_size, int stride, int pad
) {
    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocks(
        (out_w + TILE_WIDTH - 1) / TILE_WIDTH,
        (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch * out_c
    );

    int shared_h = TILE_HEIGHT + k_size - 1;
    int shared_w = TILE_WIDTH + k_size - 1;
    size_t shared_mem_size = shared_h * shared_w * sizeof(float);

    conv2d_relu_tiled_kernel<<<blocks, threads, shared_mem_size>>>(
        d_input, d_weights, d_bias, d_output,
        batch, in_c, in_h, in_w,
        out_c, out_h, out_w,
        k_size, stride, pad
    );

    CUDA_CHECK(cudaGetLastError());
}
```

#### Step 3.2.2: Update Forward Pass with Fused Kernels

```cpp
void AutoencoderGPU::forward_v2(const float* h_input, int batch_size) {
    input_->from_host(h_input);

    // ENCODER
    // Conv1 + ReLU (fused)
    launch_conv2d_relu_fused(
        input_->d_data, d_conv1_w_, d_conv1_b_, relu1_out_->d_data,
        batch_size, 3, 32, 32, 256, 32, 32, 3, 1, 1
    );

    // MaxPool1 (separate - hard to fuse)
    launch_maxpool_forward(
        relu1_out_->d_data, pool1_out_->d_data, d_maxpool1_indices_,
        batch_size, 256, 32, 32, 16, 16, 2
    );

    // Conv2 + ReLU (fused)
    launch_conv2d_relu_fused(
        pool1_out_->d_data, d_conv2_w_, d_conv2_b_, relu2_out_->d_data,
        batch_size, 256, 16, 16, 128, 16, 16, 3, 1, 1
    );

    // MaxPool2
    launch_maxpool_forward(
        relu2_out_->d_data, latent_->d_data, d_maxpool2_indices_,
        batch_size, 128, 16, 16, 8, 8, 2
    );

    // DECODER (same pattern)
    // Conv3 + ReLU (fused)
    launch_conv2d_relu_fused(
        latent_->d_data, d_conv3_w_, d_conv3_b_, relu3_out_->d_data,
        batch_size, 128, 8, 8, 128, 8, 8, 3, 1, 1
    );

    // Upsample
    launch_upsample_forward(
        relu3_out_->d_data, up1_out_->d_data,
        batch_size, 128, 8, 8, 16, 16, 2
    );

    // Conv4 + ReLU (fused)
    launch_conv2d_relu_fused(
        up1_out_->d_data, d_conv4_w_, d_conv4_b_, relu4_out_->d_data,
        batch_size, 128, 16, 16, 256, 16, 16, 3, 1, 1
    );

    // Upsample
    launch_upsample_forward(
        relu4_out_->d_data, up2_out_->d_data,
        batch_size, 256, 16, 16, 32, 32, 2
    );

    // Conv5 (no ReLU at output)
    launch_conv2d_tiled(
        up2_out_->d_data, d_conv5_w_, d_conv5_b_, output_->d_data,
        batch_size, 256, 32, 32, 3, 32, 32, 3, 1, 1
    );
}
```

**Benefits:**

- Kernel launches: 15 → 8 (47% reduction)
- Memory traffic: Intermediate relu outputs eliminated
- Expected speedup: 1.5-2× over tiled version

### 3.3 Testing Kernel Fusion

**Correctness test:**

```cpp
void test_fused_kernels() {
    std::vector<float> h_input(64 * 3 * 32 * 32);
    // ... initialize ...

    // Version 1 (tiled, separate kernels)
    AutoencoderGPU model_v1;
    model_v1.forward_v1(h_input.data(), 64);
    std::vector<float> output_v1(64 * 3 * 32 * 32);
    model_v1.get_output(output_v1.data());

    // Version 2 (fused kernels)
    AutoencoderGPU model_v2;
    model_v2.forward_v2(h_input.data(), 64);
    std::vector<float> output_v2(64 * 3 * 32 * 32);
    model_v2.get_output(output_v2.data());

    // Compare
    float max_diff = 0.0f;
    for (size_t i = 0; i < output_v1.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(output_v1[i] - output_v2[i]));
    }

    printf("Max diff (v1 vs v2): %e\n", max_diff);
    assert(max_diff < 1e-5);
}
```

---

## 4. Optimization Version 3: Multi-Stream Pipelining (Optional)

**Duration:** December 7 (0.5 days)

### 4.1 Theory

**Current bottleneck:**

- H2D transfer: ~10ms (copy batch to GPU)
- Compute: ~20ms (forward + backward)
- D2H transfer: ~5ms (copy gradients back)
- **Total:** 35ms (sequential)

**Pipelining solution:**

- Use 3 CUDA streams
- Stream 1: H2D copy for batch N+1
- Stream 2: Compute for batch N
- Stream 3: D2H copy for batch N-1
- **Overlap:** 20-30% faster

### 4.2 Implementation

**File:** `src/train_gpu_streams.cu`

```cpp
#include "models/autoencoder_gpu.cuh"
#include <cuda_runtime.h>

void train_with_streams() {
    const int num_streams = 3;
    const int batch_size = 64;

    // Create CUDA streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Pinned host memory for faster transfers
    float *h_batch[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaMallocHost(&h_batch[i], batch_size * 3 * 32 * 32 * sizeof(float));
    }

    // GPU buffers (one per stream)
    GPUTensor* d_batch[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        d_batch[i] = new GPUTensor({batch_size, 3, 32, 32});
    }

    AutoencoderGPU model;
    CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
    dataset.load_data();

    int batches_per_epoch = dataset.size() / batch_size;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        dataset.shuffle();
        dataset.reset();

        for (int batch = 0; batch < batches_per_epoch; ++batch) {
            int stream_id = batch % num_streams;

            // Get batch data
            auto cpu_batch = dataset.get_batch(batch_size);
            memcpy(h_batch[stream_id], cpu_batch.data->data(),
                   batch_size * 3 * 32 * 32 * sizeof(float));

            // Async H2D transfer
            cudaMemcpyAsync(d_batch[stream_id]->d_data, h_batch[stream_id],
                            batch_size * 3 * 32 * 32 * sizeof(float),
                            cudaMemcpyHostToDevice, streams[stream_id]);

            // Launch compute on stream
            model.forward_async(d_batch[stream_id]->d_data, batch_size,
                                streams[stream_id]);

            // Backward pass (async)
            model.backward_async(d_batch[stream_id]->d_data, 0.001f,
                                 batch_size, streams[stream_id]);
        }

        // Synchronize all streams at end of epoch
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
    }

    // Cleanup
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaFreeHost(h_batch[i]);
        delete d_batch[i];
    }
}
```

**Key changes needed in autoencoder:**

```cpp
// Add stream parameter to kernel launches
void AutoencoderGPU::forward_async(const float* d_input, int batch_size,
                                    cudaStream_t stream) {
    // Replace all kernel launches with stream versions
    conv2d_relu_tiled_kernel<<<blocks, threads, shared_mem, stream>>>(...);
    maxpool_forward_kernel<<<blocks, threads, 0, stream>>>(...);
    // etc.
}
```

### 4.3 Testing Streams

**Verify correctness:**

```cpp
// Outputs should be identical to non-stream version
// Just verify training converges normally
```

**Profile with Nsight Systems:**

```bash
nsys profile -o timeline_streams ./bin/train_gpu_streams
nsys-ui timeline_streams.qdrep

# Look for:
# - Overlapping H2D, compute, D2H regions
# - GPU utilization >90% (vs ~70% without streams)
```

---

## 5. Testing and Verification

### 5.1 Performance Comparison Table

**Create benchmark script:** `scripts/benchmark_all_versions.sh`

```bash
#!/bin/bash

echo "Version,Epoch_Time(s),Speedup"

# CPU baseline
time_cpu=$(./bin/train_cpu --epochs 1 | grep "seconds" | awk '{print $5}')
echo "CPU,$time_cpu,1.0"

# GPU naive
time_naive=$(./bin/train_gpu_naive --epochs 1 | grep "seconds" | awk '{print $5}')
speedup=$(echo "scale=2; $time_cpu / $time_naive" | bc)
echo "GPU_Naive,$time_naive,$speedup"

# GPU tiled
time_tiled=$(./bin/train_gpu_v1 --epochs 1 | grep "seconds" | awk '{print $5}')
speedup=$(echo "scale=2; $time_cpu / $time_tiled" | bc)
echo "GPU_Tiled,$time_tiled,$speedup"

# GPU fused
time_fused=$(./bin/train_gpu_v2 --epochs 1 | grep "seconds" | awk '{print $5}')
speedup=$(echo "scale=2; $time_cpu / $time_fused" | bc)
echo "GPU_Fused,$time_fused,$speedup"

# GPU streams (optional)
time_streams=$(./bin/train_gpu_v3 --epochs 1 | grep "seconds" | awk '{print $5}')
speedup=$(echo "scale=2; $time_cpu / $time_streams" | bc)
echo "GPU_Streams,$time_streams,$speedup"
```

**Run:**

```bash
bash scripts/benchmark_all_versions.sh > results/performance_comparison.csv
```

### 5.2 Correctness Tests

**All versions should produce identical results:**

```cpp
void test_all_versions_match() {
    std::vector<float> input(64 * 3 * 32 * 32);
    for (auto& x : input) x = (rand() % 1000) / 1000.0f;

    AutoencoderGPU model_naive, model_v1, model_v2;

    // Load same weights
    model_naive.load_weights("test_weights.bin");
    model_v1.load_weights("test_weights.bin");
    model_v2.load_weights("test_weights.bin");

    // Run forward pass
    model_naive.forward(input.data(), 64);
    model_v1.forward_v1(input.data(), 64);
    model_v2.forward_v2(input.data(), 64);

    // Get outputs
    std::vector<float> out_naive(64 * 3 * 32 * 32);
    std::vector<float> out_v1(64 * 3 * 32 * 32);
    std::vector<float> out_v2(64 * 3 * 32 * 32);

    model_naive.get_output(out_naive.data());
    model_v1.get_output(out_v1.data());
    model_v2.get_output(out_v2.data());

    // Compare
    float max_diff_v1 = 0.0f, max_diff_v2 = 0.0f;
    for (size_t i = 0; i < out_naive.size(); ++i) {
        max_diff_v1 = std::max(max_diff_v1,
                               std::abs(out_naive[i] - out_v1[i]));
        max_diff_v2 = std::max(max_diff_v2,
                               std::abs(out_naive[i] - out_v2[i]));
    }

    printf("Max diff (naive vs v1): %e\n", max_diff_v1);
    printf("Max diff (naive vs v2): %e\n", max_diff_v2);

    assert(max_diff_v1 < 1e-4);
    assert(max_diff_v2 < 1e-4);

    printf("✓ All versions produce identical results!\n");
}
```

---

## 6. Performance Analysis

### 6.1 Profiling Data to Collect

For each optimization version, record:

**Timing:**

- Training time per epoch
- Forward pass time
- Backward pass time
- Data transfer time

**GPU Metrics (from Nsight Compute):**

- Memory bandwidth utilization (%)
- Compute throughput (GFLOPS)
- Occupancy (%)
- Shared memory bank conflicts
- Warp execution efficiency (%)

**Memory:**

- GPU memory used
- Peak memory usage
- Shared memory per block

### 6.2 Expected Results Table

| Metric              | Naive    | Tiled    | Fused    | Streams  |
| ------------------- | -------- | -------- | -------- | -------- |
| **Epoch Time**      | 150s     | 45s      | 28s      | 20s      |
| **Speedup vs CPU**  | 7×       | 24×      | 38×      | 54×      |
| **Memory BW**       | 150 GB/s | 300 GB/s | 350 GB/s | 380 GB/s |
| **Occupancy**       | 75%      | 60%      | 65%      | 70%      |
| **GPU Memory**      | 2.5 GB   | 2.6 GB   | 2.6 GB   | 3.2 GB   |
| **Kernel Launches** | 15/batch | 15/batch | 8/batch  | 8/batch  |

### 6.3 Analysis Questions

For your report, answer:

1. **Which optimization had the biggest impact? Why?**

   - Likely: Shared memory (convolution dominates)

2. **Did any optimization not work as expected?**

   - Streams may give <20% improvement if compute-bound

3. **What's the current bottleneck?**

   - Use profiler to identify (likely still memory bandwidth)

4. **What further optimizations could help?**
   - Tensor cores (FP16), Winograd convolution, etc.

---

## 7. Google Colab Notes

### 7.1 Checking GPU Capabilities

```python
!nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# T4: Compute Capability 7.5, 16 GB
# Shared memory per block: 48 KB (default), 96 KB (max)
```

### 7.2 Compiling with Correct Architecture

```bash
# For T4 (Compute Capability 7.5)
nvcc -arch=sm_75 -O3 ...

# For V100 (Compute Capability 7.0)
nvcc -arch=sm_70 -O3 ...

# For A100 (Compute Capability 8.0)
nvcc -arch=sm_80 -O3 ...
```

### 7.3 Running Profiler on Colab

```python
# Install Nsight Compute (if not available)
!apt-get install -y nsight-compute

# Profile
!ncu --set full -o profile ./bin/train_gpu_v2

# Download profile file
from google.colab import files
files.download('profile.ncu-rep')

# Open on local machine with ncu-ui
```

---

## 8. Common Pitfalls

### 8.1 Shared Memory Issues

❌ **Pitfall:** Shared memory bank conflicts

```cpp
// BAD: Threads access same bank
shared_data[threadIdx.x * stride];  // If stride causes conflicts

// GOOD: Pad shared memory to avoid conflicts
__shared__ float data[TILE_SIZE + 1];  // +1 padding
```

❌ **Pitfall:** Insufficient shared memory

```cpp
// Error: "too many resources requested for launch"
// Solution: Reduce TILE_SIZE or use multiple passes
```

### 8.2 Kernel Fusion Issues

❌ **Pitfall:** Register pressure

- Fused kernels use more registers
- Can reduce occupancy
- **Solution:** Use `-maxrregcount=64` flag if needed

❌ **Pitfall:** Incorrect synchronization

```cpp
// BAD: Missing __syncthreads() between phases
load_to_shared();
compute();  // May use stale data!

// GOOD:
load_to_shared();
__syncthreads();
compute();
```

### 8.3 Stream Issues

❌ **Pitfall:** Using non-pinned memory

```cpp
// SLOW: Pageable memory
float* h_data = new float[size];
cudaMemcpyAsync(...);  // Falls back to sync!

// FAST: Pinned memory
float* h_data;
cudaMallocHost(&h_data, size);
cudaMemcpyAsync(...);  // Actually async
```

---

## 9. Deliverables Checklist

- [ ] Shared memory optimization implemented
- [ ] Kernel fusion implemented
- [ ] Performance comparison table created
- [ ] All versions tested for correctness
- [ ] Profiling data collected (Nsight Compute reports)
- [ ] > 50× cumulative speedup achieved
- [ ] Training completes in <1 minute (20 epochs)
- [ ] Code documented with optimization explanations

---

## 10. References

**CUDA Optimization:**

- CUDA C Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Shared Memory Tutorial: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/

**Convolution Optimization:**

- Winograd Convolution: https://arxiv.org/abs/1509.09308
- Im2Col + GEMM: https://github.com/BVLC/caffe/wiki/Convolution-in-Caffe:-a-memo

**Profiling:**

- Nsight Compute Guide: https://docs.nvidia.com/nsight-compute/
- Nsight Systems Guide: https://docs.nvidia.com/nsight-systems/

---

**Congratulations on completing Phase 3! You now have a highly optimized GPU implementation. Next: SVM integration and final results!**
