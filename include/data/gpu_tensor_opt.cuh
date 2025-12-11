// GPU Tensor and Memory Pool for Optimized v1
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Lightweight GPU tensor - device memory only
struct GPUTensorOpt {
    float* d_data;
    int batch, channels, height, width, size;

    GPUTensorOpt() : d_data(nullptr), batch(0), channels(0), height(0), width(0), size(0) {}

    void allocate(int n, int c, int h, int w) {
        batch = n; channels = c; height = h; width = w;
        size = n * c * h * w;
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    }

    void free() {
        if (d_data) { cudaFree(d_data); d_data = nullptr; }
    }

    void zero() { CUDA_CHECK(cudaMemset(d_data, 0, size * sizeof(float))); }
};

// Convolution weights with gradients
struct GPUConvWeightsOpt {
    float* d_weights;
    float* d_bias;
    float* d_grad_w;
    float* d_grad_b;
    int in_c, out_c, kH, kW;
    int weight_size, bias_size;

    GPUConvWeightsOpt(int out_channels, int in_channels, int kernel_h, int kernel_w)
        : out_c(out_channels), in_c(in_channels), kH(kernel_h), kW(kernel_w) {
        weight_size = out_c * in_c * kH * kW;
        bias_size = out_c;
        CUDA_CHECK(cudaMalloc(&d_weights, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_w, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_b, bias_size * sizeof(float)));
    }

    ~GPUConvWeightsOpt() {
        cudaFree(d_weights); cudaFree(d_bias);
        cudaFree(d_grad_w); cudaFree(d_grad_b);
    }

    void initXavier() {
        float* h_w = new float[weight_size];
        float* h_b = new float[bias_size];
        float scale = sqrtf(2.0f / (in_c * kH * kW));
        for (int i = 0; i < weight_size; i++) h_w[i] = scale * ((rand() / (float)RAND_MAX) * 2 - 1);
        for (int i = 0; i < bias_size; i++) h_b[i] = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_weights, h_w, weight_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_b, bias_size * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_w; delete[] h_b;
    }

    void zeroGrad() {
        CUDA_CHECK(cudaMemset(d_grad_w, 0, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_b, 0, bias_size * sizeof(float)));
    }
};

// Memory pool - pre-allocates all buffers once
struct GPUMemoryPool {
    // Forward activations
    GPUTensorOpt act1, act2, act3, act4, act5, act6, act7, act8, output;
    // Backward gradients  
    GPUTensorOpt grad_out, grad8, grad7, grad6, grad5, grad4, grad3, grad2, grad1, grad_in;
    // Pooling indices
    int* pool1_idx;
    int* pool2_idx;
    
    int batch_size;
    bool allocated;

    GPUMemoryPool() : pool1_idx(nullptr), pool2_idx(nullptr), batch_size(0), allocated(false) {}

    void allocate(int bs) {
        if (allocated && bs == batch_size) return;
        if (allocated) free();
        
        batch_size = bs;
        // Forward pass buffers
        act1.allocate(bs, 256, 32, 32);  // Conv1 output
        act2.allocate(bs, 256, 16, 16);  // Pool1 output
        act3.allocate(bs, 128, 16, 16);  // Conv2 output
        act4.allocate(bs, 128, 8, 8);    // Pool2 output (latent)
        act5.allocate(bs, 128, 8, 8);    // Conv3 output
        act6.allocate(bs, 128, 16, 16);  // Upsample1 output
        act7.allocate(bs, 256, 16, 16);  // Conv4 output
        act8.allocate(bs, 256, 32, 32);  // Upsample2 output
        output.allocate(bs, 3, 32, 32);  // Final output

        // Backward pass buffers
        grad_out.allocate(bs, 3, 32, 32);
        grad8.allocate(bs, 256, 32, 32);
        grad7.allocate(bs, 256, 16, 16);
        grad6.allocate(bs, 128, 16, 16);
        grad5.allocate(bs, 128, 8, 8);
        grad4.allocate(bs, 128, 8, 8);
        grad3.allocate(bs, 128, 16, 16);
        grad2.allocate(bs, 256, 16, 16);
        grad1.allocate(bs, 256, 32, 32);
        grad_in.allocate(bs, 3, 32, 32);

        // Pooling indices
        CUDA_CHECK(cudaMalloc(&pool1_idx, bs * 256 * 16 * 16 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&pool2_idx, bs * 128 * 8 * 8 * sizeof(int)));
        
        allocated = true;
    }

    void free() {
        if (!allocated) return;
        act1.free(); act2.free(); act3.free(); act4.free(); act5.free();
        act6.free(); act7.free(); act8.free(); output.free();
        grad_out.free(); grad8.free(); grad7.free(); grad6.free(); grad5.free();
        grad4.free(); grad3.free(); grad2.free(); grad1.free(); grad_in.free();
        if (pool1_idx) cudaFree(pool1_idx);
        if (pool2_idx) cudaFree(pool2_idx);
        allocated = false;
    }

    ~GPUMemoryPool() { free(); }
};
