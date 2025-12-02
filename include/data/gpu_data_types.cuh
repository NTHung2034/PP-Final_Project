#ifndef GPU_DATA_TYPES_CUH
#define GPU_DATA_TYPES_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Error checking macro
#define CUDA_CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

// GPU Tensor Structure
// Stores data in device memory with metadata
// Uses pinned host memory for faster H2D/D2H transfers

struct GPUTensor {
    float* d_data;      // Device pointer (in global memory)
    float* h_data;      // Host pointer (pinned memory for fast transfer)
    
    int batch;          // Batch size (N)
    int channels;       // Number of channels (C)
    int height;         // Height (H)
    int width;          // Width (W)
    
    size_t size;        // Total number of elements = N*C*H*W
    size_t bytes;       // Total bytes = size * sizeof(float)
    
    bool owns_memory;   // Whether this tensor owns its memory
    
    // Constructor - allocates both device and pinned host memory
    // Using pinned memory (cudaMallocHost) for faster transfers via DMA
    GPUTensor(int b, int c, int h, int w, bool device_only=false) 
        : batch(b), channels(c), height(h), width(w), owns_memory(true), h_data(nullptr){
        
        size = b * c * h * w;
        bytes = size * sizeof(float);
        
        // Allocate device memory (global memory)
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        
        // Allocate pinned host memory for faster H2D/D2H transfers
        // Pinned memory allows DMA and async copies
        if (!device_only) {
            CUDA_CHECK(cudaMallocHost(&h_data, bytes));
        } 
    }
    
    ~GPUTensor() {
        if (owns_memory) {
            if (d_data) CUDA_CHECK(cudaFree(d_data));
            if (h_data) CUDA_CHECK(cudaFreeHost(h_data));
        }
    }
    
     // Copy data from host to device (asynchronous with stream)
     // Takes advantage of pinned memory for faster transfer
    void copyToDevice(cudaStream_t stream = 0) {
        if (h_data && d_data) {
            CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream));
        }
    }
    
    // Copy data from device to host (asynchronous with stream)
    void copyToHost(cudaStream_t stream = 0) {
        if (h_data && d_data) {
            CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost, stream));
        }
    }
    
    //  Zero out device memory
    void zero() {
        if (d_data) {
            CUDA_CHECK(cudaMemset(d_data, 0, bytes));
        }
    }
    
    // Get flattened index for 4D tensor
    // Memory layout: NCHW (batch, channel, height, width)
    __host__ __device__ inline int index(int n, int c, int h, int w) const {
        return ((n * channels + c) * height + h) * width + w;
    }
};

// GPU Weight Structure for Convolutional Layers
// Stores weights and biases with optimized memory access
struct GPUConvWeights {
    float* d_weights;   // Device weights: [out_channels, in_channels, kH, kW]
    float* d_bias;      // Device bias: [out_channels]
    float* d_grad_w;    // Gradient w.r.t. weights
    float* d_grad_b;    // Gradient w.r.t. bias
    
    int out_channels;
    int in_channels;
    int kernel_h;
    int kernel_w;
    
    size_t weight_size;
    size_t bias_size;
    
    GPUConvWeights(int out_c, int in_c, int kh, int kw)
        : out_channels(out_c), in_channels(in_c), kernel_h(kh), kernel_w(kw) {
        
        weight_size = out_c * in_c * kh * kw;
        bias_size = out_c;
        
        // Allocate device memory for weights and gradients
        CUDA_CHECK(cudaMalloc(&d_weights, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_w, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_b, bias_size * sizeof(float)));
        
        // Initialize gradients to zero
        CUDA_CHECK(cudaMemset(d_grad_w, 0, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_b, 0, bias_size * sizeof(float)));
    }
    
    ~GPUConvWeights() {
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_bias));
        CUDA_CHECK(cudaFree(d_grad_w));
        CUDA_CHECK(cudaFree(d_grad_b));
    }
    
    //  Initialize weights using Xavier initialization on CPU then transfer
    void initializeXavier() {
        size_t w_bytes = weight_size * sizeof(float);
        size_t b_bytes = bias_size * sizeof(float);
        
        // Allocate temporary host memory
        float* h_weights = new float[weight_size];
        float* h_bias = new float[bias_size];
        
        // Xavier initialization: scale = sqrt(2.0 / (in + out))
        float scale = sqrtf(2.0f / (in_channels * kernel_h * kernel_w + out_channels));
        
        for (size_t i = 0; i < weight_size; i++) {
            h_weights[i] = scale * ((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
        }
        
        for (size_t i = 0; i < bias_size; i++) {
            h_bias[i] = 0.0f;  // Initialize bias to zero
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights, w_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias, b_bytes, cudaMemcpyHostToDevice));
        
        delete[] h_weights;
        delete[] h_bias;
    }
};

#endif // GPU_DATA_TYPES_H