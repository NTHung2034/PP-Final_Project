#pragma once
#include "data/gpu_tensor_opt.cuh"

// V2 Memory Pool - Adds multi-stream support for overlapping transfers and compute
struct GPUMemoryPoolV2 {
    // Forward activations (same as V1)
    GPUTensorOpt act1, act2, act3, act4, act5, act6, act7, act8, output;
    // Backward gradients
    GPUTensorOpt grad_out, grad8, grad7, grad6, grad5, grad4, grad3, grad2, grad1, grad_in;
    // Pooling indices
    int* pool1_idx = nullptr;
    int* pool2_idx = nullptr;
    
    // V2: Pre-allocated loss buffer (avoids cudaMalloc per batch)
    float* d_loss = nullptr;
    
    // V2: Multi-stream pipeline
    cudaStream_t compute_stream = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaEvent_t transfer_done = nullptr;
    
    int batch_size = 0;
    bool allocated = false;
    bool streams_ready = false;

    void init_streams() {
        if (streams_ready) return;
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&transfer_stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&transfer_done, cudaEventDisableTiming));
        streams_ready = true;
    }

    void allocate(int bs) {
        if (allocated && bs == batch_size) return;
        if (allocated) free();
        batch_size = bs;
        
        // Forward buffers
        act1.allocate(bs, 256, 32, 32);
        act2.allocate(bs, 256, 16, 16);
        act3.allocate(bs, 128, 16, 16);
        act4.allocate(bs, 128, 8, 8);
        act5.allocate(bs, 128, 8, 8);
        act6.allocate(bs, 128, 16, 16);
        act7.allocate(bs, 256, 16, 16);
        act8.allocate(bs, 256, 32, 32);
        output.allocate(bs, 3, 32, 32);
        
        // Backward buffers
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
        
        // Pre-allocated loss buffer
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
        
        init_streams();
        allocated = true;
    }

    void free() {
        if (!allocated) return;
        // Free streams
        if (streams_ready) {
            cudaStreamDestroy(compute_stream);
            cudaStreamDestroy(transfer_stream);
            cudaEventDestroy(transfer_done);
            compute_stream = transfer_stream = nullptr;
            transfer_done = nullptr;
            streams_ready = false;
        }
        // Free tensors
        act1.free(); act2.free(); act3.free(); act4.free(); act5.free();
        act6.free(); act7.free(); act8.free(); output.free();
        grad_out.free(); grad8.free(); grad7.free(); grad6.free(); grad5.free();
        grad4.free(); grad3.free(); grad2.free(); grad1.free(); grad_in.free();
        if (pool1_idx) { cudaFree(pool1_idx); pool1_idx = nullptr; }
        if (pool2_idx) { cudaFree(pool2_idx); pool2_idx = nullptr; }
        if (d_loss) { cudaFree(d_loss); d_loss = nullptr; }
        allocated = false;
    }

    // Async H2D transfer on transfer_stream
    void async_input_transfer(float* d_dst, const float* h_src, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, transfer_stream));
        CUDA_CHECK(cudaEventRecord(transfer_done, transfer_stream));
    }

    // Make compute_stream wait for transfer to complete
    void sync_before_compute() {
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, transfer_done, 0));
    }
    
    // Reset loss accumulator at epoch start
    void reset_loss() {
        CUDA_CHECK(cudaMemsetAsync(d_loss, 0, sizeof(float), compute_stream));
    }
    
    // Get accumulated loss (syncs stream)
    float get_loss(int num_batches) {
        float loss;
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        return loss / num_batches;
    }

    ~GPUMemoryPoolV2() { free(); }
};
