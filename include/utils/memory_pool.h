// #pragma once
// #include <vector>
// #include <memory>
// #include "data/data_types.h"

// class MemoryPool {
// public:
//     // Pre-allocate buffers for common tensor shapes
//     static void initialize(const std::vector<std::vector<int>>& shapes);
    
//     // Get a pre-allocated tensor (no allocation)
//     static Tensor acquire_tensor(const std::vector<int>& shape);
    
//     // Return tensor to pool (no deallocation)
//     static void release_tensor(Tensor& tensor);
    
//     // GPU version for Phase 2
//     static void* acquire_gpu_buffer(size_t bytes);
//     static void release_gpu_buffer(void* ptr);
    
// private:
//     static std::vector<std::unique_ptr<AlignedBuffer<float>>> cpu_pool_;
//     static std::vector<void*> gpu_pool_;  // For Phase 2
//     static std::vector<size_t> gpu_pool_sizes_;
// };