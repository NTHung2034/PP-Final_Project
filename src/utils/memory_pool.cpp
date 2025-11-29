#include "utils/memory_pool.h"
#include <stdexcept>

// Static member initialization
std::vector<std::unique_ptr<AlignedBuffer<float>>> MemoryPool::cpu_pool_;
std::vector<void *> MemoryPool::gpu_pool_;
std::vector<size_t> MemoryPool::gpu_pool_sizes_;

void MemoryPool::initialize(const std::vector<std::vector<int>> &shapes)
{
    // Phase 1: Simple CPU memory pool
    // Pre-allocate buffers for common tensor shapes
    cpu_pool_.clear();

    for (const auto &shape : shapes)
    {
        size_t size = 1;
        for (int dim : shape)
        {
            size *= dim;
        }
        cpu_pool_.emplace_back(new AlignedBuffer<float>(size));
    }
}

Tensor MemoryPool::acquire_tensor(const std::vector<int> &shape)
{
    // Advanced pooling can be added in later phases
    // size_t total_size = 1;
    // for (int dim : shape)
    // {
    //     total_size *= dim;
    // }

    // Tensor tensor;
    // tensor.shape = shape;
    // tensor.data = std::make_shared<AlignedBuffer<float>>(total_size);
    // return tensor;

    // Phase 1: Simple implementation - just create a new tensor
    return Tensor(shape, true);
}

void MemoryPool::release_tensor(Tensor &tensor)
{
    // Phase 1: Automatic cleanup via shared_ptr
    // No explicit action needed - RAII handles it
    tensor.data.reset();
}

void *MemoryPool::acquire_gpu_buffer(size_t bytes)
{
    (void)bytes; // Suppress unused parameter warning
    // Phase 2: GPU buffer pooling (placeholder for now)
    // Will implement with cudaMalloc in Phase 2
    throw std::runtime_error("GPU buffer pooling not implemented in Phase 1 (CPU-only)");
}

void MemoryPool::release_gpu_buffer(void *ptr)
{
    (void)ptr; // Suppress unused parameter warning
    // Phase 2: GPU buffer release (placeholder for now)
    // Will implement with cudaFree in Phase 2
    throw std::runtime_error("GPU buffer release not implemented in Phase 1 (CPU-only)");
}
