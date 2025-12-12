#pragma once
#include <cstddef>

/**
 * ReLU Activation Layer (CPU)
 *
 * - Input/Output format: flat array (any shape)
 * - Applies max(0, x) element-wise
 */
class ReLUCPU
{
public:
    ReLUCPU() = default;
    ~ReLUCPU();

    // Disable copy
    ReLUCPU(const ReLUCPU &) = delete;
    ReLUCPU &operator=(const ReLUCPU &) = delete;

    /**
     * Forward pass
     * @param input     Input data (flat array)
     * @param size      Total number of elements
     * @param output    Optional output buffer. If provided, writes directly to it.
     *                  If nullptr, uses internal buffer.
     * @return          Output data (same size as input)
     */
    float *forward(const float *input, size_t size, float *output = nullptr);

    /**
     * Backward pass
     * @param grad_output   Gradient from next layer
     * @return              Gradient w.r.t. input
     */
    float *backward(const float *grad_output);

private:
    // Cached data for backward pass
    float *cached_input_ = nullptr;
    size_t cached_size_ = 0;

    // Output buffer (reused)
    float *output_buffer_ = nullptr;
    float *grad_input_buffer_ = nullptr;
    size_t buffer_size_ = 0;
};