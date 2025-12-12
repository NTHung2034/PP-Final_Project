#pragma once
#include <cstddef>

/**
 * Upsample2D Layer (CPU) - Nearest Neighbor
 *
 * - Input/Output format: NCHW (batch, channels, height, width)
 * - Upsamples by repeating pixels (nearest neighbor)
 */
class UpsampleCPU
{
public:
    explicit UpsampleCPU(int scale = 2);
    ~UpsampleCPU();

    // Disable copy
    UpsampleCPU(const UpsampleCPU &) = delete;
    UpsampleCPU &operator=(const UpsampleCPU &) = delete;

    /**
     * Forward pass
     * @param input     Input data [batch, channels, in_h, in_w]
     * @param batch     Batch size
     * @param channels  Number of channels
     * @param in_h      Input height
     * @param in_w      Input width
     * @param output    Optional output buffer. If provided, writes directly to it.
     *                  If nullptr, uses internal buffer.
     * @return          Output data [batch, channels, out_h, out_w]
     */
    float *forward(const float *input, int batch, int channels, int in_h, int in_w, float *output = nullptr);

    /**
     * Backward pass
     * @param grad_output   Gradient from next layer
     * @return              Gradient w.r.t. input
     */
    float *backward(const float *grad_output);

    // Getters for output dimensions
    int get_output_height(int in_h) const { return in_h * scale_; }
    int get_output_width(int in_w) const { return in_w * scale_; }

private:
    int scale_;

    // Cached data for backward pass
    int cached_batch_ = 0;
    int cached_channels_ = 0;
    int cached_in_h_ = 0;
    int cached_in_w_ = 0;

    // Output buffer (reused)
    float *output_buffer_ = nullptr;
    float *grad_input_buffer_ = nullptr;
    size_t output_buffer_size_ = 0;
    size_t grad_input_buffer_size_ = 0;
};