#pragma once
#include <vector>
#include <cstddef>

/**
 * Conv2D Layer (CPU)
 * 
 * - Input/Output format: NCHW (batch, channels, height, width)
 * - Uses raw float pointers for easy GPU porting
 * - Allocates output buffer internally (caller should NOT free)
 */
class Conv2DCPU {
public:
    Conv2DCPU(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);
    ~Conv2DCPU();
    
    // Disable copy
    Conv2DCPU(const Conv2DCPU&) = delete;
    Conv2DCPU& operator=(const Conv2DCPU&) = delete;

    /**
     * Forward pass
     * @param input     Input data [batch, in_channels, in_h, in_w]
     * @param batch     Batch size
     * @param in_h      Input height
     * @param in_w      Input width
     * @return          Output data [batch, out_channels, out_h, out_w]
     */
    float* forward(const float* input, int batch, int in_h, int in_w);
    
    /**
     * Backward pass
     * @param grad_output   Gradient from next layer [batch, out_channels, out_h, out_w]
     * @return              Gradient w.r.t. input [batch, in_channels, in_h, in_w]
     */
    float* backward(const float* grad_output);
    
    // Weight management
    void set_weights(const float* weights, const float* bias);
    void get_weights(float* weights, float* bias) const;
    void get_gradients(float* grad_w, float* grad_b) const;
    void update_weights(float learning_rate);
    
    // Getters for output dimensions
    int get_output_height(int in_h) const { return (in_h + 2 * pad_ - k_size_) / stride_ + 1; }
    int get_output_width(int in_w) const { return (in_w + 2 * pad_ - k_size_) / stride_ + 1; }
    int get_weight_size() const { return out_c_ * in_c_ * k_size_ * k_size_; }
    int get_bias_size() const { return out_c_; }

private:
    int in_c_, out_c_, k_size_, stride_, pad_;
    
    // Weights and biases
    std::vector<float> weights_;
    std::vector<float> bias_;
    std::vector<float> grad_w_, grad_b_;
    
    // Cached data for backward pass
    float* cached_input_ = nullptr;
    int cached_batch_ = 0;
    int cached_in_h_ = 0;
    int cached_in_w_ = 0;
    size_t cached_input_size_ = 0;
    
    // Output buffer (reused)
    float* output_buffer_ = nullptr;
    float* grad_input_buffer_ = nullptr;
    size_t output_buffer_size_ = 0;
    size_t grad_input_buffer_size_ = 0;
};