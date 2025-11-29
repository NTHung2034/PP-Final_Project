/**
 * @file upsample_cpu.cpp
 * @brief CPU implementation of 2D Upsampling Layer (Nearest Neighbor)
 * 
 * This implementation uses nearest neighbor interpolation to increase
 * spatial dimensions. Each input pixel is replicated to a scale x scale
 * block of output pixels.
 * 
 * @author CIFAR-10 Autoencoder Project Team
 * @date 2025
 */

#include "layers/upsample_cpu.h"
#include <stdexcept>
#include <algorithm>

UpsampleCPU::UpsampleCPU(int scale)
    : scale_(scale)
{
    if (scale <= 0) {
        throw std::invalid_argument("Upsample: scale must be positive");
    }
}

Tensor UpsampleCPU::forward(const Tensor& input) {
    /**
     * Forward pass: Nearest neighbor upsampling
     * 
     * Each input pixel at (ih, iw) is copied to output positions
     * (oh, ow) where oh in [ih*scale, (ih+1)*scale) and ow in [iw*scale, (iw+1)*scale)
     * 
     * Equivalently: output[n,c,oh,ow] = input[n,c,oh/scale,ow/scale]
     */
    
    // Cache input shape for backward pass
    cached_input_shape_ = input.shape;
    
    // Get input dimensions
    int batch = input.batch();
    int channels = input.channels();
    int in_h = input.height();
    int in_w = input.width();
    
    // Calculate output dimensions
    int out_h = in_h * scale_;
    int out_w = in_w * scale_;
    
    // Allocate output tensor
    Tensor output({batch, channels, out_h, out_w});
    
    // Get raw pointers
    const float* in_data = input.data->data();
    float* out_data = output.data->data();
    
    // Compute strides for NCHW layout
    const int in_stride_n = channels * in_h * in_w;
    const int in_stride_c = in_h * in_w;
    const int in_stride_h = in_w;
    
    const int out_stride_n = channels * out_h * out_w;
    const int out_stride_c = out_h * out_w;
    const int out_stride_h = out_w;
    
    // Perform upsampling
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    // Map output position to input position
                    int ih = oh / scale_;
                    int iw = ow / scale_;
                    
                    // Compute indices
                    int in_idx = n * in_stride_n + c * in_stride_c + 
                                ih * in_stride_h + iw;
                    int out_idx = n * out_stride_n + c * out_stride_c + 
                                  oh * out_stride_h + ow;
                    
                    // Copy value
                    out_data[out_idx] = in_data[in_idx];
                }
            }
        }
    }
    
    return output;
}

Tensor UpsampleCPU::backward(const Tensor& grad_output) {
    /**
     * Backward pass: Sum gradients from replicated positions
     * 
     * Since each input pixel was copied to scale*scale output pixels,
     * the gradient w.r.t. input is the sum of gradients from all those
     * output positions.
     * 
     * grad_input[n,c,ih,iw] = sum_{oh,ow mapping to ih,iw} grad_output[n,c,oh,ow]
     */
    
    // Get dimensions from cached input shape
    int batch = cached_input_shape_[0];
    int channels = cached_input_shape_[1];
    int in_h = cached_input_shape_[2];
    int in_w = cached_input_shape_[3];
    
    int out_h = grad_output.height();
    int out_w = grad_output.width();
    
    // Allocate gradient tensor (initialized to zero)
    Tensor grad_input(cached_input_shape_);
    
    // Get raw pointers
    const float* grad_out_data = grad_output.data->data();
    float* grad_in_data = grad_input.data->data();
    
    // Compute strides
    const int in_stride_n = channels * in_h * in_w;
    const int in_stride_c = in_h * in_w;
    const int in_stride_h = in_w;
    
    const int out_stride_n = channels * out_h * out_w;
    const int out_stride_c = out_h * out_w;
    const int out_stride_h = out_w;
    
    // Sum gradients from replicated positions
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int ih = 0; ih < in_h; ++ih) {
                for (int iw = 0; iw < in_w; ++iw) {
                    float sum = 0.0f;
                    
                    // Sum over all output positions that came from this input
                    int oh_start = ih * scale_;
                    int ow_start = iw * scale_;
                    
                    for (int dh = 0; dh < scale_; ++dh) {
                        for (int dw = 0; dw < scale_; ++dw) {
                            int oh = oh_start + dh;
                            int ow = ow_start + dw;
                            
                            int out_idx = n * out_stride_n + c * out_stride_c + 
                                          oh * out_stride_h + ow;
                            sum += grad_out_data[out_idx];
                        }
                    }
                    
                    // Store summed gradient
                    int in_idx = n * in_stride_n + c * in_stride_c + 
                                ih * in_stride_h + iw;
                    grad_in_data[in_idx] = sum;
                }
            }
        }
    }
    
    return grad_input;
}
