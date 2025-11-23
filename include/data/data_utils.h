#pragma once
#include "data_types.h"
#include <memory>

namespace DataUtils {
    
    // In-place normalization to [0, 1] range
    void normalize_tensor(Tensor& tensor);
    
    // Standardize tensor (mean=0, std=1)
    void standardize_tensor(Tensor& tensor);
    
    // Data augmentation: horizontal flip
    void horizontal_flip(Tensor& tensor, float probability = 0.5f);
    
    // Data augmentation: random crop
    void random_crop(Tensor& tensor, int crop_size, float probability = 0.5f);
    
    // Save tensor to binary file (for debugging)
    void save_tensor(const Tensor& tensor, const std::string& filepath);
    
    // Load tensor from binary file
    Tensor load_tensor(const std::string& filepath);
    
    // Statistics
    void compute_statistics(const Tensor& tensor, float* mean, float* std);
}