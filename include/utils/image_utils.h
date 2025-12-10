#pragma once
#include "data/data_types.h"
#include <string>
#include <vector>

namespace ImageUtils
{

    // Save a batch of images as PPM files (original vs reconstructed)
    // Creates side-by-side comparison images
    void save_reconstruction_samples(
        const Tensor &original,
        const Tensor &reconstructed,
        const std::string &output_dir,
        const std::string &prefix,
        int num_samples = 8);

    // Save a single image as PPM file
    void save_image_ppm(
        const float *image_data,
        int height,
        int width,
        const std::string &filepath);

    // Create a side-by-side comparison image
    void save_comparison_ppm(
        const float *original,
        const float *reconstructed,
        int height,
        int width,
        const std::string &filepath);

    // Calculate PSNR (Peak Signal-to-Noise Ratio) between two images
    float calculate_psnr(const Tensor &original, const Tensor &reconstructed);

    // Calculate SSIM (Structural Similarity Index) - simplified version
    float calculate_ssim(const Tensor &original, const Tensor &reconstructed);
}
