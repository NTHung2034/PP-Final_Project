#pragma once
#include <string>
#include <vector>

namespace ImageUtils
{

    // Save a batch of images as PPM files (original vs reconstructed)

    void save_reconstruction_samples(
        const float *original,
        const float *reconstructed,
        int batch_size,
        int channels,
        int height,
        int width,
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

}
