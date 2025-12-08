#include "utils/image_utils.h"
#include "utils/logger.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(dir, mode) _mkdir(dir)
#endif

namespace ImageUtils
{

    void save_image_ppm(const float *image_data, int height, int width, const std::string &filepath)
    {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            LOG_ERROR("Failed to open file for writing: %s", filepath.c_str());
            return;
        }

        // Write PPM header
        file << "P6\n"
             << width << " " << height << "\n255\n";

        // Write pixel data
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                for (int c = 0; c < 3; ++c)
                {
                    int idx = c * height * width + h * width + w; // NCHW format
                    float pixel = image_data[idx];
                    // Clamp to [0, 1] and convert to [0, 255]
                    unsigned char byte = static_cast<unsigned char>(
                        std::max(0.0f, std::min(255.0f, pixel * 255.0f)));
                    file.write(reinterpret_cast<const char *>(&byte), 1);
                }
            }
        }

        file.close();
    }

    void save_comparison_ppm(const float *original, const float *reconstructed,
                             int height, int width, const std::string &filepath)
    {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            LOG_ERROR("Failed to open file for writing: %s", filepath.c_str());
            return;
        }

        // Create side-by-side image (double width)
        int new_width = width * 2 + 4; // Add separator
        file << "P6\n"
             << new_width << " " << height << "\n255\n";

        // Write pixel data
        for (int h = 0; h < height; ++h)
        {
            // Write original image
            for (int w = 0; w < width; ++w)
            {
                for (int c = 0; c < 3; ++c)
                {
                    int idx = c * height * width + h * width + w;
                    float pixel = original[idx];
                    unsigned char byte = static_cast<unsigned char>(
                        std::max(0.0f, std::min(255.0f, pixel * 255.0f)));
                    file.write(reinterpret_cast<const char *>(&byte), 1);
                }
            }

            // Write separator (white line)
            for (int sep = 0; sep < 4; ++sep)
            {
                for (int c = 0; c < 3; ++c)
                {
                    unsigned char white = 255;
                    file.write(reinterpret_cast<const char *>(&white), 1);
                }
            }

            // Write reconstructed image
            for (int w = 0; w < width; ++w)
            {
                for (int c = 0; c < 3; ++c)
                {
                    int idx = c * height * width + h * width + w;
                    float pixel = reconstructed[idx];
                    unsigned char byte = static_cast<unsigned char>(
                        std::max(0.0f, std::min(255.0f, pixel * 255.0f)));
                    file.write(reinterpret_cast<const char *>(&byte), 1);
                }
            }
        }

        file.close();
    }

    void save_reconstruction_samples(const Tensor &original, const Tensor &reconstructed,
                                     const std::string &output_dir, const std::string &prefix,
                                     int num_samples)
    {
        // Create output directory if it doesn't exist
        mkdir(output_dir.c_str(), 0755);

        int batch_size = original.batch();
        int channels = original.channels();
        int height = original.height();
        int width = original.width();

        num_samples = std::min(num_samples, batch_size);

        LOG_INFO("Saving %d reconstruction samples to %s", num_samples, output_dir.c_str());

        for (int i = 0; i < num_samples; ++i)
        {
            // Calculate offset for this image in the batch
            int image_offset = i * channels * height * width;

            const float *orig_ptr = original.raw_data() + image_offset;
            const float *recon_ptr = reconstructed.raw_data() + image_offset;

            // Save comparison image
            std::string filepath = output_dir + "/" + prefix + "_sample_" + std::to_string(i) + ".ppm";
            save_comparison_ppm(orig_ptr, recon_ptr, height, width, channels, filepath);
        }

        std::cout << "      ✓ Saved " << num_samples << " reconstruction samples to "
                  << output_dir << "/" << prefix << "_sample_*.ppm\n";
    }

    float calculate_psnr(const Tensor &original, const Tensor &reconstructed)
    {
        if (original.size() != reconstructed.size())
        {
            LOG_ERROR("Tensor size mismatch in PSNR calculation");
            return 0.0f;
        }

        const float *orig_data = original.raw_data();
        const float *recon_data = reconstructed.raw_data();
        size_t n = original.size();

        // Calculate MSE
        double mse = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = orig_data[i] - recon_data[i];
            mse += diff * diff;
        }
        mse /= n;

        if (mse < 1e-10)
            return 100.0f; // Perfect reconstruction

        // PSNR = 10 * log10(MAX^2 / MSE)
        // For normalized images [0,1], MAX = 1
        float psnr = 10.0f * std::log10(1.0f / mse);
        return psnr;
    }

    float calculate_ssim(const Tensor &original, const Tensor &reconstructed)
    {
        // Simplified SSIM calculation (averaged over all pixels)
        if (original.size() != reconstructed.size())
        {
            LOG_ERROR("Tensor size mismatch in SSIM calculation");
            return 0.0f;
        }

        const float *orig_data = original.raw_data();
        const float *recon_data = reconstructed.raw_data();
        size_t n = original.size();

        // Calculate means
        double mean_x = 0.0, mean_y = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            mean_x += orig_data[i];
            mean_y += recon_data[i];
        }
        mean_x /= n;
        mean_y /= n;

        // Calculate variances and covariance
        double var_x = 0.0, var_y = 0.0, cov_xy = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double dx = orig_data[i] - mean_x;
            double dy = recon_data[i] - mean_y;
            var_x += dx * dx;
            var_y += dy * dy;
            cov_xy += dx * dy;
        }
        var_x /= n;
        var_y /= n;
        cov_xy /= n;

        // SSIM formula (simplified, global)
        const double C1 = 0.01 * 0.01; // (K1*L)^2
        const double C2 = 0.03 * 0.03; // (K2*L)^2

        double numerator = (2.0 * mean_x * mean_y + C1) * (2.0 * cov_xy + C2);
        double denominator = (mean_x * mean_x + mean_y * mean_y + C1) * (var_x + var_y + C2);

        float ssim = static_cast<float>(numerator / denominator);
        return ssim;
    }

} // namespace ImageUtils
