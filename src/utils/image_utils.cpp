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
            save_comparison_ppm(orig_ptr, recon_ptr, height, width, filepath);
        }

        std::cout << "      ✓ Saved " << num_samples << " reconstruction samples to "
                  << output_dir << "/" << prefix << "_sample_*.ppm\n";
    }

}
