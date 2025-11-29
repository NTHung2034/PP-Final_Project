#include "data/data_utils.h"
#include "utils/logger.h"
#include <algorithm>
#include <cmath>
#include <fstream>

namespace DataUtils {

    void normalize_tensor(Tensor& tensor) {
        const size_t total_elements = tensor.size();
        float* data = tensor.data->data();
        
        #pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < total_elements; ++i) {
            data[i] = data[i] / 255.0f;
        }
    }

    void standardize_tensor(Tensor& tensor) {
        const size_t total_elements = tensor.size();
        float* data = tensor.data->data();
        
        // Compute mean
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < total_elements; ++i) {
            sum += data[i];
        }
        float mean = static_cast<float>(sum / total_elements);
        
        // Compute std dev
        double sq_sum = 0.0;
        #pragma omp parallel for reduction(+:sq_sum)
        for (size_t i = 0; i < total_elements; ++i) {
            float diff = data[i] - mean;
            sq_sum += diff * diff;
        }
        float std_dev = std::sqrt(static_cast<float>(sq_sum / total_elements));
        
        // Apply standardization
        #pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < total_elements; ++i) {
            data[i] = (data[i] - mean) / (std_dev + 1e-7f); // Add epsilon for stability
        }
    }

    void save_tensor(const Tensor& tensor, const std::string& filepath) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + filepath);
        }
        
        // Write shape
        int ndim = static_cast<int>(tensor.shape.size());
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
        file.write(reinterpret_cast<const char*>(tensor.shape.data()), ndim * sizeof(int));
        
        // Write data
        file.write(reinterpret_cast<const char*>(tensor.data->data()), 
                static_cast<std::streamsize>(tensor.size() * sizeof(float)));
        
        file.close();
        LOG_INFO("Tensor saved to %s", filepath.c_str());
    }

    Tensor load_tensor(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for reading: " + filepath);
        }
        
        // Read shape
        int ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(int));
        std::vector<int> shape(ndim);
        file.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int));
        
        // Create tensor
        Tensor tensor(shape, false);
        
        // Read data
        file.read(reinterpret_cast<char*>(tensor.data->data()), 
                static_cast<std::streamsize>(tensor.size() * sizeof(float)));
        
        file.close();
        return tensor;
    }

} // namespace DataUtils