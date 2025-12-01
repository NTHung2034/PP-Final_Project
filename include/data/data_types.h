#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>


// Multi-dimensional tensor using standard vector (no custom alignment)
struct Tensor {
    std::vector<int> shape;                    // [N, C, H, W] or [N, H, W, C]
    std::vector<int> strides;                  // Strides for each dimension
    std::shared_ptr<std::vector<float>> data;  // Standard vector instead of AlignedBuffer
    
    // Default constructor creates empty tensor)
    Tensor() : shape(), strides(), data(nullptr) {}
    
    // Constructor with dimensions
    Tensor(const std::vector<int> &dims, bool zero_init = true) : shape(dims), data(nullptr) {
        // Calculate strides
        strides.resize(dims.size());
        strides.back() = 1;
        for (int i = dims.size() - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        // Allocate memory
        size_t total_elements = 1;
        for (int d : dims) total_elements *= d;
        
        if (zero_init) {
            data = std::make_shared<std::vector<float>>(total_elements, 0.0f);
        } else {
            data = std::make_shared<std::vector<float>>(total_elements);
        }
    }

    // Access element (with bounds checking in debug)
    float &operator()(const std::vector<int> &indices) {
        #ifndef NDEBUG
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Index dimension mismatch");
        }
        #endif

        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            #ifndef NDEBUG
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            #endif
            offset += indices[i] * strides[i];
        }
        return (*data)[offset];
    }

    const float &operator()(const std::vector<int> &indices) const {
        return const_cast<Tensor *>(this)->operator()(indices);
    }

    // Get flat size
    size_t size() const {
        size_t sz = 1;
        for (int d : shape)
            sz *= d;
        return sz;
    }
    
    // Direct data access
    float* raw_data() { return data->data(); }
    const float* raw_data() const { return data->data(); }
    
    // Check dimensions
    bool is_nchw() const { return shape.size() == 4; }
    int batch() const { return shape[0]; }
    int channels() const { return shape[1]; }
    int height() const { return shape[2]; }
    int width() const { return shape[3]; }
};