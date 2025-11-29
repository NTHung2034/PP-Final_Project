#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>

// RAII wrapper for aligned memory allocation
template<typename T>
class AlignedBuffer {
public:
    AlignedBuffer(size_t size, size_t alignment = 64) : size_(size) {
        // Use posix_memalign for cache-line alignment
        if (posix_memalign(&ptr_, alignment, size * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
    }
    
    ~AlignedBuffer() { free(ptr_); }
    
    T* data() { return static_cast<T*>(ptr_); }
    const T* data() const { return static_cast<T*>(ptr_); }
    size_t size() const { return size_; }
    
    // Disable copy, enable move
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    
private:
    void* ptr_ = nullptr;
    size_t size_;
};

// Multi-dimensional tensor optimized for NCHW format
struct Tensor {
    std::vector<int> shape;                    // [N, C, H, W] or [N, H, W, C]
    std::vector<int> strides;                  // Strides for each dimension
    std::shared_ptr<AlignedBuffer<float>> data; // Aligned memory buffer
    
    // Default constructor (creates empty tensor)
    Tensor() : shape(), strides(), data(nullptr) {}
    
    // Constructor
    Tensor(const std::vector<int>& dims, bool zero_init = true) 
        : shape(dims), data(nullptr) {
        // Calculate strides
        strides.resize(dims.size());
        if (!dims.empty()) {
            strides.back() = 1;
            for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
        }
        
        // Allocate memory
        size_t total_elements = 1;
        for (int d : dims) total_elements *= d;
        data = std::make_shared<AlignedBuffer<float>>(total_elements);
        
        if (zero_init) {
            std::fill(data->data(), data->data() + total_elements, 0.0f);
        }
    }
    
    // Access element (with bounds checking in debug)
    float& operator()(const std::vector<int>& indices) {
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
        return data->data()[offset];
    }
    
    const float& operator()(const std::vector<int>& indices) const {
        return const_cast<Tensor*>(this)->operator()(indices);
    }
    
    // Get flat size
    size_t size() const {
        size_t sz = 1;
        for (int d : shape) sz *= d;
        return sz;
    }
    
    // Check dimensions
    bool is_nchw() const { return shape.size() == 4; }
    int batch() const { return shape[0]; }
    int channels() const { return shape[1]; }
    int height() const { return shape[2]; }
    int width() const { return shape[3]; }
};