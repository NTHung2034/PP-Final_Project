// CPU Layer Unit Tests - Raw Pointer API
#include "layers/conv2d_cpu.h"
#include "layers/relu_cpu.h"
#include "layers/maxpool_cpu.h"
#include "layers/upsample_cpu.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstring>

// Helper: allocate and fill array
float* create_input(size_t size, float value = 0.5f) {
    float* data = new float[size];
    for (size_t i = 0; i < size; ++i)
        data[i] = value;
    return data;
}

// Test Conv2D forward pass shape
bool test_conv2d_forward_shape() {
    std::cout << "Test: Conv2D forward pass shape... ";

    Conv2DCPU conv(3, 64, 3, 1, 1);  // in_c=3, out_c=64, k=3, stride=1, pad=1
    
    int batch = 1, in_c = 3, h = 32, w = 32;
    size_t input_size = batch * in_c * h * w;
    float* input = create_input(input_size);

    float* output = conv.forward(input, batch, h, w);
    
    int out_h = conv.get_output_height(h);
    int out_w = conv.get_output_width(w);

    bool success = (out_h == 32 && out_w == 32);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    if (!success) {
        std::cout << "  Expected: [1, 64, 32, 32], Got: [1, 64, " 
                  << out_h << ", " << out_w << "]" << std::endl;
    }
    
    delete[] input;
    return success;
}

// Test Conv2D with stride
bool test_conv2d_stride() {
    std::cout << "Test: Conv2D with stride=2... ";

    Conv2DCPU conv(3, 64, 3, 2, 1);  // stride=2
    
    int batch = 1, in_c = 3, h = 32, w = 32;
    size_t input_size = batch * in_c * h * w;
    float* input = create_input(input_size);

    float* output = conv.forward(input, batch, h, w);
    
    int out_h = conv.get_output_height(h);
    int out_w = conv.get_output_width(w);

    bool success = (out_h == 16 && out_w == 16);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    
    delete[] input;
    return success;
}

// Test ReLU activation
bool test_relu_forward() {
    std::cout << "Test: ReLU forward pass... ";

    ReLUCPU relu;
    
    float input[4] = {-1.0f, 2.0f, -0.5f, 3.0f};
    float* output = relu.forward(input, 4);

    bool success = (output[0] == 0.0f && output[1] == 2.0f && 
                    output[2] == 0.0f && output[3] == 3.0f);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    if (!success) {
        std::cout << "  Expected: [0, 2, 0, 3], Got: [" << output[0] << ", " 
                  << output[1] << ", " << output[2] << ", " << output[3] << "]" << std::endl;
    }
    return success;
}

// Test ReLU backward pass
bool test_relu_backward() {
    std::cout << "Test: ReLU backward pass... ";

    ReLUCPU relu;
    
    float input[4] = {-1.0f, 2.0f, -0.5f, 3.0f};
    relu.forward(input, 4);

    float grad_out[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float* grad_in = relu.backward(grad_out);

    bool success = (grad_in[0] == 0.0f && grad_in[1] == 1.0f && 
                    grad_in[2] == 0.0f && grad_in[3] == 1.0f);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// Test MaxPool forward pass
bool test_maxpool_forward() {
    std::cout << "Test: MaxPool forward pass... ";

    MaxPoolCPU pool(2);
    
    int batch = 1, channels = 1, h = 4, w = 4;
    float input[16];
    for (int i = 0; i < 16; ++i) input[i] = static_cast<float>(i);

    float* output = pool.forward(input, batch, channels, h, w);
    
    int out_h = pool.get_output_height(h);
    int out_w = pool.get_output_width(w);

    bool shape_ok = (out_h == 2 && out_w == 2);
    
    // Check max values: max of [0,1,4,5]=5, [2,3,6,7]=7, [8,9,12,13]=13, [10,11,14,15]=15
    bool values_ok = (output[0] == 5.0f && output[1] == 7.0f && 
                      output[2] == 13.0f && output[3] == 15.0f);

    bool success = shape_ok && values_ok;
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    
    if (!values_ok) {
        std::cout << "  Expected: [5, 7, 13, 15], Got: [" << output[0] << ", " 
                  << output[1] << ", " << output[2] << ", " << output[3] << "]" << std::endl;
    }
    return success;
}

// Test MaxPool backward pass
bool test_maxpool_backward() {
    std::cout << "Test: MaxPool backward pass... ";

    MaxPoolCPU pool(2);
    
    int batch = 1, channels = 1, h = 4, w = 4;
    float input[16];
    for (int i = 0; i < 16; ++i) input[i] = static_cast<float>(i);

    pool.forward(input, batch, channels, h, w);

    float grad_out[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float* grad_in = pool.backward(grad_out);

    // Gradient should be routed to max positions (5,7,13,15 -> indices 5,7,13,15)
    bool success = (grad_in[5] == 1.0f && grad_in[7] == 1.0f && 
                    grad_in[13] == 1.0f && grad_in[15] == 1.0f);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// Test Upsample forward pass
bool test_upsample_forward() {
    std::cout << "Test: Upsample forward pass... ";

    UpsampleCPU upsample(2);
    
    int batch = 1, channels = 1, h = 2, w = 2;
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    float* output = upsample.forward(input, batch, channels, h, w);
    
    int out_h = upsample.get_output_height(h);
    int out_w = upsample.get_output_width(w);

    bool shape_ok = (out_h == 4 && out_w == 4);
    
    // Nearest neighbor: each input pixel becomes 2x2 block
    // [1,1,2,2]
    // [1,1,2,2]
    // [3,3,4,4]
    // [3,3,4,4]
    bool values_ok = (output[0] == 1.0f && output[1] == 1.0f && 
                      output[2] == 2.0f && output[3] == 2.0f &&
                      output[4] == 1.0f && output[5] == 1.0f);

    bool success = shape_ok && values_ok;
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    
    if (!shape_ok) {
        std::cout << "  Expected: [1, 1, 4, 4], Got: [1, 1, " 
                  << out_h << ", " << out_w << "]" << std::endl;
    }
    return success;
}

// Test Upsample backward pass
bool test_upsample_backward() {
    std::cout << "Test: Upsample backward pass... ";

    UpsampleCPU upsample(2);
    
    int batch = 1, channels = 1, h = 2, w = 2;
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    upsample.forward(input, batch, channels, h, w);

    float grad_out[16];
    for (int i = 0; i < 16; ++i) grad_out[i] = 1.0f;

    float* grad_in = upsample.backward(grad_out);

    // Each 2x2 block sums to 4.0
    bool success = (grad_in[0] == 4.0f && grad_in[1] == 4.0f && 
                    grad_in[2] == 4.0f && grad_in[3] == 4.0f);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

int main() {
    std::cout << "\n=== Layer Unit Tests (Raw Pointer API) ===\n\n";

    int passed = 0, total = 0;

    // Conv2D tests
    total++; if (test_conv2d_forward_shape()) passed++;
    total++; if (test_conv2d_stride()) passed++;

    // ReLU tests
    total++; if (test_relu_forward()) passed++;
    total++; if (test_relu_backward()) passed++;

    // MaxPool tests
    total++; if (test_maxpool_forward()) passed++;
    total++; if (test_maxpool_backward()) passed++;

    // Upsample tests
    total++; if (test_upsample_forward()) passed++;
    total++; if (test_upsample_backward()) passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n\n";

    return (passed == total) ? 0 : 1;
}
