// CPU Layer Unit Tests - Tensor API
#include "layers/conv2d_cpu.h"
#include "layers/relu_cpu.h"
#include "layers/maxpool_cpu.h"
#include "layers/upsample_cpu.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstring>

// Test Conv2D forward pass shape
bool test_conv2d_forward_shape()
{
    std::cout << "Test: Conv2D forward pass shape... ";

    Conv2DCPU conv(3, 64, 3, 1, 1);

    Tensor input({1, 3, 32, 32});
    for (size_t i = 0; i < input.size(); i++)
    {
        input.raw_data()[i] = 0.5f;
    }

    Tensor output = conv.forward(input);

    bool success = (output.height() == 32 && output.width() == 32 &&
                    output.channels() == 64 && output.batch() == 1);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    if (!success)
    {
        std::cout << "  Expected: [1, 64, 32, 32], Got: [" << output.batch() << ", "
                  << output.channels() << ", " << output.height() << ", " << output.width() << "]" << std::endl;
    }

    return success;
}

// Test Conv2D with stride
bool test_conv2d_stride()
{
    std::cout << "Test: Conv2D with stride=2... ";

    Conv2DCPU conv(3, 64, 3, 2, 1);

    Tensor input({1, 3, 32, 32});
    for (size_t i = 0; i < input.size(); i++)
    {
        input.raw_data()[i] = 0.5f;
    }

    Tensor output = conv.forward(input);

    bool success = (output.height() == 16 && output.width() == 16);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;

    return success;
}

// Test ReLU activation
bool test_relu_forward()
{
    std::cout << "Test: ReLU forward pass... ";

    ReLUCPU relu;

    Tensor input({1, 1, 1, 4});
    input.raw_data()[0] = -1.0f;
    input.raw_data()[1] = 2.0f;
    input.raw_data()[2] = -0.5f;
    input.raw_data()[3] = 3.0f;

    Tensor output = relu.forward(input);

    bool success = (output.raw_data()[0] == 0.0f && output.raw_data()[1] == 2.0f &&
                    output.raw_data()[2] == 0.0f && output.raw_data()[3] == 3.0f);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    if (!success)
    {
        std::cout << "  Expected: [0, 2, 0, 3], Got: [" << output.raw_data()[0] << ", "
                  << output.raw_data()[1] << ", " << output.raw_data()[2] << ", " << output.raw_data()[3] << "]" << std::endl;
    }
    return success;
}

// Test ReLU backward pass
bool test_relu_backward()
{
    std::cout << "Test: ReLU backward pass... ";

    ReLUCPU relu;

    Tensor input({1, 1, 1, 4});
    input.raw_data()[0] = -1.0f;
    input.raw_data()[1] = 2.0f;
    input.raw_data()[2] = -0.5f;
    input.raw_data()[3] = 3.0f;

    relu.forward(input);

    Tensor grad_out({1, 1, 1, 4});
    for (int i = 0; i < 4; i++)
    {
        grad_out.raw_data()[i] = 1.0f;
    }

    Tensor grad_in = relu.backward(grad_out);

    bool success = (grad_in.raw_data()[0] == 0.0f && grad_in.raw_data()[1] == 1.0f &&
                    grad_in.raw_data()[2] == 0.0f && grad_in.raw_data()[3] == 1.0f);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// Test MaxPool forward pass
bool test_maxpool_forward()
{
    std::cout << "Test: MaxPool forward pass... ";

    MaxPoolCPU pool(2);

    Tensor input({1, 1, 4, 4});
    for (int i = 0; i < 16; i++)
    {
        input.raw_data()[i] = static_cast<float>(i);
    }

    Tensor output = pool.forward(input);

    bool shape_ok = (output.height() == 2 && output.width() == 2);

    bool values_ok = (output.raw_data()[0] == 5.0f && output.raw_data()[1] == 7.0f &&
                      output.raw_data()[2] == 13.0f && output.raw_data()[3] == 15.0f);

    bool success = shape_ok && values_ok;
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;

    if (!values_ok)
    {
        std::cout << "  Expected: [5, 7, 13, 15], Got: [" << output.raw_data()[0] << ", "
                  << output.raw_data()[1] << ", " << output.raw_data()[2] << ", " << output.raw_data()[3] << "]" << std::endl;
    }
    return success;
}

// Test MaxPool backward pass
bool test_maxpool_backward()
{
    std::cout << "Test: MaxPool backward pass... ";

    MaxPoolCPU pool(2);

    Tensor input({1, 1, 4, 4});
    for (int i = 0; i < 16; i++)
    {
        input.raw_data()[i] = static_cast<float>(i);
    }

    pool.forward(input);

    Tensor grad_out({1, 1, 2, 2});
    for (int i = 0; i < 4; i++)
    {
        grad_out.raw_data()[i] = 1.0f;
    }

    Tensor grad_in = pool.backward(grad_out);

    bool success = (grad_in.raw_data()[5] == 1.0f && grad_in.raw_data()[7] == 1.0f &&
                    grad_in.raw_data()[13] == 1.0f && grad_in.raw_data()[15] == 1.0f);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// Test Upsample forward pass
bool test_upsample_forward()
{
    std::cout << "Test: Upsample forward pass... ";

    UpsampleCPU upsample(2);

    Tensor input({1, 1, 2, 2});
    input.raw_data()[0] = 1.0f;
    input.raw_data()[1] = 2.0f;
    input.raw_data()[2] = 3.0f;
    input.raw_data()[3] = 4.0f;

    Tensor output = upsample.forward(input);

    bool shape_ok = (output.height() == 4 && output.width() == 4);

    bool values_ok = (output.raw_data()[0] == 1.0f && output.raw_data()[1] == 1.0f &&
                      output.raw_data()[2] == 2.0f && output.raw_data()[3] == 2.0f &&
                      output.raw_data()[4] == 1.0f && output.raw_data()[5] == 1.0f);

    bool success = shape_ok && values_ok;
    std::cout << (success ? "PASSED" : "FAILED") << std::endl;

    if (!shape_ok)
    {
        std::cout << "  Expected: [1, 1, 4, 4], Got: [" << output.batch() << ", " << output.channels()
                  << ", " << output.height() << ", " << output.width() << "]" << std::endl;
    }
    return success;
}

// Test Upsample backward pass
bool test_upsample_backward()
{
    std::cout << "Test: Upsample backward pass... ";

    UpsampleCPU upsample(2);

    Tensor input({1, 1, 2, 2});
    input.raw_data()[0] = 1.0f;
    input.raw_data()[1] = 2.0f;
    input.raw_data()[2] = 3.0f;
    input.raw_data()[3] = 4.0f;

    upsample.forward(input);

    Tensor grad_out({1, 1, 4, 4});
    for (int i = 0; i < 16; i++)
    {
        grad_out.raw_data()[i] = 1.0f;
    }

    Tensor grad_in = upsample.backward(grad_out);

    bool success = (grad_in.raw_data()[0] == 4.0f && grad_in.raw_data()[1] == 4.0f &&
                    grad_in.raw_data()[2] == 4.0f && grad_in.raw_data()[3] == 4.0f);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

int main()
{
    std::cout << "\n=== Layer Unit Tests (Tensor API) ===\n\n";

    int passed = 0, total = 0;

    // Conv2D tests
    total++;
    if (test_conv2d_forward_shape())
        passed++;
    total++;
    if (test_conv2d_stride())
        passed++;

    // ReLU tests
    total++;
    if (test_relu_forward())
        passed++;
    total++;
    if (test_relu_backward())
        passed++;

    // MaxPool tests
    total++;
    if (test_maxpool_forward())
        passed++;
    total++;
    if (test_maxpool_backward())
        passed++;

    // Upsample tests
    total++;
    if (test_upsample_forward())
        passed++;
    total++;
    if (test_upsample_backward())
        passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n\n";

    return (passed == total) ? 0 : 1;
}
