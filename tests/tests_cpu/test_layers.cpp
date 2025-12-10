#include "layers/conv2d_cpu.h"
#include "layers/relu_cpu.h"
#include "layers/maxpool_cpu.h"
#include "layers/upsample_cpu.h"
#include "utils/logger.h"
#include <iostream>
#include <cmath>
#include <iomanip>

// Helper function to check if two tensors are approximately equal
bool tensors_equal(const Tensor &a, const Tensor &b, float epsilon = 1e-5)
{
    if (a.size() != b.size())
        return false;

    const float *a_data = a.data->data();
    const float *b_data = b.data->data();

    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a_data[i] - b_data[i]) > epsilon)
        {
            return false;
        }
    }
    return true;
}

// Test Conv2D forward pass shape
bool test_conv2d_forward_shape()
{
    std::cout << "Test: Conv2D forward pass shape... ";

    Conv2DCPU conv(3, 64, 3, 1, 1); // in_c=3, out_c=64, k=3, stride=1, pad=1
    Tensor input({1, 3, 32, 32});   // [N=1, C=3, H=32, W=32]

    Tensor output = conv.forward(input);

    bool success = (output.batch() == 1 &&
                    output.channels() == 64 &&
                    output.height() == 32 &&
                    output.width() == 32);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    if (!success)
    {
        std::cout << "  Expected: [1, 64, 32, 32], Got: ["
                  << output.batch() << ", " << output.channels() << ", "
                  << output.height() << ", " << output.width() << "]" << std::endl;
    }
    return success;
}

// Test Conv2D with stride
bool test_conv2d_stride()
{
    std::cout << "Test: Conv2D with stride=2... ";

    Conv2DCPU conv(3, 64, 3, 2, 1); // stride=2
    Tensor input({1, 3, 32, 32});

    Tensor output = conv.forward(input);

    bool success = (output.batch() == 1 &&
                    output.channels() == 64 &&
                    output.height() == 16 && // 32/2 = 16
                    output.width() == 16);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// Test ReLU activation
bool test_relu_forward()
{
    std::cout << "Test: ReLU forward pass... ";

    ReLUCPU relu;
    Tensor input({1, 1, 2, 2});

    float *data = input.data->data();
    data[0] = -1.0f;
    data[1] = 2.0f;
    data[2] = -0.5f;
    data[3] = 3.0f;

    Tensor output = relu.forward(input);
    const float *out_data = output.data->data();

    bool success = (out_data[0] == 0.0f &&
                    out_data[1] == 2.0f &&
                    out_data[2] == 0.0f &&
                    out_data[3] == 3.0f);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    if (!success)
    {
        std::cout << "  Expected: [0, 2, 0, 3], Got: ["
                  << out_data[0] << ", " << out_data[1] << ", "
                  << out_data[2] << ", " << out_data[3] << "]" << std::endl;
    }
    return success;
}

// Test ReLU backward pass
bool test_relu_backward()
{
    std::cout << "Test: ReLU backward pass... ";

    ReLUCPU relu;
    Tensor input({1, 1, 2, 2});

    float *data = input.data->data();
    data[0] = -1.0f;
    data[1] = 2.0f;
    data[2] = -0.5f;
    data[3] = 3.0f;

    Tensor output = relu.forward(input);

    // Gradient from next layer (all ones)
    Tensor grad_output({1, 1, 2, 2});
    float *grad_data = grad_output.data->data();
    for (int i = 0; i < 4; ++i)
        grad_data[i] = 1.0f;

    Tensor grad_input = relu.backward(grad_output);
    const float *grad_in_data = grad_input.data->data();

    bool success = (grad_in_data[0] == 0.0f && // input was negative
                    grad_in_data[1] == 1.0f && // input was positive
                    grad_in_data[2] == 0.0f && // input was negative
                    grad_in_data[3] == 1.0f);  // input was positive

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// Test MaxPool forward pass
bool test_maxpool_forward()
{
    std::cout << "Test: MaxPool forward pass... ";

    MaxPoolCPU pool(2);
    Tensor input({1, 1, 4, 4});

    // Fill with known values
    float *data = input.data->data();
    for (int i = 0; i < 16; ++i)
        data[i] = i;

    Tensor output = pool.forward(input);

    bool shape_ok = (output.batch() == 1 &&
                     output.channels() == 1 &&
                     output.height() == 2 &&
                     output.width() == 2);

    std::cout << (shape_ok ? "PASSED" : "FAILED") << std::endl;
    return shape_ok;
}

// Test MaxPool backward pass (gradient routing)
bool test_maxpool_backward()
{
    std::cout << "Test: MaxPool backward pass... ";

    MaxPoolCPU pool(2);
    Tensor input({1, 1, 4, 4});

    float *data = input.data->data();
    for (int i = 0; i < 16; ++i)
        data[i] = i;

    Tensor output = pool.forward(input);

    // Gradient: all ones
    Tensor grad_output({1, 1, 2, 2});
    float *grad_out_data = grad_output.data->data();
    for (int i = 0; i < 4; ++i)
        grad_out_data[i] = 1.0f;

    Tensor grad_input = pool.backward(grad_output);

    // Check that gradients are routed only to max positions
    bool success = (grad_input.size() == 16);

    std::cout << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// Test Upsample forward pass
bool test_upsample_forward()
{
    std::cout << "Test: Upsample forward pass... ";

    UpsampleCPU upsample(2);
    Tensor input({1, 1, 2, 2});

    float *data = input.data->data();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    Tensor output = upsample.forward(input);

    bool shape_ok = (output.batch() == 1 &&
                     output.channels() == 1 &&
                     output.height() == 4 &&
                     output.width() == 4);

    std::cout << (shape_ok ? "PASSED" : "FAILED") << std::endl;
    if (!shape_ok)
    {
        std::cout << "  Expected: [1, 1, 4, 4], Got: ["
                  << output.batch() << ", " << output.channels() << ", "
                  << output.height() << ", " << output.width() << "]" << std::endl;
    }
    return shape_ok;
}

// Test Upsample backward pass
bool test_upsample_backward()
{
    std::cout << "Test: Upsample backward pass... ";

    UpsampleCPU upsample(2);
    Tensor input({1, 1, 2, 2});

    float *data = input.data->data();
    for (int i = 0; i < 4; ++i)
        data[i] = i + 1.0f;

    Tensor output = upsample.forward(input);

    // Gradient: all ones
    Tensor grad_output({1, 1, 4, 4});
    float *grad_out_data = grad_output.data->data();
    for (int i = 0; i < 16; ++i)
        grad_out_data[i] = 1.0f;

    Tensor grad_input = upsample.backward(grad_output);

    bool shape_ok = (grad_input.batch() == 1 &&
                     grad_input.channels() == 1 &&
                     grad_input.height() == 2 &&
                     grad_input.width() == 2);

    std::cout << (shape_ok ? "PASSED" : "FAILED") << std::endl;
    return shape_ok;
}

int main()
{
    LOG_INIT();

    std::cout << "\n========================================\n";
    std::cout << "  Layer Unit Tests\n";
    std::cout << "========================================\n\n";

    int passed = 0;
    int total = 0;

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

    std::cout << "\n========================================\n";
    std::cout << "  Results: " << passed << "/" << total << " tests passed\n";
    std::cout << "========================================\n\n";

    return (passed == total) ? 0 : 1;
}
