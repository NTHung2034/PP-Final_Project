# Testing and Deliverables Finalization Guide

**Purpose:** Comprehensive testing and validation before final submission  
**Timeline:** December 9-10, 2025  
**Objective:** Ensure all requirements met, no bugs, professional submission

---

## Table of Contents

1. [Testing Strategy](#1-testing-strategy)
2. [Sanity Checks](#2-sanity-checks)
3. [Comprehensive Tests](#3-comprehensive-tests)
4. [Performance Validation](#4-performance-validation)
5. [Code Quality Checks](#5-code-quality-checks)
6. [Documentation Review](#6-documentation-review)
7. [Final Submission Checklist](#7-final-submission-checklist)
8. [Google Colab Verification](#8-google-colab-verification)

---

## 1. Testing Strategy

### 1.1 Testing Pyramid

```
         ┌─────────────────┐
         │ End-to-End Test │  1 test (full pipeline)
         ├─────────────────┤
         │ Integration     │  ~10 tests (phase combinations)
         ├─────────────────┤
         │ Unit Tests      │  ~50 tests (individual functions)
         └─────────────────┘
```

### 1.2 Test Coverage Goals

| Component          | Coverage Target | Priority |
| ------------------ | --------------- | -------- |
| Data loading       | 100%            | High     |
| CPU layers         | 90%+            | High     |
| GPU kernels        | 90%+            | High     |
| Feature extraction | 100%            | High     |
| SVM integration    | 80%             | Medium   |
| Utilities          | 70%             | Low      |

---

## 2. Sanity Checks

**Duration:** 30 minutes  
**Purpose:** Catch obvious errors quickly

### 2.1 Compilation Check

```bash
# Clean build
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Expected: No errors, no warnings
# If warnings exist, fix them!
```

**Success criteria:**

- ✅ All targets build successfully
- ✅ Zero compilation warnings with `-Wall -Wextra`
- ✅ Executables created in `build/bin/`

### 2.2 Data Loading Sanity Check

```bash
./bin/test_data_loading

# Expected output:
# ✓ Loaded 50000 training images
# ✓ Loaded 10000 test images
# ✓ Pixel values in range [0.0, 1.0]
# ✓ All labels in range [0, 9]
# ✓ Batch generation working
```

**Quick verification script:** `tests/sanity/check_data.sh`

```bash
#!/bin/bash
set -e

echo "=== Data Sanity Checks ==="

# Check CIFAR-10 exists
if [ ! -d "data/cifar-10-batches-bin" ]; then
    echo "❌ CIFAR-10 data not found!"
    exit 1
fi
echo "✓ CIFAR-10 data found"

# Check file sizes
expected_size=30730000  # ~30MB per batch
for batch in data/cifar-10-batches-bin/data_batch_*.bin; do
    size=$(stat -f%z "$batch" 2>/dev/null || stat -c%s "$batch")
    if [ $size -ne $expected_size ]; then
        echo "❌ Batch file $batch has wrong size: $size (expected $expected_size)"
        exit 1
    fi
done
echo "✓ All batch files have correct size"

# Run data loading test
./bin/test_data_loading || exit 1

echo "✓ All data sanity checks passed!"
```

### 2.3 GPU Availability Check

```bash
./bin/check_gpu

# Expected output:
# ✓ CUDA available
# ✓ Device 0: Tesla T4
# ✓ Compute Capability: 7.5
# ✓ Total Memory: 15 GB
```

**Implementation:** `tests/sanity/check_gpu.cu`

```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "❌ No CUDA devices found!\n";
        return 1;
    }

    std::cout << "✓ CUDA available\n";

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "✓ Device " << i << ": " << prop.name << "\n";
        std::cout << "✓ Compute Capability: "
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "✓ Total Memory: "
                  << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
    }

    return 0;
}
```

### 2.4 Quick Smoke Test

**Run one epoch of training:**

```bash
./bin/train_gpu --epochs 1 --batch-size 32

# Should complete in ~20-30 seconds
# Loss should decrease from ~0.15 to ~0.10
```

**If this fails, don't proceed with more tests!**

---

## 3. Comprehensive Tests

**Duration:** 2-3 hours  
**Purpose:** Thorough validation of all components

### 3.1 Unit Tests

**File:** `tests/unit/test_layers.cpp`

```cpp
#include <gtest/gtest.h>
#include "layers/conv2d_cpu.h"
#include "layers/relu_cpu.h"
#include "layers/maxpool_cpu.h"

// Test Conv2D with known input/output
TEST(Conv2DTest, IdentityKernel) {
    // 1×1×3×3 input
    Tensor input({1, 1, 3, 3});
    float* data = input.data->data();
    for (int i = 0; i < 9; ++i) data[i] = i + 1;

    // 1×1×3×3 kernel (all ones)
    Conv2DCPU conv(1, 1, 3, 1, 0);
    std::vector<float> weights(9, 1.0f);
    std::vector<float> bias(1, 0.0f);
    conv.set_weights(weights, bias);

    auto output = conv.forward(input);

    // Output: 1×1×1×1 with value 45 (sum of 1..9)
    EXPECT_EQ(output.shape[0], 1);
    EXPECT_EQ(output.shape[1], 1);
    EXPECT_EQ(output.shape[2], 1);
    EXPECT_EQ(output.shape[3], 1);
    EXPECT_NEAR(output.data->data()[0], 45.0f, 1e-5);
}

TEST(ReLUTest, BasicActivation) {
    Tensor input({1, 1, 2, 2});
    float* data = input.data->data();
    data[0] = -1.0f; data[1] = 2.0f;
    data[2] = 0.0f;  data[3] = -3.0f;

    ReLUCPU relu;
    auto output = relu.forward(input);

    EXPECT_FLOAT_EQ(output.data->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output.data->data()[1], 2.0f);
    EXPECT_FLOAT_EQ(output.data->data()[2], 0.0f);
    EXPECT_FLOAT_EQ(output.data->data()[3], 0.0f);
}

TEST(MaxPoolTest, Simple2x2) {
    Tensor input({1, 1, 4, 4});
    float* data = input.data->data();
    for (int i = 0; i < 16; ++i) data[i] = i;

    MaxPoolCPU pool(2);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape[2], 2);
    EXPECT_EQ(output.shape[3], 2);

    // Expected: max of each 2×2 window
    EXPECT_FLOAT_EQ(output.data->data()[0], 5.0f);   // max(0,1,4,5)
    EXPECT_FLOAT_EQ(output.data->data()[1], 7.0f);   // max(2,3,6,7)
    EXPECT_FLOAT_EQ(output.data->data()[2], 13.0f);  // max(8,9,12,13)
    EXPECT_FLOAT_EQ(output.data->data()[3], 15.0f);  // max(10,11,14,15)
}

// Run all tests
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

**Build and run:**

```bash
# Add to CMakeLists.txt:
# find_package(GTest REQUIRED)
# add_executable(unit_tests tests/unit/test_layers.cpp ...)
# target_link_libraries(unit_tests GTest::GTest GTest::Main)

make unit_tests
./bin/unit_tests
```

**Expected:** All tests pass

### 3.2 GPU Correctness Tests

**Test GPU kernels match CPU:**

```cpp
#include <gtest/gtest.h>
#include "models/autoencoder_cpu.h"
#include "models/autoencoder_gpu.cuh"

TEST(GPUTest, ForwardPassMatchesCPU) {
    const int batch_size = 4;

    // Create random input
    Tensor cpu_input({batch_size, 3, 32, 32});
    std::vector<float> h_input(batch_size * 3 * 32 * 32);
    for (auto& x : h_input) x = (rand() % 1000) / 1000.0f;
    memcpy(cpu_input.data->data(), h_input.data(),
           h_input.size() * sizeof(float));

    // CPU forward pass
    AutoencoderCPU cpu_model;
    auto cpu_output = cpu_model.forward(cpu_input);

    // GPU forward pass
    AutoencoderGPU gpu_model;
    gpu_model.forward(h_input.data(), batch_size);

    std::vector<float> gpu_output(batch_size * 3 * 32 * 32);
    gpu_model.get_output(gpu_output.data());

    // Compare outputs
    float max_diff = 0.0f;
    for (size_t i = 0; i < gpu_output.size(); ++i) {
        float diff = std::abs(cpu_output.data->data()[i] - gpu_output[i]);
        max_diff = std::max(max_diff, diff);
    }

    EXPECT_LT(max_diff, 1e-4) << "GPU output differs from CPU";
}

TEST(GPUTest, OptimizedVersionsMatch) {
    // Test that all GPU optimization versions produce same output
    // (Already implemented in Phase 3 guide)
}
```

### 3.3 End-to-End Pipeline Test

**Full pipeline in one script:** `tests/integration/test_full_pipeline.sh`

```bash
#!/bin/bash
set -e

echo "=== Full Pipeline Integration Test ==="

# Step 1: Train autoencoder (2 epochs for speed)
echo "Step 1: Training autoencoder..."
./bin/train_gpu --epochs 2 --batch-size 64 || exit 1
echo "✓ Training complete"

# Step 2: Extract features
echo "Step 2: Extracting features..."
./bin/extract_features || exit 1
echo "✓ Feature extraction complete"

# Step 3: Train SVM (on subset for speed)
echo "Step 3: Training SVM..."
head -10000 data/train_features.txt > data/train_subset.txt
./external/libsvm/svm-train -s 0 -t 2 -c 10 -g 0.0001 -q \
    data/train_subset.txt models/test_svm.bin || exit 1
echo "✓ SVM training complete"

# Step 4: Predict on test set (subset)
echo "Step 4: Predicting..."
head -1000 data/test_features.txt > data/test_subset.txt
./external/libsvm/svm-predict \
    data/test_subset.txt models/test_svm.bin results/test_pred.txt || exit 1

# Check accuracy is reasonable (>50%)
accuracy=$(grep "Accuracy" results/test_pred.txt | awk '{print $3}' | sed 's/%//')
if (( $(echo "$accuracy > 50.0" | bc -l) )); then
    echo "✓ Accuracy: $accuracy% (acceptable)"
else
    echo "❌ Accuracy: $accuracy% (too low!)"
    exit 1
fi

echo "✓ Full pipeline test passed!"
```

---

## 4. Performance Validation

**Verify all performance targets met**

### 4.1 Performance Test Script

**File:** `tests/performance/benchmark_all.sh`

```bash
#!/bin/bash

echo "=== Performance Validation ==="
echo ""

# CPU Baseline
echo "Testing CPU baseline..."
cpu_time=$(./bin/train_cpu --epochs 1 2>&1 | grep "seconds" | awk '{print $5}')
echo "CPU time (1 epoch): ${cpu_time}s"

# GPU Optimized
echo "Testing GPU optimized..."
gpu_time=$(./bin/train_gpu --epochs 1 2>&1 | grep "seconds" | awk '{print $5}')
echo "GPU time (1 epoch): ${gpu_time}s"

# Calculate speedup
speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc)
echo "Speedup: ${speedup}×"

# Validate targets
echo ""
echo "=== Target Validation ==="

# Target 1: GPU speedup >20×
if (( $(echo "$speedup >= 20.0" | bc -l) )); then
    echo "✓ Speedup target met: ${speedup}× >= 20×"
else
    echo "❌ Speedup target NOT met: ${speedup}× < 20×"
fi

# Target 2: Training time <10 minutes (20 epochs)
total_gpu_time=$(echo "$gpu_time * 20" | bc)
if (( $(echo "$total_gpu_time < 600" | bc -l) )); then
    echo "✓ Training time target met: ${total_gpu_time}s < 600s"
else
    echo "❌ Training time target NOT met: ${total_gpu_time}s >= 600s"
fi

# Target 3: Feature extraction <20 seconds
echo "Testing feature extraction..."
extract_time=$(./bin/extract_features 2>&1 | grep "complete in" | awk '{print $4}')
if (( $(echo "$extract_time < 20" | bc -l) )); then
    echo "✓ Feature extraction target met: ${extract_time}s < 20s"
else
    echo "❌ Feature extraction target NOT met: ${extract_time}s >= 20s"
fi

# Target 4: Test accuracy 60-65%
accuracy=$(cat results/evaluation_summary.txt | grep "Overall Accuracy" | awk '{print $3}' | sed 's/%//')
if (( $(echo "$accuracy >= 60.0 && $accuracy <= 70.0" | bc -l) )); then
    echo "✓ Accuracy target met: ${accuracy}% in [60%, 70%]"
else
    echo "⚠ Accuracy: ${accuracy}% (expected 60-65%)"
fi

echo ""
echo "=== Summary ==="
echo "CPU Time: ${cpu_time}s"
echo "GPU Time: ${gpu_time}s"
echo "Speedup: ${speedup}×"
echo "Feature Extraction: ${extract_time}s"
echo "Test Accuracy: ${accuracy}%"
```

**Run:**

```bash
bash tests/performance/benchmark_all.sh > results/performance_report.txt
cat results/performance_report.txt
```

---

## 5. Code Quality Checks

### 5.1 Memory Leak Detection

```bash
# CPU code
valgrind --leak-check=full --show-leak-kinds=all \
    ./bin/train_cpu --epochs 1 2>&1 | tee valgrind_cpu.log

# Expected: "All heap blocks were freed -- no leaks are possible"

# GPU code
cuda-memcheck --leak-check full ./bin/train_gpu --epochs 1 \
    2>&1 | tee cuda_memcheck.log

# Expected: "ERROR SUMMARY: 0 errors"
```

### 5.2 Code Style Check

**Using clang-format:**

```bash
# Check formatting
find src include -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \
    | xargs clang-format --dry-run -Werror

# Auto-format (if needed)
find src include -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \
    | xargs clang-format -i
```

### 5.3 Static Analysis

```bash
# Using cppcheck
cppcheck --enable=all --suppress=missingIncludeSystem \
    src/ include/ 2>&1 | tee cppcheck.log

# Expected: 0 errors, minimal warnings
```

---

## 6. Documentation Review

### 6.1 README Completeness

**Checklist:**

- [ ] Project description
- [ ] Prerequisites listed (CUDA version, CMake, etc.)
- [ ] Setup instructions (step-by-step)
- [ ] Build instructions
- [ ] Run instructions
- [ ] Google Colab setup
- [ ] Expected outputs
- [ ] Troubleshooting section
- [ ] Contact information

### 6.2 Code Documentation

**Check docstrings:**

```bash
# Count functions with documentation
total_funcs=$(grep -r "^[a-zA-Z].*(" src/ include/ | wc -l)
documented=$(grep -rB1 "^[a-zA-Z].*(" src/ include/ | grep "//\|/\*" | wc -l)
coverage=$(echo "scale=2; $documented / $total_funcs * 100" | bc)

echo "Documentation coverage: ${coverage}%"
# Target: >70%
```

### 6.3 Jupyter Notebook Review

**Verify notebook has all required sections:**

```python
# Script to check notebook sections
import nbformat

nb = nbformat.read('report.ipynb', as_version=4)

required_sections = [
    "Problem Description",
    "Phase 1: CPU Baseline",
    "Phase 2: GPU Implementation",
    "Phase 3: Optimization",
    "Phase 4: SVM Integration",
    "Performance Analysis",
    "Lessons Learned",
    "Conclusion"
]

found_sections = []
for cell in nb.cells:
    if cell.cell_type == 'markdown':
        for section in required_sections:
            if section in cell.source:
                found_sections.append(section)

missing = set(required_sections) - set(found_sections)
if missing:
    print(f"❌ Missing sections: {missing}")
else:
    print("✓ All required sections present")
```

---

## 7. Final Submission Checklist

### 7.1 File Structure Validation

```
PP-Final_Project/
├── README.md                    ✓
├── CMakeLists.txt               ✓
├── report.ipynb                 ✓
├── report.pdf                   ✓
├── docs/
│   ├── PROJECT_PLAN.md          ✓
│   ├── PHASE_*_GUIDE.md         ✓
│   └── ...
├── src/                         ✓
├── include/                     ✓
├── models/
│   └── saved_weights/
│       ├── cpu_encoder_weights.bin  ✓
│       └── gpu_encoder_weights.bin  ✓
├── results/
│   ├── performance_report.txt   ✓
│   ├── confusion_matrix.png     ✓
│   ├── evaluation_summary.txt   ✓
│   └── ...
├── presentation_video_link.txt  ✓
└── team_plan.md                 ✓
```

### 7.2 Pre-Submission Tests

**Run complete test suite:**

```bash
#!/bin/bash
# tests/run_all_tests.sh

echo "=== Running Complete Test Suite ==="
echo ""

# Sanity checks
echo "1. Sanity checks..."
bash tests/sanity/check_data.sh || exit 1
./bin/check_gpu || exit 1
echo "✓ Sanity checks passed"
echo ""

# Unit tests
echo "2. Unit tests..."
./bin/unit_tests || exit 1
echo "✓ Unit tests passed"
echo ""

# Integration tests
echo "3. Integration tests..."
bash tests/integration/test_full_pipeline.sh || exit 1
echo "✓ Integration tests passed"
echo ""

# Performance validation
echo "4. Performance validation..."
bash tests/performance/benchmark_all.sh || exit 1
echo "✓ Performance targets met"
echo ""

# Memory checks
echo "5. Memory leak checks..."
cuda-memcheck --leak-check full ./bin/train_gpu --epochs 1 2>&1 | \
    grep "ERROR SUMMARY: 0 errors" || exit 1
echo "✓ No memory leaks detected"
echo ""

echo "=== All Tests Passed! ==="
echo "Ready for submission."
```

### 7.3 Final Checklist

**Code Package:**

- [ ] All source files present
- [ ] No compiled binaries committed
- [ ] `.gitignore` properly configured
- [ ] README is complete
- [ ] CMakeLists.txt builds successfully

**Report:**

- [ ] Jupyter notebook has all sections
- [ ] All figures generated from code
- [ ] All tables present
- [ ] Runs on Google Colab without errors
- [ ] PDF exported successfully

**Presentation:**

- [ ] Video 15-20 minutes long
- [ ] Contains live demo
- [ ] Uploaded to YouTube (Unlisted)
- [ ] Link in `presentation_video_link.txt`

**Performance:**

- [ ] Training time <10 min (20 epochs)
- [ ] Feature extraction <20 sec
- [ ] GPU speedup >20×
- [ ] Test accuracy 60-65%

---

## 8. Google Colab Verification

### 8.1 Fresh Colab Test

**Create new notebook and run:**

```python
# Cell 1: Clone and setup
!git clone https://github.com/YOUR_REPO/PP-Final_Project.git
%cd PP-Final_Project
!bash scripts/setup_colab.sh

# Cell 2: Check GPU
!nvidia-smi

# Cell 3: Download CIFAR-10
!bash scripts/download_cifar10.sh

# Cell 4: Build
!mkdir build && cd build && cmake .. && make -j4

# Cell 5: Run full pipeline
!cd build && bash ../tests/run_all_tests.sh

# Cell 6: Display results
from IPython.display import Image
display(Image('results/confusion_matrix.png'))
```

**Success criteria:**

- All cells run without errors
- Results match local execution
- Total runtime <30 minutes

### 8.2 Colab Compatibility Checklist

- [ ] No hardcoded local paths (use relative paths)
- [ ] All dependencies installable via apt/pip
- [ ] CUDA version compatible (11.x)
- [ ] Maximum file sizes respected (<100MB per file)
- [ ] Model weights downloadable (if >100MB, use Google Drive link)

---

## 9. Common Last-Minute Issues

### 9.1 "It works on my machine"

**Solution:** Test on clean environment

```bash
# Use Docker for reproducibility
docker run --gpus all -it nvidia/cuda:11.8.0-devel-ubuntu22.04
# Then setup from scratch
```

### 9.2 File Not Found Errors

**Check all paths are relative:**

```bash
# BAD
ifstream file("/home/user/project/data/batch.bin");

# GOOD
ifstream file("../data/cifar-10-batches-bin/data_batch_1.bin");
```

### 9.3 Different Results on Colab

**Possible causes:**

- Different CUDA version → Recompile with correct arch
- Different random seed → Set seed explicitly
- Different precision → Accept small differences (<1e-4)

---

## 10. Emergency Fixes

### 10.1 If Tests Fail 24 Hours Before Deadline

**Priority 1 (Must Fix):**

- Compilation errors
- Crashes/segfaults
- Completely wrong results

**Priority 2 (Should Fix):**

- Performance below targets
- Memory leaks
- Some tests failing

**Priority 3 (Nice to Fix):**

- Code style issues
- Documentation gaps
- Minor warnings

### 10.2 Quick Wins

**If short on time:**

1. Ensure Phase 1 is perfect (foundation)
2. Get Phase 2 working (basic GPU)
3. At least one optimization (Phase 3 v1)
4. Basic SVM results (Phase 4)
5. Focus on report quality

---

## 11. Submission Package

### 11.1 Create Submission Archive

```bash
#!/bin/bash
# scripts/create_submission.sh

SUBMISSION_NAME="CSC14120_FinalProject_TeamName"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create clean directory
mkdir -p submission/$SUBMISSION_NAME

# Copy required files
cp -r src include CMakeLists.txt README.md submission/$SUBMISSION_NAME/
cp report.ipynb report.pdf submission/$SUBMISSION_NAME/
cp -r docs results models submission/$SUBMISSION_NAME/
cp team_plan.md presentation_video_link.txt submission/$SUBMISSION_NAME/

# Exclude unnecessary files
find submission/$SUBMISSION_NAME -name "*.o" -delete
find submission/$SUBMISSION_NAME -name ".DS_Store" -delete
rm -rf submission/$SUBMISSION_NAME/build

# Create archive
cd submission
zip -r ${SUBMISSION_NAME}_${TIMESTAMP}.zip $SUBMISSION_NAME

echo "Submission package created:"
echo "submission/${SUBMISSION_NAME}_${TIMESTAMP}.zip"
echo ""
echo "Size: $(du -sh ${SUBMISSION_NAME}_${TIMESTAMP}.zip | awk '{print $1}')"
```

### 11.2 Final Verification

```bash
# Extract and test submission package
cd /tmp
unzip path/to/submission.zip
cd CSC14120_FinalProject_TeamName
mkdir build && cd build
cmake .. && make -j4
./bin/test_data_loading
# Should work!
```

---

## 12. Day-of-Submission Checklist

**Morning (2 hours before deadline):**

- [ ] Run full test suite one last time
- [ ] Verify Colab notebook runs start-to-finish
- [ ] Check video link is accessible (Unlisted, not Private!)
- [ ] Spell-check report
- [ ] Review team plan and contribution percentages

**1 Hour Before:**

- [ ] Create submission archive
- [ ] Upload to submission portal
- [ ] Verify upload completed
- [ ] Submit confirmation received

**After Submission:**

- [ ] Save local backup
- [ ] Upload to Google Drive (backup)
- [ ] Send team confirmation email

---

**You're ready! Good luck with your submission!**
