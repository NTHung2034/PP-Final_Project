# CIFAR-10 Autoencoder with CUDA Acceleration

**CSC14120 - Parallel Programming Final Project**

A high-performance implementation of an autoencoder-based feature learning system for CIFAR-10 image classification, progressively optimized from CPU baseline to GPU with >50× speedup.

---

## Project Overview

This project implements a two-stage machine learning pipeline:

1. **Unsupervised Feature Learning:** Train a convolutional autoencoder (GPU-accelerated) to learn 8,192-dimensional feature representations from CIFAR-10 images
2. **Supervised Classification:** Train an SVM classifier on the learned features to achieve 60-65% test accuracy

**Key Features:**

- Progressive optimization from CPU to GPU (4 phases)
- 50-70× speedup through CUDA optimization
- Complete end-to-end pipeline
- Comprehensive testing and documentation

---

## Prerequisites

### Hardware Requirements

**Local Development (Intel Core i5 Compatible):**

- **CPU:** Multi-core x86_64 processor (Intel Core i5/i7 or AMD Ryzen 5/7)
  - Phase 1 (CPU baseline) runs locally
- **GPU (Optional):** NVIDIA GPU with Compute Capability ≥6.0
  - Recommended: RTX 3060+, Tesla T4+, V100, or A100
  - Minimum: GTX 1660 or equivalent
  - **Don't have NVIDIA GPU?** Use Google Colab for Phases 2-4 (free GPU runtime)
- **RAM:** 16GB+ system memory (8GB minimum for Phase 1)
- **Storage:** 10GB free space

**⚠️ Intel Core i5 / Systems without NVIDIA GPU:**

- **Phase 1 (CPU Baseline):** Can run locally (expect 15-20 min/epoch)
- **Phase 2-4 (GPU Required):** **Must use Google Colab** (see [Google Colab Setup](#google-colab-setup))
- Intel integrated graphics are NOT supported for CUDA operations

**Google Colab (Recommended for systems without NVIDIA GPU):**

- Free tier: T4 GPU (15GB VRAM) - **Sufficient for this project**
- Pro tier: V100/A100 (faster training, optional)
- **Intel Core i5 users:** Google Colab is your primary option for GPU phases

### Software Requirements

**Operating System:**

- Linux (Ubuntu 20.04+ recommended)
- **Windows 10/11:** Native support with PowerShell (CUDA Toolkit required) or WSL2
- macOS (CPU only, no CUDA support)

**Windows 11 Users:**

- **Option 1 (Recommended):** Use native Windows with PowerShell (all commands below include PowerShell versions)
- **Option 2:** Use WSL2 Ubuntu (follow Linux commands)
- **Option 3:** Use Google Colab for GPU phases (no local CUDA installation needed)

**Core Dependencies:**

- **CUDA Toolkit:** 11.0+ (11.8 recommended)
  - Download: https://developer.nvidia.com/cuda-downloads
- **CMake:** 3.18+
  - **Linux/Ubuntu:** `sudo apt install cmake`
  - **Windows 11:** Download installer from https://cmake.org/download/ or use `winget install Kitware.CMake`
- **C++ Compiler:** GCC 9+ or Clang 10+ (Linux), MSVC 2019+ (Windows)
  - **Linux/Ubuntu:** `sudo apt install build-essential`
  - **Windows 11:** Install Visual Studio 2019/2022 with "Desktop development with C++" workload
- **Git:** For cloning repository
  - **Linux/Ubuntu:** `sudo apt install git`
  - **Windows 11:** Download from https://git-scm.com/ or `winget install Git.Git`

**Optional:**

- **Python 3.8+:** For result visualization
- **Nsight Compute/Systems:** For profiling
- **Valgrind:** For memory leak detection

---

## Quick Start

### 1. Clone Repository

**Linux/Ubuntu/macOS:**

```bash
git clone https://github.com/YOUR_USERNAME/PP-Final_Project.git
cd PP-Final_Project
```

**Windows 11 PowerShell:**

```powershell
git clone https://github.com/YOUR_USERNAME/PP-Final_Project.git
cd PP-Final_Project
```

### 2. Download CIFAR-10 Dataset

**Linux/Ubuntu/macOS:**

```bash
bash scripts/download_cifar10.sh
```

**Windows 11 PowerShell:**

```powershell
# Manual download (Windows script provided)
powershell -ExecutionPolicy Bypass -File scripts\download_cifar10.ps1

# Or download manually:
# 1. Download: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
# 2. Extract to: data/cifar-10-batches-bin/
```

### 3. Build and Run

**Method 1: Using VS Code (Recommended)**

1. Open the project folder in VS Code
2. Install the **CMake Tools** extension (ms-vscode.cmake-tools)
3. When prompted, select **GCC** compiler from `C:\msys64\ucrt64\bin\gcc.exe`
4. Click the **Build** button in the status bar (or press `F7`)
5. Wait for build to complete (~27 seconds)

**Method 2: Manual Build (PowerShell)**

```powershell
# Navigate to project root
cd your_folder\PP-Final_Project

# Step 1: Verify dataset exists
Test-Path data\cifar-10-batches-bin\data_batch_1.bin
# Output: True

# Step 2: Create and configure build directory
New-Item -ItemType Directory -Force -Path build
cd build

# Step 3: Configure CMake with MinGW Makefiles
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug `
  -DCMAKE_C_COMPILER=your_folder\ucrt64\bin\gcc.exe `
  -DCMAKE_CXX_COMPILER=your_folder\ucrt64\bin\g++.exe

# Step 4: Build project
cmake --build . --config Debug --target all -j 8

# Step 5: Return to project root
cd ..
```

**Expected Build Output:**

- ✅ `build\bin\train_autoencoder.exe` - Main training executable
- ✅ `build\bin\train_cpu.exe` - Legacy CPU training
- ✅ `build\bin\test_layers.exe` - Layer unit tests
- ✅ `build\bin\test_autoencoder.exe` - Integration tests

**Build time:** ~27 seconds (8 parallel jobs)

### 4. Run Training

```powershell
# Run training (creates models\saved_weights\ automatically)
.\build\bin\train_autoencoder.exe

# Verify output was created
Test-Path models\saved_weights\training_summary.txt
# Output: True
```

**Expected Output:**

```
Epoch 1: 0.895746
Epoch 2: 0.425848

--- Training Summary ---
Mode: TEST
Batch size: 32
Learning rate: 0.001000
Total images processed: 64
Total time: 173.21s
```

**Training time:** ~2.5-3 minutes (TEST_MODE with 64 images)

### 5. Run Tests

```powershell
# Test individual layers
.\build\bin\test_layers.exe

# Test complete autoencoder
.\build\bin\test_autoencoder.exe
```

**Test time:** ~10-15 seconds total

### 6. Visualize Results

Upload `models\saved_weights\training_summary.txt` to Google Colab:

1. Open `visualization.ipynb` in Google Colab
2. Upload `training_summary.txt` when prompted
3. Run all cells to see:
   - Loss progression plot
   - 52.46% loss reduction analysis
   - Performance statistics
   - Training assessment report

### 7. Incremental build (after first build, after code changes):\*\*

```powershell
cd build
cmake --build . --target all -j 8
cd ..
```

---

## Project Structure

```
PP-Final_Project\
├── build\                          # Build artifacts (created by CMake)
│   ├── bin\
│   │   ├── train_autoencoder.exe  # Main training executable
│   │   ├── train_cpu.exe          # Legacy CPU training
│   │   ├── test_layers.exe        # Layer unit tests
│   │   └── test_autoencoder.exe   # Integration tests
│   ├── CMakeFiles\                # CMake internal files
│   ├── CMakeCache.txt             # CMake configuration cache
│   ├── Makefile                   # MinGW Makefiles
│   └── compile_commands.json      # Compilation database
├── models\                         # Created on first training run
│   └── saved_weights\
│       ├── encoder_epoch_1.bin     # Epoch 1 weights (~3 MB)
│       ├── encoder_epoch_2.bin     # Epoch 2 weights (~3 MB)
│       └── training_summary.txt    # Training log
├── data\
│   └── cifar-10-batches-bin\       # CIFAR-10 dataset (required)
│       ├── data_batch_1.bin        # Training batch 1 (10,000 images)
│       ├── data_batch_2.bin        # Training batch 2
│       ├── data_batch_3.bin        # Training batch 3
│       ├── data_batch_4.bin        # Training batch 4
│       ├── data_batch_5.bin        # Training batch 5
│       ├── test_batch.bin          # Test set (10,000 images)
│       └── batches.meta.txt        # Label names
├── src\                            # Source code implementation
│   ├── main_train.cpp              # Main training entry point
│   ├── train_cpu.cpp               # Legacy training (same as main_train)
│   ├── data\
│   │   ├── cifar10_dataset.cpp     # CIFAR-10 data loading
│   │   └── data_utils.cpp          # Data preprocessing utilities
│   ├── layers\
│   │   ├── conv2d_cpu.cpp          # Convolutional layer
│   │   ├── relu_cpu.cpp            # ReLU activation
│   │   ├── maxpool_cpu.cpp         # Max pooling
│   │   └── upsample_cpu.cpp        # Upsampling (decoder)
│   ├── models\
│   │   └── autoencoder_cpu.cpp     # Complete autoencoder
│   └── utils\
│       ├── logger.cpp              # Training logger
│       └── memory_pool.cpp         # Memory management
├── include\                        # Header files
│   ├── config.h                    # Global configuration
│   ├── data\
│   │   ├── cifar10_dataset.h
│   │   ├── data_types.h
│   │   └── data_utils.h
│   ├── layers\
│   │   ├── conv2d_cpu.h
│   │   ├── relu_cpu.h
│   │   ├── maxpool_cpu.h
│   │   └── upsample_cpu.h
│   ├── models\
│   │   └── autoencoder_cpu.h
│   └── utils\
│       ├── logger.h
│       └── memory_pool.h
├── tests\
│   ├── test_layers.cpp             # Layer unit tests
│   └── test_autoencoder.cpp        # Autoencoder integration tests
├── docs\                           # Project documentation
│   ├── PHASE_1_GUIDE.md            # CPU baseline implementation
│   ├── PHASE_2_GUIDE.md            # Naive GPU implementation
│   ├── PHASE_3_GUIDE.md            # Optimized GPU (shared memory)
│   ├── PHASE_4_GUIDE.md            # Advanced optimization
│   └── TESTING_DELIVERABLES.md     # Testing requirements
├── scripts\
│   ├── build_cpu.ps1               # Automated build script
│   ├── download_cifar10.ps1        # Windows dataset download
│   └── download_cifar10.sh         # Linux dataset download
├── notebooks\
│   └── visualization.ipynb         # Google Colab analysis
├── CMakeLists.txt                  # CMake build configuration
└── README.md                       # This file
```

**Key Directories:**

- **`build\bin\`**: Compiled executables (Debug build by default)
- **`models\saved_weights\`**: Automatically created by training program
- **`data\cifar-10-batches-bin\`**: Must be downloaded before building

---

## Build Configuration

**Debug vs Release:**

```powershell
# Debug build (default, includes debug symbols)
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug

# Release build (optimized, faster execution)
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
```

**Enable CUDA (requires NVIDIA GPU + CUDA Toolkit):**

```powershell
cmake .. -G "MinGW Makefiles" -DENABLE_CUDA=ON
```

## Troubleshooting Build Issues

**Error: "Generator Visual Studio 17 2022 could not find any instance of Visual Studio"**

**Solution:** Use MinGW Makefiles instead:

```powershell
cmake .. -G "MinGW Makefiles"
```

**Error: "No CMAKE_CXX_COMPILER could be found"**

**Solution:** Install MSYS2 and specify compiler:

```powershell
# Install MSYS2 from https://www.msys2.org/
# Then install GCC:
pacman -S mingw-w64-ucrt-x86_64-gcc

# Specify compiler in CMake:
cmake .. -G "MinGW Makefiles" `
  -DCMAKE_CXX_COMPILER=C:\msys64\ucrt64\bin\g++.exe
```

**Error: "OpenMP not found"**

**Solution:** OpenMP is included with MinGW GCC, ensure you're using GCC from MSYS2.

**Build succeeds but executables crash:**

**Solution:** Verify CIFAR-10 dataset exists:

```powershell
Test-Path data\cifar-10-batches-bin\data_batch_1.bin
# Should return: True
```

---
