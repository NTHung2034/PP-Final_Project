# CUDA Autoencoder for CIFAR-10 Feature Learning

---

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

---

## Requirements

### Hardware
- **CPU:** Multi-core x86_64 processor
- **GPU:** NVIDIA GPU with CUDA Compute Capability ≥ 6.0 (for GPU version)
  - Recommended: RTX 3060+ or T4+
  - Minimum: GTX 1660 or equivalent
- **RAM:** 16GB+ recommended
- **Storage:** 10GB free space

### Software
- **CMake:** 3.18 or higher                           ****IMPORTANT***
- **C++ Compiler:** GCC 9+ / Clang 10+ / MSVC 2019+   ****IMPORTANT***
- **CUDA Toolkit:** 11.0+ (optional, for GPU support)
- **OpenMP:** Required for CPU parallelization
- **LIBSVM:** For SVM classification (Phase 4)

### Windows-Specific
- MinGW-w64 with GCC or MSVC                          ****IMPORTANT***
- Windows PowerShell or Git Bash

---

## Quick Start - **SHOULD USE GOOGLE COLAB**

### 1. Clone the Repository
```bash
!git clone https://github.com/NTHung2034/PP-Final_Project.git
%cd PP-Final_Project
```

### 2. Download CIFAR-10 Dataset (root directory)
 
```bash
!bash ./scripts/download_cifar10.sh
```

### 3. Build the Project

```bash
# Create build directory
!mkdir build
%cd build
```
If you choose to build project in local

```bash
# compile CPU code only
cmake --build .
```

If you choose **Google colab**
```bash
# include CPU code only
!cmake ..

# or else (include all)
!cmake .. -DENABLE_CUDA=ON

# compile 
!make
```

### 4. Run Tests (build directory)
```bash
# Test dataloader
./test_dataloader   

# Test cpu layers
./test_layers

# Test cpu autoencoder 
./test_autoencoder

# Test gpu layers
./test_layers_gpu

# Test gpu autoencoder
./test_autoencoder_gpu
```

### 5. Train with CPU
```
# Train with cpu (2 epochs)
./train_cpu
```

---
## Project Structure

```
PP-Final_Project/
│
├── CMakeLists.txt                 # CMake build configuration
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules
│
├── include/                       # Header files
│   ├── config.h                   # Global configuration constants
│   ├── data/
│   │   ├── cifar10_dataset.h      # CIFAR-10 dataset loader class
│   │   ├── data_types.h           # Tensor struct and data type definitions
│   │   ├── data_utils.h           # Data preprocessing utilities
│   │   └── gpu_data_types.cuh     # GPU tensor and weight structures
│   ├── layers/
│   │   ├── conv2d_cpu.h           # 2D Convolution layer (CPU)
│   │   ├── maxpool_cpu.h          # Max Pooling layer (CPU)
│   │   ├── relu_cpu.h             # ReLU activation layer (CPU)
│   │   └── upsample_cpu.h         # Upsampling layer (CPU)
│   ├── layers_gpu/
│   │   ├── conv2d_gpu.cuh         # 2D Convolution layer (GPU)
│   │   ├── maxpool_gpu.cuh        # Max Pooling layer (GPU)
│   │   ├── relu_gpu.cuh           # ReLU activation layer (GPU)
│   │   ├── upsample_gpu.cuh       # Upsampling layer (GPU)
│   │   └── mse_loss_gpu.cuh       # MSE Loss layer (GPU)
│   ├── models/
│   │   ├── autoencoder_cpu.h      # Autoencoder model (CPU)
│   │   └── autoencoder_gpu.cuh    # Autoencoder model (GPU)
│   └── utils/
│       ├── logger.h               # Logging utilities
│       └── memory_pool.h          # Memory pool for efficient allocation
│
├── src/                           # Source implementations
│   ├── train_cpu.cpp              # CPU training entry point
│   ├── train_gpu.cu               # GPU training entry point
│   ├── data/
│   │   ├── cifar10_dataset.cpp    # Dataset loader implementation
│   │   ├── data_utils.cpp         # Data utilities implementation
│   │   └── README.md              # Data module documentation
│   ├── layers/
│   │   ├── conv2d_cpu.cpp         # Conv2D forward/backward pass (CPU)
│   │   ├── maxpool_cpu.cpp        # MaxPool forward/backward pass (CPU)
│   │   ├── relu_cpu.cpp           # ReLU forward/backward pass (CPU)
│   │   └── upsample_cpu.cpp       # Upsample forward/backward pass (CPU)
│   ├── layers_gpu/
│   │   ├── conv2d_gpu.cu          # Conv2D forward/backward pass (GPU)
│   │   ├── maxpool_gpu.cu         # MaxPool forward/backward pass (GPU)
│   │   ├── relu_gpu.cu            # ReLU forward/backward pass (GPU)
│   │   ├── upsample_gpu.cu        # Upsample forward/backward pass (GPU)
│   │   └── mse_loss_gpu.cu        # MSE Loss forward/backward pass (GPU)
│   ├── models/
│   │   ├── autoencoder_cpu.cpp    # Autoencoder training logic (CPU)
│   │   └── autoencode_gpu.cu      # Autoencoder training logic (GPU)
│   └── utils/
│       ├── logger.cpp             # Logger implementation
│       └── memory_pool.cpp        # Memory pool implementation
│
├── tests/                         # Unit tests
│   ├── test_dataloader.cpp        # Tests for data loading
│   ├── test_layers.cpp            # Tests for neural network layers (CPU)
│   ├── test_autoencoder.cpp       # Tests for autoencoder model (CPU)
│   ├── test_layers_gpu.cu         # Tests for neural network layers (GPU)
│   └── test_autoencoder_gpu.cu    # Tests for autoencoder model (GPU)
│
├── data/                          # CIFAR-10 dataset (run download script)
│   ├── data_batch_1.bin           # Training batch 1 (10,000 images)
│   ├── data_batch_2.bin           # Training batch 2
│   ├── data_batch_3.bin           # Training batch 3
│   ├── data_batch_4.bin           # Training batch 4
│   ├── data_batch_5.bin           # Training batch 5
│   ├── test_batch.bin             # Test batch (10,000 images)
│   ├── batches.meta.txt           # Class labels metadata
│   └── readme.html                # Dataset documentation
│
├── models/
│   └── saved_weights/             # Trained model weights output
│
├── notebooks/
│   └── visualization.ipynb        # Jupyter notebook for visualizations
│
├── scripts/
│   └── download_cifar10.sh        # Script to download CIFAR-10 dataset
│
├── docs/                          # Project documentation
│   ├── CSC14120_2025_Final_Project.md   # Full project requirements
│   ├── PROJECT_PLAN.md            # Development timeline & milestones
│   ├── PHASE_1_GUIDE.md           # Phase 1: CPU baseline guide
│   ├── PHASE_2_GUIDE.md           # Phase 2: GPU implementation guide
│   ├── PHASE_3_GUIDE.md           # Phase 3: Optimization guide
│   ├── PHASE_4_GUIDE.md           # Phase 4: SVM integration guide
│   └── TESTING_DELIVERABLES.md    # Testing requirements
│
├── external/                      # External libraries (LIBSVM, etc.)
│
└── build/                         # Build output directory (generated)
    ├── test_dataloader            # Data loader tests
    ├── test_layers                # Layer tests (CPU)
    ├── test_autoencoder           # Autoencoder tests (CPU)
    ├── test_layers_gpu            # Layer tests (GPU)
    ├── test_autoencoder_gpu       # Autoencoder tests (GPU)
    ├── train_cpu                  # CPU training executable
    └── train_gpu                  # GPU training executable
```


