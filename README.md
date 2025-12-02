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

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/NTHung2034/PP-Final_Project.git
cd PP-Final_Project
```

### 2. Download CIFAR-10 Dataset
 
```bash
bash ./scripts/download_cifar10.sh
```

### 3. Build the Project

```bash
# Create build directory
mkdir build
cd build

# --In local--
# compile CPU only
cmake --build .

# --On Google colab--
# compile CPU only
!cmake ..
!make
```

### 4. Run Tests (build directory)
```bash
# Test dataloader
./test_dataloader   

# Test cpu layers
./test_layers

# Test autoencoder 
./test_autoencoder
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
│   │   └── data_utils.h           # Data preprocessing utilities
│   ├── layers/
│   │   ├── conv2d_cpu.h           # 2D Convolution layer (CPU)
│   │   ├── maxpool_cpu.h          # Max Pooling layer (CPU)
│   │   ├── relu_cpu.h             # ReLU activation layer (CPU)
│   │   └── upsample_cpu.h         # Upsampling layer (CPU)
│   ├── models/
│   │   └── autoencoder_cpu.h      # Autoencoder model (CPU)
│   └── utils/
│       ├── logger.h               # Logging utilities
│       └── memory_pool.h          # Memory pool for efficient allocation
│
├── src/                           # Source implementations
│   ├── train_cpu.cpp              # CPU training entry point
│   ├── data/
│   │   ├── cifar10_dataset.cpp    # Dataset loader implementation
│   │   ├── data_utils.cpp         # Data utilities implementation
│   │   └── README.md              # Data module documentation
│   ├── layers/
│   │   ├── conv2d_cpu.cpp         # Conv2D forward/backward pass
│   │   ├── maxpool_cpu.cpp        # MaxPool forward/backward pass
│   │   ├── relu_cpu.cpp           # ReLU forward/backward pass
│   │   └── upsample_cpu.cpp       # Upsample forward/backward pass
│   ├── models/
│   │   └── autoencoder_cpu.cpp    # Autoencoder training logic
│   └── utils/
│       ├── logger.cpp             # Logger implementation
│       └── memory_pool.cpp        # Memory pool implementation
│
├── tests/                         # Unit tests
│   ├── test_dataloader.cpp        # Tests for data loading
│   ├── test_layers.cpp            # Tests for neural network layers
│   └── test_autoencoder.cpp       # Tests for autoencoder model
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
│   ├── CSC14120_2025_Final Project.pdf  # Project requirements (PDF)
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
    └── bin/                       # Compiled executables
        ├── train_cpu.exe          # Main training executable
        ├── test_dataloader.exe    # Data loader tests
        ├── test_layers.exe        # Layer tests
        └── test_autoencoder.exe   # Autoencoder tests
``` build/                      # Build output (generated)
```


