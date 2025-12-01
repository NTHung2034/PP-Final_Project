# CUDA Autoencoder for CIFAR-10 Feature Learning

---

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Building the Project](#building-the-project)
- [Downloading the Dataset](#downloading-the-dataset)
- [Running the Project](#running-the-project)
- [Implementation Phases](#implementation-phases)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [References](#references)

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
- **CMake:** 3.18 or higher
- **C++ Compiler:** GCC 9+ / Clang 10+ / MSVC 2019+
- **CUDA Toolkit:** 11.0+ (optional, for GPU support)
- **OpenMP:** Required for CPU parallelization
- **LIBSVM:** For SVM classification (Phase 4)

### Windows-Specific
- MinGW-w64 with GCC or MSVC
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

# Configure (CPU-only)
cmake --build .

# Configure (with CUDA support)
cmake -DENABLE_CUDA=ON ..

```

### 4. Run Tests
```bash
# From build directory
./test_dataloader    # Linux/macOS
```

---

## Project Structure

```
PP-Final_Project/
├── CMakeLists.txt              # CMake build configuration
├── README.md                   # This file
│
├── include/                    # Header files
│   ├── config.h               # Project configuration constants
│   ├── data/
│   │   ├── cifar10_dataset.h  # CIFAR-10 dataset loader
│   │   ├── data_types.h       # Tensor and data type definitions
│   │   └── data_utils.h       # Data utility functions
│   └── utils/
│       ├── logger.h           # Logging utilities
│       └── memory_pool.h      # Memory pool for efficient allocation
│
├── src/                        # Source files
│   ├── main_train.cpp         # Main training entry point
│   ├── test_data.cpp          # Data loading tests
│   ├── data/
│   │   ├── cifar10_dataset.cpp
│   │   └── data_utils.cpp
│   └── utils/
│       ├── logger.cpp
│       └── memory_pool.cpp
│
├── data/                       # Dataset directory (download required)
│   ├── data_batch_1.bin       # Training batch 1 (10,000 images)
│   ├── data_batch_2.bin       # Training batch 2
│   ├── data_batch_3.bin       # Training batch 3
│   ├── data_batch_4.bin       # Training batch 4
│   ├── data_batch_5.bin       # Training batch 5
│   └── test_batch.bin         # Test batch (10,000 images)
│
├── models/
│   └── saved_weights/          # Trained model weights
│
├── scripts/
│   └── download_cifar10.sh    # Dataset download script
│
├── docs/                       # Documentation
│   ├── CSC14120_2025_Final_Project.md  # Full project requirements
│   ├── PROJECT_PLAN.md        # Development timeline
│   ├── PHASE_1_GUIDE.md       # CPU baseline guide
│   ├── PHASE_2_GUIDE.md       # GPU implementation guide
│   ├── PHASE_3_GUIDE.md       # Optimization guide
│   └── PHASE_4_GUIDE.md       # SVM integration guide
│
├── external/                   # External libraries (LIBSVM, etc.)
└── build/                      # Build output (generated)
```
