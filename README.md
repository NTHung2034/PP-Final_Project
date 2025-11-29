# CIFAR-10 Autoencoder with CUDA Acceleration

**CSC14120 - Parallel Programming Final Project**

A high-performance implementation of an autoencoder-based feature learning system for CIFAR-10 image classification, progressively optimized from CPU baseline to GPU with >50× speedup.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Google Colab Setup](#google-colab-setup)
7. [Project Structure](#project-structure)
8. [Performance Targets](#performance-targets)
9. [Documentation](#documentation)
10. [Troubleshooting](#troubleshooting)

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

**Local Development:**

- **CPU:** Multi-core x86_64 processor
- **GPU:** NVIDIA GPU with Compute Capability ≥6.0
  - Recommended: RTX 3060+, Tesla T4+, V100, or A100
  - Minimum: GTX 1660 or equivalent
- **RAM:** 16GB+ system memory
- **Storage:** 10GB free space

**Google Colab:**

- Free tier: T4 GPU (15GB VRAM)
- Pro tier: V100/A100 (faster training)

### Software Requirements

**Operating System:**

- Linux (Ubuntu 20.04+ recommended)
- Windows 10/11 with WSL2
- macOS (CPU only, no CUDA support)

**Core Dependencies:**

- **CUDA Toolkit:** 11.0+ (11.8 recommended)
  - Download: https://developer.nvidia.com/cuda-downloads
- **CMake:** 3.18+
  - Install: `sudo apt install cmake` (Ubuntu)
- **C++ Compiler:** GCC 9+ or Clang 10+
  - Install: `sudo apt install build-essential`
- **Git:** For cloning repository
  - Install: `sudo apt install git`

**Optional:**

- **Python 3.8+:** For result visualization
- **Nsight Compute/Systems:** For profiling
- **Valgrind:** For memory leak detection

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/PP-Final_Project.git
cd PP-Final_Project
```

### 2. Download CIFAR-10 Dataset

```bash
cd scripts && chmod +x download_cifar10.sh && ./download_cifar10.sh && cd ..
```

### 3. Build Project (Phase 1 - CPU Only)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 4. Run Tests

```bash
# Test CPU layers (should pass all 23 tests)
./bin/test_cpu_layers

# Test data loading (requires CIFAR-10 dataset)
./bin/test_data_loading
```

### 5. Train Autoencoder (Phase 1 - CPU)

```bash
# Quick test (2 epochs)
./bin/train_cpu --epochs 2 --batch-size 32

# Full training (20 epochs, ~15-20 min/epoch on CPU)
./bin/train_cpu --epochs 20 --batch-size 32
```

### 6. Future Phases (GPU)

```bash
# Build with CUDA support (Phase 2+)
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
make -j$(nproc)

# Phase 2-3: Train GPU optimized autoencoder
./bin/train_gpu --epochs 20

# Phase 4: Extract features and train SVM
./bin/extract_features
bash ../scripts/train_svm.sh
bash ../scripts/predict_svm.sh
```

**Expected Runtime:**

| Phase | Component | Time |
|-------|-----------|------|
| Phase 1 | CPU Training (20 epochs) | ~5-7 hours |
| Phase 2 | GPU Naive (20 epochs) | ~40-60 min |
| Phase 3 | GPU Optimized (20 epochs) | ~5-10 min |
| Phase 4 | Feature extraction + SVM | ~5 min |

---

## Installation

### Step-by-Step Setup (Local)

#### 1. Install CUDA Toolkit

**Ubuntu/Debian:**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
```

**Verify installation:**

```bash
nvcc --version
nvidia-smi
```

#### 2. Install Build Tools

```bash
sudo apt update
sudo apt install -y \
    cmake \
    build-essential \
    git \
    libomp-dev \
    python3-pip \
    wget \
    unzip
```

#### 3. Install Python Dependencies (Optional)

```bash
pip3 install numpy matplotlib seaborn scikit-learn jupyter
```

#### 4. Clone and Setup Project

```bash
git clone https://github.com/YOUR_USERNAME/PP-Final_Project.git
cd PP-Final_Project

# Download CIFAR-10
bash scripts/download_cifar10.sh

# Download LIBSVM
cd external
git clone https://github.com/cjlin1/libsvm.git
cd libsvm && make
cd ../..
```

#### 5. Build Project

```bash
mkdir build && cd build

# Configure (specify GPU architecture if needed)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75  # 75=T4, 70=V100, 80=A100

# Build
make -j$(nproc)

# Verify
./bin/check_gpu
```

**Expected output:**

```
✓ CUDA available
✓ Device 0: Tesla T4
✓ Compute Capability: 7.5
✓ Total Memory: 15 GB
```

---

## Usage

### Training Commands

#### Phase 1: CPU Baseline (Optional)

```bash
cd build

# Train CPU version (2 epochs for testing)
./bin/train_cpu --epochs 2 --batch-size 32

# Expected: ~15-20 min/epoch
```

#### Phase 2-3: GPU Optimized Training

```bash
# Train GPU version (optimized)
./bin/train_gpu --epochs 20 --batch-size 64

# Expected: ~20-30 sec/epoch
# Output: models/saved_weights/gpu_encoder_weights.bin
```

**Training Options:**

- `--epochs N`: Number of training epochs (default: 20)
- `--batch-size N`: Batch size (default: 64 for GPU, 32 for CPU)
- `--learning-rate F`: Learning rate (default: 0.001)
- `--save-interval N`: Save weights every N epochs
- `--profile`: Enable profiling output

### Feature Extraction

```bash
# Extract 8,192-dim features for all images
./bin/extract_features

# Output:
#   data/train_features.txt (50,000 samples)
#   data/test_features.txt (10,000 samples)

# Expected: ~15-20 seconds
```

### SVM Training and Evaluation

```bash
# Train SVM on extracted features
bash scripts/train_svm.sh
# Output: models/svm_model.bin
# Expected: 2-5 minutes

# Predict on test set
bash scripts/predict_svm.sh
# Output: results/test_predictions.txt
# Shows: Accuracy = 62.34% (6234/10000)

# Generate detailed evaluation
python scripts/evaluate_results.py
# Output:
#   results/confusion_matrix.png
#   results/evaluation_summary.txt
#   results/tsne_features.png (if enabled)
```

### Benchmarking

```bash
# Run performance benchmark
bash scripts/benchmark_all.sh

# Compare all versions
./bin/benchmark_versions

# Output: Performance comparison table
```

---

## Google Colab Setup

### One-Click Setup

**Open in Colab:** [Launch Notebook](https://colab.research.google.com/)

**Cell 1: Setup Environment**

```python
# Enable GPU runtime first!
# Runtime → Change runtime type → GPU (T4/V100/A100)

# Verify GPU
!nvidia-smi

# Clone repository
!git clone https://github.com/YOUR_USERNAME/PP-Final_Project.git
%cd PP-Final_Project

# Install dependencies
!apt-get update
!apt-get install -y cmake build-essential libomp-dev

# Download CIFAR-10
!bash scripts/download_cifar10.sh

# Setup LIBSVM
!cd external && git clone https://github.com/cjlin1/libsvm.git
!cd external/libsvm && make
```

**Cell 2: Build Project**

```python
# Create build directory
!mkdir -p build
%cd build

# Configure for Colab GPU (usually T4 = Compute Capability 7.5)
!cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75

# Build (use 2 cores to avoid timeout)
!make -j2
```

**Cell 3: Run Full Pipeline**

```python
# Train autoencoder
!./bin/train_gpu --epochs 20

# Extract features
!./bin/extract_features

# Train SVM
!cd .. && bash scripts/train_svm.sh
!bash scripts/predict_svm.sh

# Evaluate
!python scripts/evaluate_results.py
```

**Cell 4: View Results**

```python
from IPython.display import Image, display

# Display confusion matrix
display(Image('../results/confusion_matrix.png'))

# Print accuracy
with open('../results/evaluation_summary.txt', 'r') as f:
    print(f.read())
```

**Cell 5: Save to Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')

# Save results
!mkdir -p /content/drive/MyDrive/CIFAR10_Results
!cp -r ../results/* /content/drive/MyDrive/CIFAR10_Results/
!cp -r ../models/saved_weights/* /content/drive/MyDrive/CIFAR10_Results/

print("Results saved to Google Drive!")
```

### Colab-Specific Notes

**Runtime Limits:**

- Free tier: 12 hours max session, may disconnect
- Pro tier: 24 hours, better GPU availability

**Memory Management:**

- T4 GPU: 15GB VRAM (sufficient for batch_size=64)
- If OOM error: Reduce batch size to 32

**Persistence:**

- Save checkpoints to Google Drive every few epochs
- Download model weights before session ends

**Common Issues:**

- **"CUDA out of memory":** Reduce batch size
- **"Session disconnected":** Enable background execution
- **"No GPU available":** Reset runtime or wait

---

## Project Structure

```
PP-Final_Project/
├── CMakeLists.txt              # Build configuration (CUDA optional for Phase 1)
├── README.md                   # This file
├── .gitignore                  # Git ignore patterns
│
├── docs/                       # Documentation
│   ├── PROJECT_PLAN.md         # Comprehensive project timeline
│   ├── PHASE_1_GUIDE.md        # CPU implementation guide
│   ├── PHASE_1_IMPLEMENTATION.md # Phase 1 implementation details (NEW)
│   ├── PHASE_2_GUIDE.md        # GPU porting guide
│   ├── PHASE_3_GUIDE.md        # GPU optimization guide
│   ├── PHASE_4_GUIDE.md        # SVM integration guide
│   ├── TESTING_DELIVERABLES.md # Testing and submission guide
│   └── CSC14120_2025_Final_Project.md  # Official requirements
│
├── include/                    # Header files
│   ├── config.h                # Global configuration constants
│   │
│   ├── data/                   # Data handling headers
│   │   ├── cifar10_dataset.h   # CIFAR-10 data loader class
│   │   ├── data_types.h        # Tensor struct definition
│   │   └── data_utils.h        # Preprocessing utilities
│   │
│   ├── layers/                 # Neural network layers (CPU) - Phase 1
│   │   ├── conv2d_cpu.h        # 2D Convolution layer
│   │   ├── relu_cpu.h          # ReLU activation
│   │   ├── maxpool_cpu.h       # Max pooling layer
│   │   ├── upsample_cpu.h      # Nearest neighbor upsampling
│   │   └── loss_functions.h    # MSE loss function
│   │
│   ├── models/                 # Model definitions
│   │   └── autoencoder_cpu.h   # CPU Autoencoder (751,875 params)
│   │
│   └── utils/                  # Utility classes
│       ├── logger.h            # Logging macros
│       └── memory_pool.h       # Memory pool for efficiency
│
├── src/                        # Source files
│   ├── data/                   # Data module implementation
│   │   ├── cifar10_dataset.cpp # CIFAR-10 binary loader
│   │   ├── data_utils.cpp      # Preprocessing functions
│   │   └── README.md           # Data module documentation
│   │
│   ├── layers/                 # CPU layer implementations - Phase 1
│   │   ├── conv2d_cpu.cpp      # Forward/backward convolution
│   │   ├── relu_cpu.cpp        # ReLU forward/backward
│   │   ├── maxpool_cpu.cpp     # MaxPool with index tracking
│   │   ├── upsample_cpu.cpp    # Nearest neighbor upsample
│   │   └── loss_functions.cpp  # MSE computation
│   │
│   ├── models/                 # Model implementations
│   │   └── autoencoder_cpu.cpp # Encoder-decoder network
│   │
│   ├── utils/                  # Utility implementations
│   │   ├── logger.cpp          # Logging functions
│   │   └── memory_pool.cpp     # Memory pool
│   │
│   ├── train_cpu.cpp           # CPU training entry point (Phase 1)
│   └── main_train.cpp          # Legacy training entry
│
├── tests/                      # Test files
│   ├── test_data_loading.cpp   # Data loading tests
│   └── test_cpu_layers.cpp     # CPU layer tests (23 tests)
│
├── scripts/                    # Helper scripts
│   └── download_cifar10.sh     # CIFAR-10 dataset download
│
├── data/                       # Dataset directory (created by build)
│   └── cifar-10-batches-bin/   # CIFAR-10 binary files
│
├── models/                     # Model weights directory
│   └── saved_weights/          # Trained model weights (.bin)
│
├── results/                    # Evaluation results
│
└── external/                   # External libraries
    └── libsvm/                 # LIBSVM (Phase 4)
```

### Implementation Status

| Component | Status | Tests |
|-----------|--------|-------|
| Data Loading | ✅ Complete | 7 tests |
| Conv2D Layer | ✅ Complete | 2 tests |
| ReLU Layer | ✅ Complete | 3 tests |
| MaxPool Layer | ✅ Complete | 2 tests |
| Upsample Layer | ✅ Complete | 3 tests |
| MSE Loss | ✅ Complete | 4 tests |
| Autoencoder | ✅ Complete | 4 tests |
| Training Loop | ✅ Complete | - |
| Weight I/O | ✅ Complete | 3 tests |
| **Total** | **Phase 1 Complete** | **23 tests passing** |

---

## Performance Targets

| Metric                              | Target      | Actual (Expected) |
| ----------------------------------- | ----------- | ----------------- |
| **Training Time** (20 epochs)       | <10 minutes | ~5-8 minutes      |
| **Feature Extraction** (60K images) | <20 seconds | ~15 seconds       |
| **GPU Speedup** vs CPU              | >20×        | 50-70×            |
| **Test Accuracy**                   | 60-65%      | 62-64%            |
| **GPU Memory Usage**                | <4GB        | ~2.5GB            |

### Performance Progression

| Implementation      | Time/Epoch | Speedup | Status               |
| ------------------- | ---------- | ------- | -------------------- |
| CPU Baseline        | 18 min     | 1×      | Phase 1              |
| GPU Naive           | 2.5 min    | 7×      | Phase 2              |
| GPU + Shared Memory | 45 sec     | 24×     | Phase 3.1            |
| GPU + Kernel Fusion | 28 sec     | 38×     | Phase 3.2            |
| GPU + Multi-Stream  | 20 sec     | 54×     | Phase 3.3 (Optional) |

---

## Documentation

### Implementation Guides

- **[PROJECT_PLAN.md](docs/PROJECT_PLAN.md):** Complete project timeline and milestones
- **[PHASE_1_GUIDE.md](docs/PHASE_1_GUIDE.md):** CPU baseline implementation guide
- **[PHASE_1_IMPLEMENTATION.md](docs/PHASE_1_IMPLEMENTATION.md):** Phase 1 implementation details ✅ Complete
- **[PHASE_2_GUIDE.md](docs/PHASE_2_GUIDE.md):** GPU porting and naive kernels
- **[PHASE_3_GUIDE.md](docs/PHASE_3_GUIDE.md):** Advanced GPU optimizations
- **[PHASE_4_GUIDE.md](docs/PHASE_4_GUIDE.md):** SVM integration and evaluation
- **[TESTING_DELIVERABLES.md](docs/TESTING_DELIVERABLES.md):** Testing and submission

### External References

**Autoencoders:**

- Hinton & Salakhutdinov (2006): https://www.cs.toronto.edu/~hinton/science.pdf
- Deep Learning Book Ch 14: https://www.deeplearningbook.org/contents/autoencoders.html

**CUDA Programming:**

- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

**Reference Implementations:**

- turkdogan/autoencoder: https://github.com/turkdogan/autoencoder
- tbennun/cudnn-training: https://github.com/tbennun/cudnn-training

**LIBSVM:**

- Official Guide: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
- GitHub: https://github.com/cjlin1/libsvm

---

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found

**Error:** `CMake Error: Could not find CUDA`

**Solution:**

```bash
# Check CUDA installation
which nvcc
nvidia-smi

# Add to PATH (if needed)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or specify in CMake
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

#### 2. GPU Out of Memory

**Error:** `CUDA error: out of memory`

**Solutions:**

```bash
# Reduce batch size
./bin/train_gpu --batch-size 32  # Instead of 64

# Clear GPU memory
nvidia-smi --gpu-reset

# Check GPU usage
nvidia-smi
```

#### 3. Wrong CUDA Architecture

**Error:** `no kernel image is available for execution on the device`

**Solution:**

```bash
# Find your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Rebuild with correct architecture
# T4: 75, V100: 70, A100: 80
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
make clean && make
```

#### 4. Compilation Errors

**Error:** `undefined reference to cudaMalloc`

**Solution:**

```bash
# Ensure CUDA is linked
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Or edit CMakeLists.txt to add:
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(target CUDA::cudart)
```

#### 5. CIFAR-10 Download Fails

**Solution:**

```bash
# Manual download
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz -C data/
```

#### 6. Low Accuracy (<55%)

**Possible causes:**

- Autoencoder not trained enough → Increase epochs
- Features not normalized → Try `svm-scale`
- Poor hyperparameters → Run grid search

**Debug:**

```bash
# Check reconstruction quality
./bin/visualize_reconstructions

# Check training loss curve
grep "Loss:" training.log
```

### Getting Help

**Resources:**

1. Check [documentation](docs/) for detailed guides
2. Review [TESTING_DELIVERABLES.md](docs/TESTING_DELIVERABLES.md) for common issues
3. Search GitHub issues
4. Contact instructor during office hours

---

## Running Commands Summary

### Complete Pipeline (One-Shot)

```bash
# Setup (once)
git clone https://github.com/YOUR_USERNAME/PP-Final_Project.git
cd PP-Final_Project
bash scripts/download_cifar10.sh
mkdir build && cd build
cmake .. && make -j4

# Run (each time)
./bin/train_gpu --epochs 20          # ~8 minutes
./bin/extract_features                # ~15 seconds
bash ../scripts/train_svm.sh          # ~3 minutes
bash ../scripts/predict_svm.sh        # ~5 seconds
python ../scripts/evaluate_results.py # ~10 seconds

# Total: ~12-15 minutes
```

### Individual Phases

```bash
# Phase 1: CPU Baseline (optional)
./bin/train_cpu --epochs 2

# Phase 2: GPU Naive
./bin/train_gpu_naive --epochs 2

# Phase 3: GPU Optimized (recommended)
./bin/train_gpu --epochs 20

# Phase 4: SVM
./bin/extract_features
bash ../scripts/train_svm.sh
bash ../scripts/predict_svm.sh
```

### Testing

```bash
# Unit tests
./bin/unit_tests

# Integration test
bash ../tests/integration/test_full_pipeline.sh

# Performance benchmark
bash ../scripts/benchmark_all.sh

# Memory check
cuda-memcheck ./bin/train_gpu --epochs 1
```

---

## License

This project is for educational purposes as part of CSC14120 - Parallel Programming course.

---

## Authors

**Team:** [Your Team Name]

- Member 1: [Name] - [Contribution]
- Member 2: [Name] - [Contribution]
- Member 3: [Name] - [Contribution]

**Course:** CSC14120 - Parallel Programming  
**Institution:** University of Science, VNU-HCM  
**Semester:** 2025

---

## Acknowledgments

- CIFAR-10 dataset: Alex Krizhevsky, Geoffrey Hinton
- LIBSVM library: Chih-Chung Chang, Chih-Jen Lin
- Reference implementations: turkdogan/autoencoder, tbennun/cudnn-training
- Course instructors and TAs

---

**For detailed implementation guides, see [docs/](docs/) directory.**
