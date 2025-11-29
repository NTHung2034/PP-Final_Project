# CIFAR-10 Autoencoder with CUDA Acceleration

**CSC14120 - Parallel Programming Final Project**

A high-performance implementation of an autoencoder-based feature learning system for CIFAR-10 image classification, progressively optimized from CPU baseline to GPU with >50× speedup.

---

## 🚀 Quick Start for Intel Core i5 (Windows 11)

**Don't have an NVIDIA GPU? No problem!** You can still work on this project:

### Step 1: Install Dependencies

```powershell
# Run automated setup script
.\scripts\setup_windows.ps1
```

### Step 2: Download CIFAR-10 Dataset

```powershell
# Download and extract CIFAR-10 (162 MB)
.\scripts\download_cifar10.ps1
```

### Step 3: Build CPU Version (Phase 1)

```powershell
# Build CPU-only version (no CUDA required)
.\scripts\build_cpu.ps1
```

### Step 4: Run Training

```powershell
# Run Phase 1 CPU baseline
.\build\bin\Release\train_autoencoder.exe
```

### Step 5: GPU Phases (Use Google Colab)

For Phases 2-4 (GPU optimization), use the free GPU runtime in Google Colab:

- See [Google Colab Setup](#google-colab-setup) section below
- All GPU code runs in the cloud - no local NVIDIA GPU needed!

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

### 3. Build Project

**Linux/Ubuntu/macOS:**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Windows 11 PowerShell:**

```powershell
New-Item -ItemType Directory -Force -Path build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release --parallel
```

### 4. Run Complete Pipeline

**Linux/Ubuntu/macOS:**

```bash
# Phase 1: Train CPU baseline (optional, for comparison)
./bin/train_cpu --epochs 2

# Phase 2-3: Train GPU optimized autoencoder
./bin/train_gpu --epochs 20

# Phase 4: Extract features and train SVM
./bin/extract_features
bash ../scripts/train_svm.sh
bash ../scripts/predict_svm.sh

# View results
python ../scripts/evaluate_results.py
```

**Windows 11 PowerShell:**

```powershell
# Phase 1: Train CPU baseline (optional, for comparison)
.\bin\Release\train_cpu.exe --epochs 2

# Phase 2-3: Train GPU optimized autoencoder (requires NVIDIA GPU)
.\bin\Release\train_gpu.exe --epochs 20

# Phase 4: Extract features and train SVM
.\bin\Release\extract_features.exe
powershell -ExecutionPolicy Bypass -File ..\scripts\train_svm.ps1
powershell -ExecutionPolicy Bypass -File ..\scripts\predict_svm.ps1

# View results
python ..\scripts\evaluate_results.py
```

**⚠️ No NVIDIA GPU? Use Google Colab for Phases 2-4:**

```powershell
# See "Google Colab Setup" section below for complete instructions
# You can run Phase 1 (CPU baseline) locally on Intel Core i5
```

**Expected Runtime (with optimized GPU):**

- GPU training: ~5-10 minutes (20 epochs)
- Feature extraction: ~15 seconds
- SVM training: ~3 minutes
- **Total:** ~10-15 minutes

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

**Windows 11:**

```powershell
# Download CUDA Toolkit 11.8 installer
# Visit: https://developer.nvidia.com/cuda-11-8-0-download-archive
# Select: Windows > x86_64 > 11 > exe (network)

# Run installer (requires NVIDIA GPU)
# Default installation path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

# Add to PATH (automatic if you check "Add to PATH" during installation)
# Or manually:
$env:Path += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
```

**⚠️ Intel Core i5 without NVIDIA GPU:**

```powershell
# CUDA installation will fail - this is expected
# Skip to "Google Colab Setup" section for GPU phases
# You can still run Phase 1 (CPU baseline) without CUDA
```

**Verify installation:**

**Linux:**

```bash
nvcc --version
nvidia-smi
```

**Windows 11 PowerShell:**

```powershell
nvcc --version
nvidia-smi
```

#### 2. Install Build Tools

**Linux/Ubuntu:**

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

**Windows 11 PowerShell:**

```powershell
# Install Visual Studio 2022 Build Tools (or full IDE)
winget install Microsoft.VisualStudio.2022.BuildTools

# Or download manually from:
# https://visualstudio.microsoft.com/downloads/
# Select "Desktop development with C++" workload during installation

# Install CMake
winget install Kitware.CMake

# Install Git
winget install Git.Git

# Install Python (for result visualization)
winget install Python.Python.3.11

# Verify installations
git --version
cmake --version
python --version
```

#### 3. Install Python Dependencies (Optional)

**Linux/Ubuntu:**

```bash
pip3 install numpy matplotlib seaborn scikit-learn jupyter
```

**Windows 11 PowerShell:**

```powershell
pip install numpy matplotlib seaborn scikit-learn jupyter
```

#### 4. Clone and Setup Project

**Linux/Ubuntu:**

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

**Windows 11 PowerShell:**

```powershell
git clone https://github.com/YOUR_USERNAME/PP-Final_Project.git
cd PP-Final_Project

# Download CIFAR-10
powershell -ExecutionPolicy Bypass -File scripts\download_cifar10.ps1

# Download LIBSVM (pre-compiled for Windows)
cd external
git clone https://github.com/cjlin1/libsvm.git
cd libsvm
# For Windows, use pre-compiled binaries from:
# https://www.csie.ntu.edu.tw/~cjlin/libsvm/
# Or build with Visual Studio (open windows/ folder)
cd ..\..
```

#### 5. Build Project

**Linux/Ubuntu:**

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

**Windows 11 PowerShell:**

```powershell
New-Item -ItemType Directory -Force -Path build
cd build

# Configure (specify GPU architecture if needed)
cmake .. `
    -DCMAKE_BUILD_TYPE=Release `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_CUDA_ARCHITECTURES=75  # 75=T4, 70=V100, 80=A100, 86=RTX 3060

# Build (multi-threaded)
cmake --build . --config Release --parallel

# Verify
.\bin\Release\check_gpu.exe
```

**Expected output:**

```
✓ CUDA available
✓ Device 0: Tesla T4 (or your GPU model)
✓ Compute Capability: 7.5
✓ Total Memory: 15 GB
```

**⚠️ If check_gpu fails (Intel Core i5 without NVIDIA GPU):**

```
✗ No CUDA-capable device detected
→ Use Google Colab for GPU phases (see section below)
→ You can still compile and run Phase 1 (CPU baseline) locally
```

---

## Usage

### Training Commands

#### Phase 1: CPU Baseline (Optional)

**Linux/Ubuntu:**

```bash
cd build

# Train CPU version (2 epochs for testing)
./bin/train_cpu --epochs 2 --batch-size 32

# Expected: ~15-20 min/epoch
```

**Windows 11 PowerShell:**

```powershell
cd build

# Train CPU version (2 epochs for testing)
.\bin\Release\train_cpu.exe --epochs 2 --batch-size 32

# Expected: ~15-20 min/epoch
```

**✅ Intel Core i5 users can run this phase locally**

#### Phase 2-3: GPU Optimized Training

**Linux/Ubuntu:**

```bash
# Train GPU version (optimized)
./bin/train_gpu --epochs 20 --batch-size 64

# Expected: ~20-30 sec/epoch
# Output: models/saved_weights/gpu_encoder_weights.bin
```

**Windows 11 PowerShell (requires NVIDIA GPU):**

```powershell
# Train GPU version (optimized)
.\bin\Release\train_gpu.exe --epochs 20 --batch-size 64

# Expected: ~20-30 sec/epoch
# Output: models\saved_weights\gpu_encoder_weights.bin
```

**⚠️ Intel Core i5 / No NVIDIA GPU? Use Google Colab:**

```powershell
# See "Google Colab Setup" section below
# This phase REQUIRES GPU - cannot run on Intel integrated graphics
```

**Training Options:**

- `--epochs N`: Number of training epochs (default: 20)
- `--batch-size N`: Batch size (default: 64 for GPU, 32 for CPU)
- `--learning-rate F`: Learning rate (default: 0.001)
- `--save-interval N`: Save weights every N epochs
- `--profile`: Enable profiling output

### Feature Extraction

**Linux/Ubuntu:**

```bash
# Extract 8,192-dim features for all images
./bin/extract_features

# Output:
#   data/train_features.txt (50,000 samples)
#   data/test_features.txt (10,000 samples)

# Expected: ~15-20 seconds
```

**Windows 11 PowerShell (requires NVIDIA GPU):**

```powershell
# Extract 8,192-dim features for all images
.\bin\Release\extract_features.exe

# Output:
#   data\train_features.txt (50,000 samples)
#   data\test_features.txt (10,000 samples)

# Expected: ~15-20 seconds
```

**⚠️ Intel Core i5 / No GPU:**

```powershell
# Run this in Google Colab (see Colab setup section)
# Features will be saved and can be downloaded to your local machine
```

### SVM Training and Evaluation

**Linux/Ubuntu:**

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

**Windows 11 PowerShell:**

```powershell
# Train SVM on extracted features
powershell -ExecutionPolicy Bypass -File scripts\train_svm.ps1
# Output: models\svm_model.bin
# Expected: 2-5 minutes

# Predict on test set
powershell -ExecutionPolicy Bypass -File scripts\predict_svm.ps1
# Output: results\test_predictions.txt
# Shows: Accuracy = 62.34% (6234/10000)

# Generate detailed evaluation
python scripts\evaluate_results.py
# Output:
#   results\confusion_matrix.png
#   results\evaluation_summary.txt
#   results\tsne_features.png (if enabled)
```

**✅ Intel Core i5 users:** SVM training can run on CPU - extract features from Colab first

### Benchmarking

**Linux/Ubuntu:**

```bash
# Run performance benchmark
bash scripts/benchmark_all.sh

# Compare all versions
./bin/benchmark_versions

# Output: Performance comparison table
```

**Windows 11 PowerShell:**

```powershell
# Run performance benchmark
powershell -ExecutionPolicy Bypass -File scripts\benchmark_all.ps1

# Compare all versions
.\bin\Release\benchmark_versions.exe

# Output: Performance comparison table
```

---

## Google Colab Setup

### 🎯 Recommended for Intel Core i5 / Systems without NVIDIA GPU

**Why use Google Colab:**

- ✅ Free access to NVIDIA T4 GPU (15GB VRAM)
- ✅ No local CUDA installation required
- ✅ Perfect for Intel Core i5 systems without discrete GPU
- ✅ Pre-configured environment
- ⚠️ Session limit: 12 hours (free tier), save checkpoints to Google Drive

### One-Click Setup

**Step 1: Open Google Colab**

1. Visit: https://colab.research.google.com/
2. Click **File → New notebook**
3. **Important:** Go to **Runtime → Change runtime type → GPU** (select T4, V100, or A100)

**Cell 1: Setup Environment**

```python
# ============================================
# IMPORTANT: Enable GPU runtime first!
# Runtime → Change runtime type → GPU
# ============================================

# Verify GPU is available
!nvidia-smi
# Expected output: Tesla T4 (or V100/A100)

# Clone repository
!git clone https://github.com/YOUR_USERNAME/PP-Final_Project.git
%cd PP-Final_Project

# Install dependencies
!apt-get update
!apt-get install -y cmake build-essential libomp-dev

# Download CIFAR-10 dataset (~160MB)
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

# Configure for Colab GPU (T4 = Compute Capability 7.5)
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

**Cell 5: Save to Google Drive (Important for Intel Core i5 users)**

```python
from google.colab import drive
drive.mount('/content/drive')

# Save results to your Google Drive
!mkdir -p /content/drive/MyDrive/CIFAR10_Results
!cp -r ../results/* /content/drive/MyDrive/CIFAR10_Results/
!cp -r ../models/saved_weights/* /content/drive/MyDrive/CIFAR10_Results/
!cp -r ../data/train_features.txt /content/drive/MyDrive/CIFAR10_Results/
!cp -r ../data/test_features.txt /content/drive/MyDrive/CIFAR10_Results/

print("✅ Results saved to Google Drive!")
print("📂 Location: MyDrive/CIFAR10_Results/")
print("\nYou can now:")
print("1. Download features to your local Windows 11 machine")
print("2. Run SVM training locally (CPU-only, works on Intel Core i5)")
print("3. View results and confusion matrix")
```

**Cell 6: Download Results to Local Machine (Alternative)**

```python
# Or download as ZIP file directly
!zip -r results.zip ../results ../models/saved_weights ../data/*_features.txt
from google.colab import files
files.download('results.zip')

print("✅ Download started!")
print("Extract on Windows 11 and run SVM training locally")
```

### Colab-Specific Notes

**Runtime Limits:**

- **Free tier:** 12 hours max session, may disconnect after inactivity
- **Pro tier:** 24 hours, better GPU availability, faster GPUs
- **Tip:** Save to Google Drive frequently to avoid data loss

**Memory Management:**

- **T4 GPU:** 15GB VRAM (sufficient for batch_size=64)
- **If "CUDA out of memory" error:** Reduce batch size to 32
  ```python
  # In Cell 3, modify:
  !./bin/train_gpu --epochs 20 --batch-size 32  # Reduced from 64
  ```

**Persistence (Critical for Long Sessions):**

```python
# Add this cell to save checkpoints every 5 epochs
# Insert before training cell

# Modify training command to save checkpoints
!./bin/train_gpu --epochs 20 --batch-size 64 --save-interval 5

# Auto-save to Google Drive every 5 epochs
# (implement in train_gpu.cu with --checkpoint flag)
```

**Common Issues & Solutions:**

| Issue                      | Solution                                                                                |
| -------------------------- | --------------------------------------------------------------------------------------- |
| **"CUDA out of memory"**   | Reduce `--batch-size` to 32 or 16                                                       |
| **"Session disconnected"** | Enable background execution: Runtime → Change runtime type → Enable "Run in background" |
| **"No GPU available"**     | Wait 10-15 minutes, or switch to different GPU type, or upgrade to Colab Pro            |
| **"Session timeout"**      | Save to Google Drive frequently, reconnect and resume from checkpoint                   |
| **Build errors**           | Ensure GPU runtime is enabled, run `!nvcc --version` to verify CUDA                     |

### 💡 Workflow for Intel Core i5 Users

**Step-by-Step Guide:**

1. **Phase 1 (CPU Baseline):** Run locally on Windows 11

   ```powershell
   # On your local machine
   .\bin\Release\train_cpu.exe --epochs 2
   ```

2. **Phase 2-4 (GPU Required):** Run on Google Colab

   - Train autoencoder on Colab (Cell 3)
   - Extract features on Colab (Cell 3)
   - Save features to Google Drive (Cell 5)

3. **SVM Training:** Download features and run locally

   ```powershell
   # On your local Windows 11 machine
   # Download train_features.txt and test_features.txt from Google Drive
   # Place in: data\ folder

   # Train SVM (CPU-only, works on Intel Core i5)
   powershell -ExecutionPolicy Bypass -File scripts\train_svm.ps1
   powershell -ExecutionPolicy Bypass -File scripts\predict_svm.ps1
   python scripts\evaluate_results.py
   ```

4. **Results:** View locally on Windows 11
   - Confusion matrix: `results\confusion_matrix.png`
   - Accuracy report: `results\evaluation_summary.txt`

---

## Project Structure

```
PP-Final_Project/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── docs/
│   ├── PROJECT_PLAN.md         # Comprehensive project timeline
│   ├── PHASE_1_GUIDE.md        # CPU implementation guide
│   ├── PHASE_2_GUIDE.md        # GPU porting guide
│   ├── PHASE_3_GUIDE.md        # GPU optimization guide
│   ├── PHASE_4_GUIDE.md        # SVM integration guide
│   ├── TESTING_DELIVERABLES.md # Testing and submission guide
│   └── CSC14120_2025_Final_Project.md  # Official requirements
├── include/
│   ├── config.h                # Global configuration
│   ├── data/
│   │   ├── cifar10_dataset.h   # CIFAR-10 data loader
│   │   ├── data_types.h        # Tensor definitions
│   │   └── data_utils.h        # Preprocessing utilities
│   ├── layers/                 # Neural network layers (CPU)
│   │   ├── conv2d_cpu.h
│   │   ├── relu_cpu.h
│   │   ├── maxpool_cpu.h
│   │   └── upsample_cpu.h
│   ├── cuda/                   # GPU implementations
│   │   ├── gpu_tensor.cuh
│   │   └── kernels/
│   │       ├── conv2d_kernel.cuh
│   │       ├── relu_kernel.cuh
│   │       ├── maxpool_kernel.cuh
│   │       └── upsample_kernel.cuh
│   ├── models/
│   │   ├── autoencoder_cpu.h
│   │   ├── autoencoder_gpu.cuh
│   │   └── feature_extractor.cuh
│   └── utils/
│       ├── logger.h
│       └── memory_pool.h
├── src/
│   ├── data/
│   │   ├── cifar10_dataset.cpp
│   │   └── data_utils.cpp
│   ├── layers/                 # CPU layer implementations
│   ├── cuda/                   # GPU kernel implementations
│   ├── models/                 # Autoencoder implementations
│   ├── utils/
│   ├── main_train.cpp          # CPU training entry point
│   ├── train_cpu.cpp
│   ├── train_gpu.cu
│   └── extract_features.cu
├── external/
│   └── libsvm/                 # LIBSVM library (downloaded)
├── data/
│   └── cifar-10-batches-bin/   # CIFAR-10 dataset (downloaded)
├── models/
│   └── saved_weights/          # Trained model weights
├── results/                    # Evaluation results and figures
├── scripts/
│   ├── download_cifar10.sh
│   ├── train_svm.sh
│   ├── predict_svm.sh
│   ├── evaluate_results.py
│   └── benchmark_all.sh
└── tests/
    ├── unit/                   # Unit tests
    ├── integration/            # Integration tests
    └── performance/            # Performance benchmarks
```

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
- **[PHASE_1_GUIDE.md](docs/PHASE_1_GUIDE.md):** CPU baseline implementation
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
