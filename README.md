# CUDA Autoencoder for CIFAR-10 Feature Learning
- **Link youtube**: [https://youtu.be/a8Oy_k7BFmk](https://youtu.be/a8Oy_k7BFmk)
- **Link github**: [https://github.com/NTHung2034/PP-Final_Project](https://github.com/NTHung2034/PP-Final_Project)
- **SVM weight**: [Drive](https://drive.google.com/file/d/1TWnHNML5OJB_OUQVgGkouz4s0W8Mnyof/view?fbclid=IwY2xjawO8GMNleHRuA2FlbQIxMABicmlkETExMVFBekFzZlB5NWFVSGFTc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHrN_0gPbF_rB5xf8qLo0FvzPl-MAiLmCLZfigeIaF_eWACIQFGUOm1ynX5BU_aem_JQrEhVuOB0ZtqnnZdwkQ0Q)
---

## **Table of Contents**

- [**Quick Start with Google Colab**](#quick-start-with-google-colab)
- [**Overview**](#overview)
- [**Project Architecture**](#project-architecture)
- [**Requirements**](#requirements)
- [**Setup**](#setup)
- [**Compilation**](#compilation)
- [**Execution**](#execution)
- [**Expected Outputs**](#expected-outputs)
- [**Notes**](#notes)
- [**Project Structure**](#project-structure)

---

## **Quick Start with Google Colab**

### 1. Clone the Repository
```bash
!git clone https://github.com/NTHung2034/PP-Final_Project.git
%cd PP-Final_Project
```

### 2. Download CIFAR-10 Dataset (root directory)
```bash
!bash ./scripts/download_cifar10.sh
```

### 3. Build
```bash
!mkdir -p build
%cd build
!cmake .. -DENABLE_CUDA=ON

# For colab usage
!make
```

### 4. Run Correctness Test
```bash
!./test_comparison_cpu_gpu
```

### 5. Train
```bash
!./train_cpu
!./train_gpu_naive
!./train_gpu_opt_v1
!./train_gpu_opt_v2
```

---

## **Overview**

This project implements a convolutional autoencoder for CIFAR-10 and compares multiple implementations:
- CPU baseline (OpenMP-ready)
- GPU Naive CUDA implementation
- GPU Optimized v1 (shared memory + memory pool)
- GPU Optimized v2 (kernel fusion + CUDA streams + loop unrolling)

The learned latent features can be used for downstream classification with SVM (LIBSVM / cuML pipeline).

---

## **Project Architecture**

- **Autoencoder**: Conv2D → ReLU → MaxPool (Encoder) and Upsample → Conv2D (Decoder)
- **Pipeline**: Train autoencoder → extract latent features → train SVM classifier
- **Correctness check**: `test_comparison_cpu_gpu` compares decoded outputs across CPU/GPU variants with the same weights and input

---

## **Requirements**

### Hardware
- **CPU:** Multi-core x86_64 processor
- **GPU:** T4 - VRAM: 15GB

### Software
- **CMake:** 3.18 or higher                           ****IMPORTANT***
- **C++ Compiler:** GCC 9+ / Clang 10+ / MSVC 2019+   ****IMPORTANT***
- **CUDA Toolkit:** 11.0+ (recommended 11.8+) (optional, for GPU support)
- **OpenMP:** Required for CPU parallelization
- **LIBSVM:** For SVM classification (Phase 4)
- **cuDNN (optional):** Improves convolution performance when available

### Colab specific environment
- 2025.10link
- Ubuntu 22.04.4 LTS
- Python 3.12.12
- numpy 2.0.2
- PyTorch 2.8.0
- Jax 0.5.3
- TensorFlow 2.19.0 (not included in TPU runtimes)
- R version 4.5.1 (2025-06-13) -- "Great Square Root"
- julia version 1.11.5

---

## **Setup**

- Install CMake ≥ 3.18 and a C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+).
- Install CUDA Toolkit 11.0+; recommended 11.8+ with cuDNN for best performance.
- Verify `nvcc --version` and GPU driver compatibility.
- Add CUDA `bin` and compiler toolchain to your PATH.
- Download CIFAR-10 with `./scripts/download_cifar10.sh` from repo root.
- LIBSVM is vendored under `external/`; no extra install needed for default build.
- Optional (cuML SVM on Google Colab): `cuml`, `scikit-learn`, `matplotlib`, `seaborn`

---

## **Compilation**
> If you want to run in local. Create a clean build directory (recommended):
```bash
mkdir -p build
cd build
```

CPU-only build:
```bash
cmake .. -DENABLE_CUDA=OFF
cmake --build . --config Release
```

GPU build (Naive, Opt V1, Opt V2):
```bash
cmake .. -DENABLE_CUDA=ON
cmake --build . --config Release
```

### **Using ThunderSVM (Alternative GPU-accelerated SVM)**

If you want to use ThunderSVM instead of LIBSVM/cuML:

1. **Build ThunderSVM**:
   ```bash
   cd external/thundersvm
   mkdir build && cd build && cmake .. && make -j
   cd ../../
   ```

2. **Replace CMake configuration**:
   ```bash
   # Use ThunderSVM configuration
   cp CMakeThunder.txt CMakeLists.txt
   ```

3. **Build with ThunderSVM**:
   ```bash
   mkdir -p build
   cd build
   cmake .. -DENABLE_CUDA=ON
   make
   ```

4. **Run ThunderSVM executables**:
   ```bash
   # From build directory only after have naive weight
   ./thundersvm_naive      # GPU Naive + ThunderSVM
   ```

Notes:
- CUDA architectures are set in `CMakeLists.txt` (`89;75;70;61;50`); adjust if your GPU differs.
- CMake auto-creates `data/` and the `models/saved_weights*` directories.
- On Windows, use the same generator for configure and build (e.g., `-G "Visual Studio 17 2022"`).

---

## **Execution**

> From the `build` directory:

Correctness test (recommended before training):
```bash
./test_comparison_cpu_gpu          # CPU vs GPU Naive / Opt V1 / Opt V2 outputs
```

Training:
```bash
./train_cpu                        # CPU baseline
./train_gpu_naive                  # GPU Naive
./train_gpu_opt_v1                 # GPU Optimized v1
./train_gpu_opt_v2                 # GPU Optimized v2
```

Feature extraction + SVM:
```bash
./extract_cpu
./extract_naive
./extract_opt_v1
./extract_opt_v2
./train_svm_cpu
```

ThunderSVM training (if using CMakeThunder.txt):
```bash
# After training the autoencoder with train_gpu_naive
./thundersvm_naive      # Combines feature extraction + ThunderSVM classification
```

cuML SVM training (Google Colab):
> From `root` directory
```bash
# Run after `extract_*` generates svm_features/*.bin
python src/models_svm/cuml_svm_train.py naive
python src/models_svm/cuml_svm_train.py opt_v1
python src/models_svm/cuml_svm_train.py opt_v2
```

Data and outputs:
- CIFAR-10 binaries are read from `data/` (see `include/config.h`).
- Model checkpoints are written to `models/saved_weights*`.

---

## **Expected Outputs**

Run outputs (files may vary slightly by variant, but follow these patterns):

- **Correctness** (`test_comparison_cpu_gpu`): prints reconstruction error; **MSE should be very small** (typically \( \sim 10^{-6} \)–\(10^{-3}\)).
- **Training** (`train_*`): epoch loss logs + weights under `models/saved_weights*` (e.g. `models/saved_weights/cpu_final.bin`, GPU `models/saved_weights_gpu_*/training_summary.txt`).
- **Feature extraction** (`extract_*`):
  - CPU: `models/saved_weights/train_features_cpu.bin`, `models/saved_weights/test_features_cpu.bin`
  - GPU (for cuML): `models/saved_weights_gpu_*/svm_features/{train_features.bin,train_labels.bin,test_features.bin,test_labels.bin}`
- **SVM**:
  - CPU (LIBSVM): `models/saved_weights/test_predictions.txt`, `models/saved_weights/svm_results.txt`, `models/saved_weights/svm_model_cpu.bin`
  - GPU (for cuML): `models/saved_weights_gpu_*/svm_features/confusion_matrix.png`, `training_summary.txt`, `cuml_svm_model.pkl`

---

## **Notes**

- If you use PowerShell, executables may be invoked as `.\train_cpu`, `.\test_comparison_cpu_gpu`, ...
- CUDA architectures are set in `CMakeLists.txt` (`89;75;70;61;50`). Adjust if your GPU is not covered.
- **GPU memory**: ≥ 6GB VRAM minimum; 8–12GB recommended for batch size 64.
- **Recommended GPUs**: RTX 3060+ or NVIDIA T4+ (Compute Capability ≥ 6.0).

---
## **Project Structure**

```
PP-Final_Project/
│
├── CMakeLists.txt
├── CMakeThunder.txt                # Alternative CMake config (ThunderSVM)
├── README.md
├── docs/                           # Final project PDF
├── Nvisight/                       # Nsight profiling outputs
├── include/
│   ├── config.h
│   ├── data/                      # CIFAR-10 loader + tensor types
│   ├── layers/                    # CPU layers
│   ├── layers_gpu_naive/          # GPU Naive layers
│   ├── layers_gpu_opt_v1/         # GPU Optimized v1 layers
│   ├── layers_gpu_opt_v2/         # GPU Optimized v2 layers
│   ├── models/                    # Autoencoder headers (CPU/GPU)
│   └── utils/                     # Logging / memory helpers
├── models/
│   ├── saved_weights/             # CPU outputs (weights/features/SVM results)
│   ├── saved_weights_gpu_naive/   # GPU naive checkpoints/results
│   ├── saved_weights_gpu_opt_v1/  # GPU opt v1 checkpoints/results
│   └── saved_weights_gpu_opt_v2/  # GPU opt v2 checkpoints/results
├── notebooks/
│   ├── main_report.ipynb          # Main report notebook
│   ├── images/                    # Figures used in report
│   └── demo/                      # Demo notebooks
│       ├── CPU_demo.ipynb         # CPU pipeline
│       ├── Execute_naive.ipynb    # GPU naive pipeline
│       ├── Execute_v1.ipynb       # GPU optimized v1 pipeline
│       ├── Execute_v2.ipynb       # GPU optimized v2 pipeline
│       └── thunder.ipynb          # ThunderSVM workflow
├── src/
│   ├── train_cpu.cpp
│   ├── train_gpu_naive.cu
│   ├── train_gpu_opt_v1.cu
│   ├── train_gpu_opt_v2.cu
│   ├── data/
│   ├── layers/
│   ├── layers_gpu_naive/
│   ├── layers_gpu_opt_v1/
│   ├── layers_gpu_opt_v2/
│   ├── models/
│   └── models_svm/                # SVM pipeline + feature extraction
│       ├── extract_cpu.cpp
│       ├── extract_naive.cu
│       ├── extract_opt_v1.cu
│       ├── extract_opt_v2.cu
│       ├── train_svm_cpu.cpp
│       └── cuml_svm_train.py      # cuML SVM (Colab)
├── tests/
│   ├── comparison_cpu_gpu.cu      # `test_comparison_cpu_gpu`
│   └── ...                        # Other tests
├── external/                      # SVM backends + dependencies
│   ├── libsvm/
│   └── thundersvm/
├── scripts/                       # Dataset download script
├── data/                          # CIFAR-10 binaries (generated by download script)
└── build/                         # Generated build outputs

```
