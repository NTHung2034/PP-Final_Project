# CSC14120 – PARALLEL PROGRAMMING — FINAL PROJECT

**Vietnam National University, Ho Chi Minh City**  
**University of Science**  
**Faculty of Information Technology**

---

## Table of Contents

1. [Introduction](#introduction)  
   1.1 [Problem Statement](#problem-statement)  
   1.2 [CIFAR-10 Dataset](#cifar-10-dataset)  
   1.3 [Expected Performance Targets](#expected-performance-targets)
2. [Background Knowledge](#background-knowledge)  
   2.1 [Autoencoders for Unsupervised Feature Learning](#autoencoders)  
   2.2 [Support Vector Machine (SVM) for Classification](#svm)
3. [Project Pipeline](#project-pipeline)
4. [Network Architecture](#network-architecture)  
   4.1 [Architecture Overview](#architecture-overview)  
   4.2 [Detailed Architecture Specification](#detailed-architecture-specification)
5. [Implementation Guideline](#implementation-guideline)
   - Phase 1: CPU Baseline & Data Pipeline
   - Phase 2: Naive GPU Implementation
   - Phase 3: Advanced Optimization (May have several versions here)
   - Phase 4: SVM Integration and analysed result
6. [Project Report](#project-report)
7. [Project Deliverable](#project-deliverable)

---

## 1. Introduction

### 1.1 Problem Statement

Feature engineering is a fundamental challenge in machine learning: how do we automatically discover good representations of data that capture its underlying structure? In this project, you will implement an Autoencoder-based unsupervised feature learning system for image classification on the CIFAR-10 dataset.

Traditional supervised learning trains a model end-to-end using labelled examples. However, autoencoders take a different approach: they learn meaningful representations by attempting to reconstruct the input data itself, without requiring any labels during the feature learning phase. This unsupervised pre-training can discover features that are often more robust and generalizable than those learned through direct supervision alone.

Your task is to build and optimize a complete two-stage pipeline:

**Stage 1 — Unsupervised Feature Learning:**

- Train a convolutional autoencoder to reconstruct CIFAR-10 images.
- The autoencoder learns to encode `32×32×3` images into an 8,192‑dimensional feature representation that captures meaningful visual patterns.
- No labels are used during this training phase.
- The network learns features that capture important visual patterns (edges, textures, shapes).

**Stage 2 — Supervised Classification:**

- Extract features from the trained encoder for all images.
- Train an SVM classifier on these learned features with class labels.
- Evaluate classification performance on the test set.

The primary focus is on implementing and optimizing the autoencoder training and inference in CUDA, while using existing libraries (e.g., LIBSVM) for the SVM classifier. Through systematic optimization, you will accelerate the autoencoder from taking hours on CPU to completing in seconds on GPU.

### 1.2 CIFAR-10 Dataset

CIFAR-10 is one of the most widely used benchmark datasets in computer vision and machine learning research. It provides a challenging yet computationally manageable image classification task.

**Dataset Specifications:**

- Image size: `32×32` pixels (RGB)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Training set: 50,000 images (5,000 per class)
- Test set: 10,000 images (1,000 per class)
- Total images: 60,000
- Format: Binary files with `uint8` pixel values

**Figure 1** — Examples from CIFAR-10 Dataset _(refer to original dataset for images)._

**Dataset Organization for This Project:**

- Autoencoder training: All 50,000 training images (labels ignored)
- SVM training: Same 50,000 images with labels (using extracted features)
- Evaluation: 10,000 test images with labels

**Dataset Link:** https://www.cs.toronto.edu/~kriz/cifar.html

### 1.3 Expected Performance Targets

| Metric                                   | Target       |
| ---------------------------------------- | ------------ |
| Autoencoder training time                | < 10 minutes |
| Feature extraction time (all 60K images) | < 20 seconds |
| Test classification accuracy             | 60–65%       |
| GPU speedup over CPU                     | > 20×        |

---

## 2. Background knowledge

### 2.1 Autoencoders for Unsupervised Feature Learning

**What is an Autoencoder?**  
An autoencoder is a neural network trained to reconstruct its input, forcing the network to learn a compressed, meaningful representation in the process. It consists of two parts:

- **Encoder:** Compresses the input into a low-dimensional latent representation (feature vector).
- **Decoder:** Reconstructs the original input from the latent representation.

**Figure 2** — The architecture of a standard autoencoder neural network.

**Training Objective:**

```
Loss = MSE(Input, Reconstructed_Output)
     = (1/N) * Σ(x - decoder(encoder(x)))^2
```

The network learns to minimize reconstruction error, forcing the encoder to capture essential information about the input.

**Key Concepts:**

1. **Bottleneck Layer (Latent Space):**

   - The smallest layer in the middle.
   - Forces compression and feature extraction.
   - In this project: `(6, 6, 128) = 4,608` dimensions _(note: project architecture uses (8,8,128)=8192 later — see Network Architecture section)_.

2. **Symmetric Architecture:**

   - Decoder mirrors the encoder.
   - Up sampling operations reverse the down sampling.
   - Helps in reconstruction quality.

3. **Feature Extraction:**
   - After training, discard the decoder.
   - Use only the encoder to extract features.
   - These features feed into the SVM classifier.

**Recommended Reading Materials:**

- Hinton, G. E., & Salakhutdinov, R. R. (2006). _Reducing the Dimensionality of Data with Neural Networks._ Science.  
  Link: https://www.cs.toronto.edu/~hinton/science.pdf
- Goodfellow, I., Bengio, Y., & Courville, A. _Deep Learning_ (2016), Chapter 14: Autoencoders.  
  Link: https://www.deeplearningbook.org/contents/autoencoders.html
- Video guide: Autoencoders and Representation Learning — https://www.youtube.com/watch?v=R3DNKE3zKFk
- Reference source code examples:
  - https://github.com/turkdogan/autoencoder
  - https://github.com/tbennun/cudnn-training

Additional CNN references and guides are also suggested in the PDF (Stanford cheat-sheets, Dive into Deep Learning, etc.).

### 2.2 Support Vector Machine (SVM) for Classification

**Your Role with SVM:** You do **not** need to implement SVM from scratch. Use existing optimized libraries.

**Recommended SVM Libraries:**

1. **LIBSVM (Primary Recommendation)**

   - Popular C/C++ SVM library, supports multi-class classification and RBF kernel.
   - Link: https://www.csie.ntu.edu.tw/~cjlin/libsvm/ and https://github.com/cjlin1/libsvm

2. **LIBLINEAR (Alternative for Linear SVM)**

   - Faster for linear kernels; useful for experiments.
   - Link: https://www.csie.ntu.edu.tw/~cjlin/liblinear/

3. **ThunderSVM (GPU-Accelerated)**
   - GPU implementation of SVM — optional for extra credit.
   - Link: https://github.com/Xtra-Computing/thundersvm

---

## 3. Project Pipeline

The project follows a clear 5-step pipeline:

**Step 1: Load CIFAR-10 Images**

- 50,000 training images (for autoencoder)
- 10,000 test images
- Preprocess: normalize to `[0,1]`

**Step 2: Train Autoencoder (Your CUDA Code)**

- Use all 50k images (unsupervised training; ignore labels)
- CNN-based autoencoder architecture (see Section 4)
- Train to minimize reconstruction loss
- Save encoder weights

**Step 3: Extract Features (Your CUDA Code)**

- Load trained encoder weights
- Run encoder forward pass (no decoder)
- `train_features`: `(50000, 8192)`
- `test_features`: `(10000, 8192)`

**Step 4: Train SVM (Library)**

- Input: `train_features + labels`
- Kernel: RBF (Radial Basis Function)
- Hyperparameters: `C=10, gamma=auto`
- Output: trained SVM model

**Step 5: Evaluate**

- Predict on `test_features` using SVM
- Calculate accuracy, confusion matrix
- Expected accuracy: 60–65%

**Key Points:**

- Focus is on Steps 2 and 3 (autoencoder implementation in CUDA).
- Use libraries for Step 4 (SVM training).
- Evaluate in Step 5 to measure success.

---

## 4. Network Architecture

### 4.1 Architecture Overview

```
INPUT: (32, 32, 3) ➔ ENCODER (compress) ➔ LATENT: (8, 8, 128) = 8,192 features ➔ DECODER (reconstruct) ➔ OUTPUT: (32, 32, 3)
```

### 4.2 Detailed Architecture Specification

**ENCODER (Down sampling Path)**

- **INPUT:** `(32, 32, 3)`
- `Conv2D(256 filters, 3×3 kernel, padding=1, stride=1) + ReLU` → `(32, 32, 256)`
- `MaxPool2D(2×2, stride=2)` → `(16, 16, 256)`
- `Conv2D(128 filters, 3×3 kernel, padding=1, stride=1) + ReLU` → `(16, 16, 128)`
- `MaxPool2D(2×2, stride=2)` → `(8, 8, 128)`
- **LATENT REPRESENTATION:** `(8, 8, 128) = 8192 dimensions`

**DECODER (Upsampling Path — Mirror of Encoder)**

- **LATENT:** `(8, 8, 128)`
- `Conv2D(128 filters, 3×3 kernel, padding=1, stride=1) + ReLU` → `(8, 8, 128)`
- `UpSample2D(2×2)` [Nearest neighbor or bilinear] → `(16, 16, 128)`
- `Conv2D(256 filters, 3×3 kernel, padding=1, stride=1) + ReLU` → `(16, 16, 256)`
- `UpSample2D(2×2)` [Nearest neighbor or bilinear] → `(32, 32, 256)`
- `Conv2D(3 filters, 3×3 kernel, padding=1, stride=1)` [No activation] → `(32, 32, 3)`
- **OUTPUT:** `(32, 32, 3)`

**Keras reference setup:**

```python
input_size = (32, 32, 3)

input_image = Input(shape=input_size)

# Encoder
x = Conv2D(256, (3, 3), activation='relu', padding='same')(input_image)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoded_layer')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(3, (3, 3), padding='same')(x)
```

**Parameter summary (as given):**

```
Total params: 751,875
Trainable params: 751,875
Non-trainable params: 0
```

Layer list includes `conv2d_1` through `conv2d_5` and pooling/upsampling layers — see original for details.

---

## 5. Implementation Guideline

You will implement this project in at least 4 progressive phases, focusing on CUDA optimization for the autoencoder training and inference.

### Phase 1: CPU Baseline & Data Pipeline

**Objective:** Set up project infrastructure and create a working CPU baseline.

**What to Implement:**

#### 1.1 Data Loading and Preprocessing

- Create a `CIFAR10 Dataset` class to handle data loading.
- Read CIFAR-10 binary files (5 training batches + 1 test batch).
- Parse the binary format: `1 byte label + 3,072 bytes image per record`.
- Convert `uint8` pixel values `[0, 255]` to `float [0, 1]`.
- Implement batch generation for training.
- Add data shuffling capability.
- Organize train images (50,000), test images (10,000), and their labels in memory.

#### 1.2 CPU Neural Network Layers

Implement CPU versions of all necessary operations:

- **Convolution (Conv2D):** Apply `3×3` kernels with padding and stride.
- **ReLU Activation:** Element-wise `max(0, x)`.
- **Max Pooling:** `2×2` pooling to downsample by half.
- **Upsampling:** Nearest neighbor interpolation to double spatial dimensions.
- **MSE Loss:** Mean squared error between output and target for reconstruction.

#### 1.3 Autoencoder

- Create an `Autoencoder` class to encapsulate the network.
- Allocate memory for weights and biases (5 conv layers total).
- Implement weight initialization.
- Allocate memory for intermediate activations between layers.
- Implement forward pass (encoder → decoder).
- Implement backward pass (gradients).
- Implement feature extraction: run encoder and return latent representation.
- Add weight saving/loading functionality.

#### 1.4 Training

- Hyperparameters: `batch_size = 32`, `epochs = 20`, `learning_rate = 0.001`.
- Loop over epochs and batches: forward → loss → backward → update weights.
- Track and display training loss; measure time per epoch; save weights.

**Deliverables (Phase 1):**

- Working data loading & preprocessing pipeline.
- CPU implementation of layers.
- Baseline performance measurements.

---

### Phase 2: Naive GPU Implementation

**Objective:** Port operations to GPU with basic parallelization.

**What to Implement:**

#### 2.1 GPU Memory Management

- Create `GPUAutoencoder` class for GPU operations.
- Allocate device memory for weights, activation buffers, gradients.
- Implement host↔device weight copies.
- Proper memory cleanup (`cudaFree`).

#### 2.2 Naive GPU Kernels

Design basic GPU kernels where each thread handles simple work:

- **Convolution Kernel:**

  - Each thread computes one output pixel.
  - Thread loops over kernel and input channels.
  - Uses global memory for reads/writes.
  - Handles padding boundaries.

- **ReLU Kernel:**

  - Each thread processes one element: `x = max(0, x)`.

- **MaxPooling Kernel:**

  - Each thread computes one output element and finds maximum in `2×2` window.

- **Upsampling Kernel:**

  - Each thread computes one output pixel and maps coordinates back to input (nearest neighbour).

- **MSE Loss Kernel:**
  - Parallel reduction to sum squared differences.
  - Use shared memory for partial sums and `atomicAdd` for final accumulation.

#### 2.3 GPU Forward Pass

- Copy batch to device.
- Launch kernels sequentially for layers.
- Synchronize and copy output back to host.
- Compute loss.

#### 2.4 GPU Backward Pass

- Implement gradient kernels for conv, relu, maxpool, upsample.
- Backpropagate errors, compute weight gradients.
- Implement SGD update: `weight -= learning_rate * gradient`.

#### 2.5 GPU Training Loop

- Increase batch size for GPU (e.g., `64`).
- For each batch: copy to device → forward → loss → backward → update.
- Use GPU timer, display progress, save trained weights.

**Deliverables (Phase 2):**

- All layers ported to GPU and working training loop.
- Forward and backward passes implemented and correctness verified.

---

### Phase 3: Advanced Optimization (May have several versions here)

**Objective:** Optimize memory access patterns and apply advanced techniques for maximum performance.

**Optimization Ideas (suggested — not all required):**

**Category 1: Memory Optimization**

1. Shared Memory Tiling for Convolution.
2. Convert convolution to matrix multiplication (im2col + GEMM).
3. Memory coalescing optimization (reorder data/layout).
4. Use constant memory for small weights (biases/kernels).
5. Pinned (page-locked) memory (`cudaMallocHost`) for faster transfers.
6. Unified Memory for simplified management.
7. Memory pool / buffer reuse to avoid frequent `cudaMalloc/cudaFree`.

**Category 2: Kernel-Level Optimization** 8. Kernel Fusion (Conv + ReLU + Bias). 9. Block-Level Fusion (fuse encoder/decoder). 10. Loop unrolling in kernels. 11. Vectorized memory access (`float4` loads/stores). 12. Tune thread block dimensions for occupancy. 13. Mixed precision training (`FP16/FP32`).

**Category 3: Parallelism & Concurrency** 14. Gradient checkpointing to trade compute for memory. 15. Multi-stream pipeline to overlap H2D transfers and computation. 16. Batched operations to amortize kernel launch overhead.

---

### Phase 4: SVM Integration and analysed result

**Objective:** Complete end-to-end pipeline and evaluate performance.

- Extract features using trained encoder.
- Train SVM (LIBSVM or alternatives) on learned features.
- Evaluate classification performance and report results.

---

## 6. Project Report

**Report Format and Structure**

**Submission Format:** Jupyter Notebook (`.ipynb`) runnable on Google Colab.

### Section 1: Problem Description

Include:

- Problem statement and motivation for GPU acceleration.
- CIFAR-10 dataset overview, sample images, preprocessing steps.
- Autoencoder architecture (diagram, layer specs, latent rep).
- Project objectives (performance goals, learning objectives, success criteria).

### Section 2: Implementation Phases

For each phase, provide a structured description following this template:

**Phase 2.1: CPU Baseline Implementation**

- **Objectives:**
  - What you aimed to achieve in this phase
  - Why this phase is necessary
- **Implementation Details:**
  - **Data Pipeline:** How you loaded and pre-processed CIFAR-10 data
  - **Layer Implementations:** Brief description of each layer (Conv2D, ReLU, MaxPool, Upsample)
  - **Training Loop:** How you structured the training process
  - **Key Code Snippets:** Show 2-3 critical functions (e.g., convolution function signature and main loop structure)
- **Results:**
  - Training time per epoch and total training time
  - Final reconstruction loss
  - Sample reconstructed images (show original vs reconstructed)
  - Memory usage
- **Key Takeaways:**
  - What did you learn about the algorithm?
  - What insights guided your GPU implementation?

**Phase 2.2: GPU Basic Implementation**

- **Objectives:**
  - Port CPU code to GPU with basic parallelization
  - Verify correctness of GPU kernels
  - Establish baseline GPU performance
- **Implementation Details:**
  - **Parallelization Strategy:** How you mapped operations to GPU threads
  - **Kernel Designs:**
    - Convolution kernel: thread-to-output mapping
    - Pooling kernel: how threads handle 2×2 windows
    - Other kernels (ReLU, upsampling)
  - **Memory Management:** Device memory allocation strategy
  - **Key Code Snippets:** Show kernel signatures and launch configurations
- **Results:**
  - Training time per epoch and total training time
  - Speedup over CPU baseline (include table and chart)
  - GPU memory usage
  - Verification that outputs match CPU (show error metrics)
- **Profiling Analysis:**
  - Basic profiling results (time spent in each kernel type)
  - Memory bandwidth utilization (if measured)
  - Initial bottleneck identification
- **Key Takeaways:**
  - What was surprisingly fast or slow?
  - Where do you see optimization opportunities?

**Phase 2.3: GPU Optimized Implementation — Version 1**

- **Optimization Focus:** (e.g., Memory Optimization)
- **Objectives:**
  - What specific optimization(s) you targeted
  - Expected performance improvement
- **Implementation Details:**
  - **Optimization Technique(s) Applied:**
    - Detailed explanation of the optimization (e.g., shared memory tiling)
    - Why this optimization should help
    - Implementation approach
  - **Key Code Snippets:** Show the optimized kernel or key changes
- **Results:**
  - Training time comparison with previous version
  - Speedup over previous phase (incremental and cumulative)
  - Performance metrics (bandwidth utilization, occupancy)
  - Profiling comparison: before vs after
- **Analysis:**
  - Why did this optimization work (or not work as expected)?
  - What did profiling reveal?
  - What's the next bottleneck?
- **Key Takeaways:**
  - Lessons learned from this optimization
  - Applicability to other problems

**Phase 2.4: GPU Optimized Implementation — Version 2 (if applicable)**

- **Optimization Focus:** (e.g., Kernel Fusion and Advanced Techniques)
- Follow the same structure as Version 1:
  - Objectives
  - Implementation Details
  - Results
  - Analysis
  - Key Takeaways
- _Note: You may have multiple optimization versions. Create a separate subsection for each major optimization iteration._

**Phase 2.5: SVM Integration**

- **Objectives:**
  - Extract features using trained encoder
  - Train SVM classifier on learned features
  - Evaluate end-to-end classification performance
- **Implementation Details:**
  - **Feature Extraction:** How you extracted 8,192-dim features from encoder
  - **LIBSVM Integration:** How you interfaced with LIBSVM
  - **Hyperparameter Selection:** SVM parameters chosen (C, gamma, kernel)
  - **Key Code Snippets:** Feature extraction and SVM training code
- **Results:**
  - Feature extraction time (50K train + 10K test)
  - SVM training time
  - Classification accuracy on test set
  - Per-class accuracy breakdown (table)
  - Confusion matrix (visualization)
  - Comparison with baseline methods (if available)
- **Analysis:**
  - Which classes are easiest/hardest to classify?
  - What does the confusion matrix reveal?
  - How does accuracy compare to expectations?
- **Key Takeaways:**
  - Quality of learned features
  - Effectiveness of two-stage approach

### Section 3: Comprehensive Performance Analysis

Provide comparison across phases including a table with:

- Phase, Training Time, Speedup (vs CPU), Incremental Speedup, Memory Usage, Key Optimization.

Include visualizations: bar charts and line graphs showing training time and cumulative speedup.

Table format:

| Phase        | Training Time | Speedup (vs CPU) | Incremental Speedup | Memory Usage | Key Optimization        |
| ------------ | ------------- | ---------------- | ------------------- | ------------ | ----------------------- |
| CPU Baseline | 1800s         | 1.0×             | -                   | -            | -                       |
| GPU Basic    | 180s          | 10.0×            | 10.0×               | 2.1 GB       | Parallelization         |
| GPU Opt v1   | 45s           | 40.0×            | 4.0×                | 2.3 GB       | Shared memory           |
| GPU Opt v2   | 25s           | 72.0×            | 1.8×                | 2.5 GB       | Kernel Fusion + Streams |

### Section 4: Lessons Learned and Challenges Overcome

- Key technical insights (CUDA, deep learning, performance optimization).
- Major challenges and solutions (format: Challenge / Problem / Solution / Lesson).

### Section 5: Conclusion and Future Work

- Project summary, final metrics, achievements.
- Limitations and possible future improvements.

---

## 7. Project Deliverable

You must submit the following items:

1. **Team Plan and Work Distribution**

   - Document detailing each member's responsibilities, task breakdown, timeline, contribution percentages.

2. **Project Report**

   - Jupyter Notebook (`.ipynb`) as specified.
   - Must be executable on Google Colab.
   - Export to PDF for archival purposes.

3. **Source Code Package**

   - All source code files (`.cpp`, `.cu`, `.h`, etc.).
   - `README.md` with setup instructions (dependencies, CUDA version, libraries), compilation and execution commands, hardware requirements, expected outputs.
   - Trained model weights (if file size permits, otherwise provide links).

4. **Presentation Video (15–20 minutes)**

   - Presentation Video (15–20 minutes)
   - Upload to YouTube with Unlisted visibility.
   - Include the link in your notebook report and README file.
   - DO NOT upload video to Moodle (file size restrictions).
   - Video Content Should Cover:
   - Problem overview and approach (2–3 min)
   - Live demonstration of running code (5–7 min)
   - Performance results and analysis (5–7 min)
   - Key optimizations and lessons learned (3–5 min)
