# CIFAR-10 Autoencoder Project Plan

**Project Duration:** November 28, 2025 - December 10, 2025 (13 days)  
**Team:** [Your Team Name]  
**Objective:** Implement and optimize a CUDA-accelerated autoencoder for CIFAR-10 feature learning

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Timeline and Milestones](#timeline-and-milestones)
3. [Phase Breakdown](#phase-breakdown)
4. [Resource Requirements](#resource-requirements)
5. [Risk Management](#risk-management)
6. [Success Criteria](#success-criteria)

---

## 1. Project Overview

### 1.1 Project Goals

- **Primary Goal:** Build a two-stage pipeline combining unsupervised feature learning (autoencoder) with supervised classification (SVM)
- **Performance Target:** Achieve 60-65% classification accuracy on CIFAR-10 test set
- **Optimization Goal:** Achieve >20× GPU speedup over CPU baseline
- **Learning Objective:** Master CUDA optimization techniques through progressive implementation

### 1.2 Architecture Summary

```
Input: (32×32×3) RGB Images
   ↓
[ENCODER]
   Conv2D(256, 3×3) + ReLU → (32×32×256)
   MaxPool(2×2) → (16×16×256)
   Conv2D(128, 3×3) + ReLU → (16×16×128)
   MaxPool(2×2) → (8×8×128)
   ↓
Latent: (8×8×128) = 8,192 features
   ↓
[DECODER]
   Conv2D(128, 3×3) + ReLU → (8×8×128)
   UpSample(2×2) → (16×16×128)
   Conv2D(256, 3×3) + ReLU → (16×16×256)
   UpSample(2×2) → (32×32×256)
   Conv2D(3, 3×3) → (32×32×3)
   ↓
Output: Reconstructed (32×32×3) RGB Images
   ↓
[SVM CLASSIFIER]
   Input: 8,192-dim features
   Output: 10 class predictions
```

### 1.3 Technology Stack

- **Language:** C++17, CUDA C
- **Build System:** CMake 3.18+
- **CUDA:** 11.0+ (cuDNN optional)
- **SVM Library:** LIBSVM
- **Development:** Local GPU workstation + Google Colab
- **Version Control:** Git

---

## 2. Timeline and Milestones

### 2.1 Project Schedule

| Week       | Dates          | Phase             | Key Deliverables                                       |
| ---------- | -------------- | ----------------- | ------------------------------------------------------ |
| **Week 1** | Nov 28 - Dec 1 | Phase 1 + 2 Start | CPU baseline complete, GPU kernels started             |
| **Week 2** | Dec 2 - Dec 7  | Phase 2 + 3       | GPU implementation complete, optimizations in progress |
| **Week 3** | Dec 8 - Dec 10 | Phase 4 + Report  | SVM integration, full report, presentation video       |

### 2.2 Detailed Milestone Schedule

#### **Day 1-2: November 28-29** (Setup + Phase 1 Start)

**Focus:** Environment setup and data pipeline

- [ ] **Milestone 1.1:** Development environment ready
  - CMake configured, CUDA detected
  - Google Colab notebook created with GPU runtime
  - CIFAR-10 dataset downloaded (60,000 images)
- [ ] **Milestone 1.2:** Data loading working
  - Binary file parsing complete
  - Normalization verified ([0,1] range)
  - Batch generation tested (32 images/batch)

**Verification Test:**

```bash
./bin/test_data_loading
# Expected: "Successfully loaded 50,000 training images"
# Expected: "Sample pixel values in range [0.0, 1.0]"
```

#### **Day 3-4: November 30 - December 1** (Phase 1 Complete)

**Focus:** CPU neural network implementation

- [ ] **Milestone 1.3:** CPU layers implemented
  - Conv2D with 3×3 kernels
  - ReLU activation
  - MaxPooling 2×2
  - Upsampling (nearest neighbor)
  - MSE loss function
- [ ] **Milestone 1.4:** CPU training working
  - Forward pass: input → encoder → decoder → output
  - Backward pass: gradients computed
  - Weight updates via SGD
  - Loss decreasing over epochs

**Verification Test:**

```bash
./bin/train_cpu --epochs 2 --batch-size 32
# Expected: Loss decreases from ~0.15 to ~0.08 in 2 epochs
# Expected: Training time ~15-20 min/epoch
```

**Deliverable:** Phase 1 complete, baseline metrics recorded

---

#### **Day 5-6: December 2-3** (Phase 2 Start)

**Focus:** Basic GPU porting

- [ ] **Milestone 2.1:** GPU memory management
  - Device memory allocated for weights, activations
  - Host↔Device transfer working
  - Memory cleanup (no leaks)
- [ ] **Milestone 2.2:** Naive GPU kernels
  - Convolution kernel (1 thread = 1 output pixel)
  - ReLU kernel (element-wise)
  - MaxPooling kernel
  - Upsampling kernel
  - MSE reduction kernel

**Verification Test:**

```bash
./bin/test_gpu_kernels
# Expected: All kernel outputs match CPU within 1e-5 tolerance
# Expected: No CUDA errors
```

#### **Day 7: December 4** (Phase 2 Continue)

**Focus:** GPU training loop

- [ ] **Milestone 2.3:** GPU forward/backward pass
  - All layers chained on GPU
  - Gradients computed correctly
  - Weight updates on device
- [ ] **Milestone 2.4:** GPU training complete
  - Full training run successful
  - Speedup measured vs CPU
  - Profiling data collected

**Verification Test:**

```bash
./bin/train_gpu --epochs 2 --batch-size 64
# Expected: Loss curve matches CPU baseline
# Expected: 5-10× speedup vs CPU
# Expected: Training time ~2-3 min/epoch
```

**Deliverable:** Phase 2 complete, initial GPU speedup achieved

---

#### **Day 8-9: December 5-6** (Phase 3: Optimization v1)

**Focus:** Shared memory optimization for convolution

- [ ] **Milestone 3.1:** Shared memory tiling
  - Tile-based convolution kernel
  - Reduced global memory accesses
  - Profiling shows bandwidth improvement

**Verification Test:**

```bash
./bin/train_gpu_opt_v1 --profile
# Expected: 2-4× speedup vs naive GPU
# Expected: Memory bandwidth utilization >70%
# Expected: Training time ~45-60 sec/epoch
```

**Deliverable:** Optimization v1 complete, 20-40× cumulative speedup

#### **Day 9-10: December 6-7** (Phase 3: Optimization v2)

**Focus:** Kernel fusion

- [ ] **Milestone 3.2:** Fused kernels
  - Conv+ReLU fused
  - Conv+ReLU+Bias fused
  - Reduced kernel launch overhead

**Verification Test:**

```bash
./bin/train_gpu_opt_v2 --profile
# Expected: 1.5-2× speedup vs v1
# Expected: Fewer kernel launches in profiler
# Expected: Training time ~25-35 sec/epoch
```

**Deliverable:** Optimization v2 complete, >50× cumulative speedup

#### **Day 10-11: December 7-8** (Phase 3: Optimization v3 - Optional)

**Focus:** Multi-stream pipelining

- [ ] **Milestone 3.3:** Concurrent execution (Optional)
  - Multiple CUDA streams
  - H2D transfer overlapped with compute
  - D2H transfer overlapped with next batch H2D

**Verification Test:**

```bash
./bin/train_gpu_opt_v3 --profile
# Expected: 1.3-1.8× speedup vs v2 (if successful)
# Expected: Stream overlap visible in nvprof timeline
```

**Deliverable:** All Phase 3 optimizations complete, >70× target speedup

---

#### **Day 11: December 8** (Phase 4: SVM)

**Focus:** Feature extraction and SVM training

- [ ] **Milestone 4.1:** Feature extraction
  - Load trained encoder weights
  - Extract 8,192-dim features for all 60K images
  - Save features to disk
- [ ] **Milestone 4.2:** SVM integration
  - LIBSVM library integrated
  - Train SVM on extracted features
  - Hyperparameters tuned (C=10, gamma=auto)

**Verification Test:**

```bash
./bin/extract_features --model models/encoder_weights.bin
# Expected: features_train.txt (50K × 8192)
# Expected: features_test.txt (10K × 8192)
# Expected: Extraction time <20 seconds

./bin/train_svm --features features_train.txt
# Expected: SVM training converges
# Expected: Training time <5 minutes
```

#### **Day 11-12: December 8-9** (Phase 4: Evaluation)

**Focus:** Classification and analysis

- [ ] **Milestone 4.3:** Classification results
  - Test set predictions
  - Accuracy: 60-65%
  - Confusion matrix generated
  - Per-class accuracy analyzed

**Verification Test:**

```bash
./bin/evaluate_svm --model svm_model.bin
# Expected: Test accuracy 60-65%
# Expected: Confusion matrix saved
# Expected: Per-class breakdown displayed
```

**Deliverable:** Phase 4 complete, end-to-end pipeline working

---

#### **Day 12-13: December 9-10** (Final Deliverables)

**Focus:** Documentation and presentation

- [ ] **Milestone 5.1:** Report complete
  - Jupyter notebook with all results
  - Runnable on Google Colab
  - PDF exported
- [ ] **Milestone 5.2:** Code package ready
  - All source code organized
  - README with setup instructions
  - Model weights included (or linked)
- [ ] **Milestone 5.3:** Presentation video
  - 15-20 minute video
  - Live demo included
  - Uploaded to YouTube (Unlisted)

**Final Verification:**

- [ ] Run full pipeline on Colab from scratch
- [ ] Verify all performance targets met
- [ ] Check all deliverables against rubric

**Deliverable:** Complete submission package ready

---

## 3. Phase Breakdown

### Phase 1: CPU Baseline (Nov 28-Dec 1, 4 days)

**Objectives:**

- Establish working data pipeline
- Implement all neural network operations on CPU
- Create baseline for GPU comparison
- Verify mathematical correctness

**Key Tasks:**

1. CIFAR-10 data loader (binary format parsing)
2. CPU layer implementations (Conv2D, ReLU, MaxPool, UpSample)
3. Autoencoder class (forward/backward pass)
4. Training loop with SGD optimizer
5. Weight save/load functionality

**Exit Criteria:**

- ✓ Training completes successfully for 2-5 epochs
- ✓ Loss decreases monotonically
- ✓ Reconstructed images visually similar to inputs
- ✓ Baseline timing recorded (~15-20 min/epoch)

---

### Phase 2: Naive GPU Implementation (Dec 2-4, 3 days)

**Objectives:**

- Port all operations to GPU
- Establish GPU correctness
- Measure initial GPU speedup
- Identify optimization opportunities

**Key Tasks:**

1. GPU memory management (cudaMalloc, cudaMemcpy)
2. Basic CUDA kernels for all layers
3. GPU forward/backward pass
4. GPU training loop
5. Profiling with nvprof/Nsight Compute

**Exit Criteria:**

- ✓ GPU output matches CPU (tolerance <1e-5)
- ✓ Training converges to same loss as CPU
- ✓ Initial speedup: 5-10× vs CPU
- ✓ No CUDA errors or memory leaks

---

### Phase 3: Advanced Optimization (Dec 5-7, 3 days)

**Objectives:**

- Apply systematic GPU optimizations
- Achieve >50× speedup target
- Document optimization impact

**Optimization Priority:**

**Version 1: Shared Memory (Dec 5-6, 1.5 days)**

- Tile-based convolution
- Shared memory for kernel weights
- Coalesced global memory access
- **Target:** 20-40× cumulative speedup

**Version 2: Kernel Fusion (Dec 6-7, 1 day)**

- Conv+ReLU combined kernel
- Conv+ReLU+Bias combined kernel
- Reduced kernel launch overhead
- **Target:** 40-60× cumulative speedup

**Version 3: Multi-Stream Pipeline (Dec 7, 0.5 days, Optional)**

- Concurrent H2D, compute, D2H
- Double/triple buffering
- Stream synchronization
- **Target:** 60-80× cumulative speedup

**Exit Criteria:**

- ✓ At least 2 optimization versions implemented
- ✓ Cumulative speedup >50× vs CPU
- ✓ Profiling data for each version
- ✓ Training time <1 minute total for 20 epochs

---

### Phase 4: SVM Integration (Dec 8, 1 day)

**Objectives:**

- Extract features using trained encoder
- Train SVM classifier
- Evaluate classification performance
- Analyze results

**Key Tasks:**

1. Feature extraction (encoder forward pass only)
2. LIBSVM integration
3. Hyperparameter tuning (grid search on C, gamma)
4. Test set evaluation
5. Confusion matrix and per-class analysis

**Exit Criteria:**

- ✓ Feature extraction time <20 seconds
- ✓ Test accuracy 60-65%
- ✓ Confusion matrix generated
- ✓ Results documented in report

---

## 4. Resource Requirements

### 4.1 Hardware

**Local Development:**

- CPU: Multi-core x86_64 (for baseline)
- GPU: NVIDIA GPU with CUDA Compute Capability ≥6.0
  - Recommended: RTX 3060+ or T4+
  - Minimum: GTX 1660 or equivalent
- RAM: 16GB+ (for 50K images in memory)
- Storage: 10GB (dataset + models + code)

**Google Colab:**

- GPU Runtime: T4 (free tier) or V100/A100 (Pro)
- RAM: 12-25GB depending on tier
- Persistent storage: Google Drive integration

### 4.2 Software

**Required:**

- CUDA Toolkit 11.0+ (11.8 recommended)
- CMake 3.18+
- GCC/G++ 9+ or Clang 10+
- Python 3.8+ (for Jupyter notebook)
- Git

**Libraries:**

- LIBSVM (https://github.com/cjlin1/libsvm)
- OpenMP (typically included with compiler)

**Optional:**

- cuDNN 8.0+ (for potential cuDNN comparisons)
- Nsight Compute/Systems (for profiling)

### 4.3 Dataset

**CIFAR-10 Binary Version:**

- Download: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
- Size: ~163 MB compressed, ~168 MB uncompressed
- Files: 5 training batches + 1 test batch
- Format: Binary (uint8)

---

## 5. Risk Management

### 5.1 Technical Risks

| Risk                                     | Probability | Impact | Mitigation                                          |
| ---------------------------------------- | ----------- | ------ | --------------------------------------------------- |
| CUDA version incompatibility             | Medium      | High   | Test on Colab early, use CUDA 11.8                  |
| Memory overflow on GPU                   | Medium      | Medium | Implement gradient checkpointing if needed          |
| Optimization doesn't improve performance | Low         | Medium | Have fallback optimizations ready                   |
| LIBSVM integration issues                | Low         | Low    | Test early, use simple C interface                  |
| Colab GPU quota exhausted                | Medium      | High   | Use local GPU for development, Colab for final runs |

### 5.2 Schedule Risks

| Risk                                     | Probability | Impact | Mitigation                                   |
| ---------------------------------------- | ----------- | ------ | -------------------------------------------- |
| Phase 1 takes longer than expected       | Low         | Medium | Phase 1 is well-understood, start early      |
| Optimization debugging is time-consuming | High        | Medium | Allocate buffer days in Phase 3              |
| SVM accuracy below target                | Medium      | High   | Tune hyperparameters, try different features |
| Report writing takes longer              | Medium      | Medium | Write incrementally, document as you go      |

### 5.3 Contingency Plans

**If GPU not available:**

- Use Google Colab exclusively (max 12 hours/session)
- Request lab GPU access from instructor
- Reduce training epochs or dataset size

**If accuracy below 60%:**

- Increase autoencoder training epochs (20 → 30-40)
- Try different SVM kernels (linear, polynomial)
- Experiment with feature normalization
- Verify autoencoder is learning (check reconstructions)

**If optimization stuck:**

- Focus on 2 solid optimizations rather than 3 weak ones
- Prioritize shared memory and kernel fusion
- Document attempted optimizations even if unsuccessful

---

## 6. Success Criteria

### 6.1 Performance Targets

| Metric                        | Target                   | Measurement                    |
| ----------------------------- | ------------------------ | ------------------------------ |
| **Autoencoder Training Time** | <10 minutes (20 epochs)  | Wall-clock time on GPU         |
| **Feature Extraction Time**   | <20 seconds (60K images) | Wall-clock time                |
| **GPU Speedup vs CPU**        | >20× (stretch: >50×)     | Ratio of CPU time / GPU time   |
| **Classification Accuracy**   | 60-65%                   | Test set accuracy (10K images) |
| **Reconstruction Loss**       | <0.05 (final epoch)      | MSE between input and output   |

### 6.2 Code Quality

- [ ] All code compiles without warnings (`-Wall -Wextra`)
- [ ] No memory leaks (verified with `cuda-memcheck`)
- [ ] No CUDA errors (checked with `cudaGetLastError()`)
- [ ] Modular, well-commented code
- [ ] README with clear setup instructions

### 6.3 Documentation

- [ ] Jupyter notebook runnable on Colab
- [ ] All figures and tables generated from code
- [ ] Performance comparison tables for all phases
- [ ] Confusion matrix and per-class accuracy
- [ ] Lessons learned section completed

### 6.4 Presentation

- [ ] 15-20 minute video (not shorter than 15, not longer than 20)
- [ ] Live demo showing training and inference
- [ ] Clear explanation of optimizations
- [ ] YouTube link included in submission

---

## 7. Daily Checklist Template

Use this for daily standups/progress tracking:

### Date: ****\_\_\_****

**Yesterday's Progress:**

- [ ] Milestone completed: ****\_\_****
- [ ] Code commits: ****\_\_****
- [ ] Tests passed: ****\_\_****

**Today's Plan:**

- [ ] Task 1: ****\_\_****
- [ ] Task 2: ****\_\_****
- [ ] Task 3: ****\_\_****

**Blockers:**

- [ ] None / [Description]

**Metrics:**

- Code: \_\_\_ lines added/modified
- Tests: **_ passing / _** total
- Time spent: \_\_\_ hours

---

## 8. References and Resources

### 8.1 Project Documentation

- **Requirements:** `docs/CSC14120_2025_Final_Project.md`
- **Phase 1 Guide:** `docs/PHASE_1_GUIDE.md`
- **Phase 2 Guide:** `docs/PHASE_2_GUIDE.md`
- **Phase 3 Guide:** `docs/PHASE_3_GUIDE.md`
- **Phase 4 Guide:** `docs/PHASE_4_GUIDE.md`
- **Testing Guide:** `docs/TESTING_DELIVERABLES.md`

### 8.2 External References

**Autoencoders:**

- Hinton & Salakhutdinov (2006): https://www.cs.toronto.edu/~hinton/science.pdf
- Deep Learning Book, Ch 14: https://www.deeplearningbook.org/contents/autoencoders.html

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

## 9. Team Communication

### 9.1 Meeting Schedule

- **Daily Standup:** 10:00 AM (15 minutes)

  - What did you complete yesterday?
  - What will you work on today?
  - Any blockers?

- **Mid-Week Review:** Wednesday 4:00 PM (30 minutes)

  - Review progress against milestones
  - Adjust timeline if needed
  - Pair programming for blockers

- **Weekly Demo:** Saturday 2:00 PM (1 hour)
  - Demo working features
  - Discuss challenges
  - Plan next week

### 9.2 Communication Channels

- **Urgent Issues:** Group chat (Discord/Telegram)
- **Code Reviews:** GitHub Pull Requests
- **Documentation:** Shared Google Drive
- **Questions:** Email + office hours

---

## 10. Final Submission Checklist

**Code Package:**

- [ ] All source files (.cpp, .cu, .h)
- [ ] CMakeLists.txt (builds successfully)
- [ ] README.md (setup instructions)
- [ ] Trained model weights (or download link)
- [ ] No generated/binary files in repo

**Report:**

- [ ] Jupyter notebook (.ipynb)
- [ ] Runnable on Google Colab
- [ ] All figures/tables generated
- [ ] PDF export included
- [ ] All sections complete per rubric

**Presentation:**

- [ ] Video 15-20 minutes long
- [ ] YouTube link (Unlisted)
- [ ] Link in README and notebook
- [ ] Covers all required topics

**Verification:**

- [ ] Full pipeline runs end-to-end
- [ ] All performance targets met
- [ ] No compilation warnings
- [ ] No CUDA errors

---

**Good luck with the project! Follow the phase guides for detailed implementation steps.**
