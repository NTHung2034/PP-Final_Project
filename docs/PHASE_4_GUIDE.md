# Phase 4: SVM Integration and Result Analysis Guide

**Duration:** December 8, 2025 (1 day)  
**Objective:** Complete end-to-end pipeline with classification and evaluation  
**Prerequisite:** Phase 3 complete, trained autoencoder weights saved

---

## Table of Contents

1. [Overview](#1-overview)
2. [Feature Extraction](#2-feature-extraction)
3. [LIBSVM Integration](#3-libsvm-integration)
4. [Classification and Evaluation](#4-classification-and-evaluation)
5. [Result Analysis](#5-result-analysis)
6. [Google Colab Notes](#6-google-colab-notes)

---

## 1. Overview

### 1.1 Goals

By the end of Phase 4, you will have:

- ✅ Feature extraction working (8,192-dim vectors)
- ✅ LIBSVM integrated and trained
- ✅ 60-65% test accuracy achieved
- ✅ Confusion matrix and per-class analysis
- ✅ Complete end-to-end pipeline demonstrated

### 1.2 Pipeline Recap

```
TRAINING PHASE:
  50K Images → Autoencoder (GPU) → 50K × 8,192 features
                                         ↓
                                    + Labels
                                         ↓
                                  SVM Training (CPU)
                                         ↓
                                   SVM Model

TESTING PHASE:
  10K Images → Encoder (GPU) → 10K × 8,192 features
                                         ↓
                                   SVM Model
                                         ↓
                                  Predictions → Accuracy
```

### 1.3 Expected Performance

| Metric                   | Target Value             |
| ------------------------ | ------------------------ |
| Feature extraction time  | <20 seconds (60K images) |
| SVM training time        | 2-5 minutes              |
| Test accuracy            | 60-65%                   |
| Feature extraction/image | <0.3 ms                  |

---

## 2. Feature Extraction

### **Task 4.1: Implement Feature Extraction (Morning)**

#### Step 4.1.1: Create Feature Extractor Class

**File:** `include/models/feature_extractor.cuh`

```cpp
#pragma once
#include "models/autoencoder_gpu.cuh"
#include <string>
#include <vector>

class FeatureExtractor {
public:
    FeatureExtractor(const std::string& encoder_weights_path);
    ~FeatureExtractor();

    // Extract features for a batch of images
    // Input: [batch_size, 3, 32, 32] on host
    // Output: [batch_size, 8192] on host
    void extract_batch(const float* h_images, float* h_features,
                       int batch_size);

    // Extract features for entire dataset
    // Saves to file in LIBSVM format
    void extract_dataset(const std::string& data_path,
                         const std::string& output_path,
                         bool is_train_set = true);

    // Get feature dimension
    int get_feature_dim() const { return 8 * 8 * 128; }  // 8192

private:
    std::unique_ptr<AutoencoderGPU> model_;
    std::unique_ptr<GPUTensor> latent_buffer_;
};
```

#### Step 4.1.2: Implement Encoder-Only Forward Pass

**File:** `src/models/feature_extractor.cu`

```cpp
#include "models/feature_extractor.cuh"
#include "data/cifar10_dataset.h"
#include "utils/logger.h"
#include <fstream>
#include <chrono>

FeatureExtractor::FeatureExtractor(const std::string& weights_path) {
    model_ = std::make_unique<AutoencoderGPU>();
    model_->load_weights(weights_path);

    LOG_INFO("Feature extractor initialized with weights from %s",
             weights_path.c_str());
}

FeatureExtractor::~FeatureExtractor() = default;

void FeatureExtractor::extract_batch(const float* h_images,
                                     float* h_features,
                                     int batch_size) {
    // Allocate latent buffer if needed
    if (!latent_buffer_ || latent_buffer_->batch() != batch_size) {
        latent_buffer_ = std::make_unique<GPUTensor>(
            std::vector<int>{batch_size, 128, 8, 8}
        );
    }

    // Run encoder only (no decoder)
    model_->extract_features(h_images, latent_buffer_->d_data, batch_size);

    // Copy features back to host
    latent_buffer_->to_host(h_features);
}

void FeatureExtractor::extract_dataset(const std::string& data_path,
                                       const std::string& output_path,
                                       bool is_train_set) {
    LOG_INFO("Extracting features from %s...", data_path.c_str());

    // Load dataset
    CIFAR10Dataset dataset(
        data_path,
        is_train_set ? CIFAR10Dataset::Mode::TRAIN
                     : CIFAR10Dataset::Mode::TEST
    );
    dataset.load_data();

    int num_images = dataset.size();
    int batch_size = 256;  // Large batch for feature extraction
    int num_batches = (num_images + batch_size - 1) / batch_size;

    LOG_INFO("Extracting %d images in %d batches", num_images, num_batches);

    // Open output file
    std::ofstream out_file(output_path);
    if (!out_file) {
        throw std::runtime_error("Cannot open output file: " + output_path);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    dataset.reset();

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        int current_batch_size = std::min(batch_size,
                                          num_images - batch_idx * batch_size);

        // Get batch
        auto images = dataset.get_batch(current_batch_size);
        auto labels = dataset.get_batch_labels(current_batch_size);

        // Extract features
        std::vector<float> features(current_batch_size * 8192);
        extract_batch(images.data->data(), features.data(), current_batch_size);

        // Write to file in LIBSVM format: label feat1:val1 feat2:val2 ...
        for (int i = 0; i < current_batch_size; ++i) {
            out_file << labels[i];

            for (int j = 0; j < 8192; ++j) {
                float val = features[i * 8192 + j];
                // Only write non-zero features (sparse format)
                if (std::abs(val) > 1e-8) {
                    out_file << " " << (j + 1) << ":" << val;
                }
            }
            out_file << "\n";
        }

        if (batch_idx % 10 == 0) {
            LOG_INFO("Processed %d/%d batches", batch_idx + 1, num_batches);
        }
    }

    out_file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();

    LOG_INFO("Feature extraction complete in %ld seconds", duration);
    LOG_INFO("Saved to %s", output_path.c_str());
}
```

#### Step 4.1.3: Create Feature Extraction Script

**File:** `src/extract_features.cu`

```cpp
#include "models/feature_extractor.cuh"
#include "utils/logger.h"
#include "config.h"
#include <iostream>

int main(int argc, char** argv) {
    LOG_INIT();
    LOG_INFO("CIFAR-10 Feature Extraction");

    // Parse arguments
    std::string weights_path = MODEL_SAVE_DIR "/gpu_encoder_weights.bin";
    if (argc > 1) {
        weights_path = argv[1];
    }

    try {
        // Create feature extractor
        FeatureExtractor extractor(weights_path);

        // Extract training features
        LOG_INFO("Extracting training features...");
        extractor.extract_dataset(
            CIFAR_BIN_DIR,
            "data/train_features.txt",
            true  // training set
        );

        // Extract test features
        LOG_INFO("Extracting test features...");
        extractor.extract_dataset(
            CIFAR_BIN_DIR,
            "data/test_features.txt",
            false  // test set
        );

        LOG_INFO("Feature extraction complete!");
        LOG_INFO("Train features: data/train_features.txt (50000 samples)");
        LOG_INFO("Test features: data/test_features.txt (10000 samples)");

    } catch (const std::exception& e) {
        LOG_ERROR("Error: %s", e.what());
        return 1;
    }

    return 0;
}
```

**Build and run:**

```bash
cd build
make extract_features
./bin/extract_features
```

**Expected output:**

```
Feature extraction complete in 18 seconds
Train features: data/train_features.txt (50000 samples)
Test features: data/test_features.txt (10000 samples)
```

---

## 3. LIBSVM Integration

### **Task 4.2: Setup and Train SVM (Afternoon)**

#### Step 4.2.1: Download and Build LIBSVM

**Download:**

```bash
cd external
git clone https://github.com/cjlin1/libsvm.git
cd libsvm
make
```

**Verify installation:**

```bash
./svm-train
# Should show usage instructions
```

#### Step 4.2.2: Train SVM with Default Parameters

**Simple training script:**

```bash
#!/bin/bash
# scripts/train_svm.sh

TRAIN_FILE="data/train_features.txt"
MODEL_FILE="models/svm_model.bin"

echo "Training SVM on extracted features..."
time ./external/libsvm/svm-train \
    -s 0 \      # C-SVC (classification)
    -t 2 \      # RBF kernel
    -c 10 \     # Cost parameter
    -g 0.0001 \ # Gamma (1/num_features ≈ 1/8192)
    -q \        # Quiet mode
    $TRAIN_FILE \
    $MODEL_FILE

echo "SVM training complete!"
echo "Model saved to $MODEL_FILE"
```

**Run:**

```bash
chmod +x scripts/train_svm.sh
./scripts/train_svm.sh
```

**Expected output:**

```
Training SVM on extracted features...
*
optimization finished, #iter = 12543
...
SVM training complete in 3m 24s
Model saved to models/svm_model.bin
```

#### Step 4.2.3: Hyperparameter Tuning (Optional)

**Grid search script:** `scripts/svm_grid_search.sh`

```bash
#!/bin/bash

TRAIN_FILE="data/train_features.txt"
OUTPUT_FILE="results/svm_grid_search.log"

echo "Running SVM grid search..." > $OUTPUT_FILE

# Grid search over C and gamma
for C in 1 10 100; do
    for gamma in 0.00001 0.0001 0.001; do
        echo "Testing C=$C, gamma=$gamma" >> $OUTPUT_FILE

        ./external/libsvm/svm-train \
            -v 5 \  # 5-fold cross-validation
            -s 0 -t 2 \
            -c $C -g $gamma \
            -q \
            $TRAIN_FILE >> $OUTPUT_FILE 2>&1

        echo "" >> $OUTPUT_FILE
    done
done

echo "Grid search complete. Results in $OUTPUT_FILE"
cat $OUTPUT_FILE | grep "Cross Validation Accuracy"
```

**Recommended parameters based on literature:**

- **C (Cost):** 10 (regularization parameter)
- **gamma:** 1 / num_features ≈ 0.0001 (RBF kernel width)
- **kernel:** RBF (works well for high-dimensional features)

---

## 4. Classification and Evaluation

### **Task 4.3: Test SVM and Evaluate (Afternoon)**

#### Step 4.3.1: Predict on Test Set

**Prediction script:** `scripts/predict_svm.sh`

```bash
#!/bin/bash

TEST_FILE="data/test_features.txt"
MODEL_FILE="models/svm_model.bin"
OUTPUT_FILE="results/test_predictions.txt"

echo "Predicting on test set..."
./external/libsvm/svm-predict \
    $TEST_FILE \
    $MODEL_FILE \
    $OUTPUT_FILE

echo "Predictions saved to $OUTPUT_FILE"
```

**Run:**

```bash
./scripts/predict_svm.sh
```

**Expected output:**

```
Accuracy = 62.34% (6234/10000)
Predictions saved to results/test_predictions.txt
```

#### Step 4.3.2: Create Evaluation Script

**File:** `scripts/evaluate_results.py`

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_labels(filename):
    """Load true labels from feature file."""
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            label = int(line.split()[0])
            labels.append(label)
    return np.array(labels)

def load_predictions(filename):
    """Load predictions from LIBSVM output."""
    predictions = []
    with open(filename, 'r') as f:
        for line in f:
            pred = int(float(line.strip()))
            predictions.append(pred)
    return np.array(predictions)

def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('CIFAR-10 Classification Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")

    return cm

def compute_per_class_accuracy(cm):
    """Compute per-class accuracy from confusion matrix."""
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return per_class_acc

def main():
    print("Loading data...")
    y_true = load_labels('data/test_features.txt')
    y_pred = load_predictions('results/test_predictions.txt')

    # Overall accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    per_class_acc = compute_per_class_accuracy(cm)
    for i, acc in enumerate(per_class_acc):
        print(f"  {class_names[i]:12s}: {acc * 100:5.2f}%")

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save results
    with open('results/evaluation_summary.txt', 'w') as f:
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Per-Class Accuracy:\n")
        for i, acc in enumerate(per_class_acc):
            f.write(f"  {class_names[i]:12s}: {acc * 100:5.2f}%\n")
        f.write("\n" + classification_report(y_true, y_pred,
                                             target_names=class_names))

    print("\nResults saved to results/evaluation_summary.txt")

if __name__ == '__main__':
    main()
```

**Run:**

```bash
python scripts/evaluate_results.py
```

**Expected output:**

```
Overall Test Accuracy: 62.34%

Per-Class Accuracy:
  airplane    : 68.20%
  automobile  : 71.50%
  bird        : 51.30%
  cat         : 44.80%
  deer        : 57.90%
  dog         : 55.60%
  frog        : 72.10%
  horse       : 67.40%
  ship        : 73.80%
  truck       : 70.20%

Confusion matrix saved to results/confusion_matrix.png
Results saved to results/evaluation_summary.txt
```

---

## 5. Result Analysis

### 5.1 Understanding the Results

**Expected Accuracy Range: 60-65%**

This is reasonable for unsupervised feature learning because:

1. No labels used during autoencoder training
2. Features learned purely from reconstruction objective
3. Linear SVM classifier (not deep neural network)

**Comparison with baselines:**

- Random guessing: 10%
- Linear SVM on raw pixels: ~40%
- **Autoencoder + SVM: 60-65%** ✓
- Supervised CNN (state-of-art): 95%+

### 5.2 Typical Per-Class Performance

**Easier classes (70%+ accuracy):**

- **Ship, Truck, Automobile, Frog:** Distinctive shapes and textures
- High inter-class variation
- Autoencoder captures shape well

**Harder classes (45-55% accuracy):**

- **Cat, Dog, Bird:** Similar fur/feather textures
- High intra-class variation (pose, color)
- Often confused with each other

**Confusion matrix insights:**

- Cat ↔ Dog confusion common
- Bird ↔ Airplane some confusion
- Truck ↔ Automobile some confusion

### 5.3 Feature Quality Analysis

**Visualize feature space (t-SNE):**

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_features():
    # Load features (first 1000 for speed)
    features = []
    labels = []
    with open('data/test_features.txt', 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000: break
            parts = line.split()
            labels.append(int(parts[0]))
            feat = np.zeros(8192)
            for item in parts[1:]:
                idx, val = item.split(':')
                feat[int(idx) - 1] = float(val)
            features.append(feat)

    features = np.array(features)
    labels = np.array(labels)

    # t-SNE projection
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(10),
                 label='Class')
    plt.title('t-SNE Visualization of Learned Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('results/tsne_features.png', dpi=300)
    print("Saved to results/tsne_features.png")

if __name__ == '__main__':
    visualize_features()
```

**Good features show:**

- Distinct clusters for each class
- Minimal overlap between dissimilar classes
- Some overlap for similar classes (cat/dog) is expected

### 5.4 Autoencoder Reconstruction Quality

**Visualize reconstructions:**

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_reconstructions():
    # Load a few test images
    from data.cifar10_dataset import CIFAR10Dataset
    from models.autoencoder_gpu import AutoencoderGPU

    dataset = CIFAR10Dataset('../data/cifar-10-batches-bin', is_train=False)
    dataset.load_data()

    model = AutoencoderGPU()
    model.load_weights('../models/gpu_encoder_weights.bin')

    # Get 10 random images
    batch = dataset.get_batch(10)

    # Reconstruct
    reconstructed = model.forward(batch.data)

    # Plot
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        # Original
        img_orig = batch.data[i].transpose(1, 2, 0)  # CHW → HWC
        axes[0, i].imshow(img_orig)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        # Reconstructed
        img_recon = reconstructed[i].transpose(1, 2, 0)
        axes[1, i].imshow(img_recon)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12)

    plt.tight_layout()
    plt.savefig('results/reconstructions.png', dpi=300)
    print("Saved to results/reconstructions.png")
```

**Good reconstructions:**

- Preserve overall shape and color
- May lose fine details (expected for 8×8 bottleneck)
- No artifacts or noise

---

## 6. Google Colab Notes

### 6.1 Full Pipeline on Colab

**Notebook cell structure:**

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_REPO/PP-Final_Project.git
%cd PP-Final_Project
!bash scripts/setup_colab.sh

# Cell 2: Train autoencoder (or load pretrained)
!./bin/train_gpu  # Takes ~5-10 minutes with Phase 3 optimizations

# Cell 3: Extract features
!./bin/extract_features
# Output: train_features.txt, test_features.txt

# Cell 4: Install LIBSVM
!cd external && git clone https://github.com/cjlin1/libsvm.git
!cd external/libsvm && make

# Cell 5: Train SVM
!bash scripts/train_svm.sh
# Takes 2-5 minutes

# Cell 6: Evaluate
!bash scripts/predict_svm.sh
!python scripts/evaluate_results.py

# Cell 7: Visualize results
from IPython.display import Image, display
display(Image('results/confusion_matrix.png'))
display(Image('results/tsne_features.png'))
```

### 6.2 Save Results to Drive

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/drive/MyDrive/CIFAR10_Results
!cp results/* /content/drive/MyDrive/CIFAR10_Results/
!cp models/*.bin /content/drive/MyDrive/CIFAR10_Results/

print("Results saved to Google Drive!")
```

---

## 7. Common Issues and Solutions

### 7.1 Low Accuracy (<55%)

**Possible causes:**

1. **Autoencoder not trained well**

   - Check reconstruction loss (should be <0.05)
   - Visualize reconstructions
   - Train for more epochs

2. **Features not normalized**

   - Try feature scaling: `svm-scale` before training

   ```bash
   ./external/libsvm/svm-scale -s scale_params train_features.txt > train_scaled.txt
   ./external/libsvm/svm-scale -r scale_params test_features.txt > test_scaled.txt
   ```

3. **Poor SVM hyperparameters**
   - Run grid search
   - Try different kernels (linear, polynomial)

### 7.2 SVM Training Very Slow (>10 minutes)

**Solutions:**

1. **Use subset for initial experiments**

   ```bash
   head -10000 train_features.txt > train_subset.txt
   ```

2. **Try linear SVM (faster)**

   ```bash
   ./external/libsvm/svm-train -s 0 -t 0 ...  # Linear kernel
   ```

3. **Use LIBLINEAR instead**
   - Much faster for linear classification
   - Download: https://github.com/cjlin1/liblinear

### 7.3 Memory Issues

**For large feature files:**

```python
# Extract features in smaller batches
extractor.extract_dataset(data_path, output_path, batch_size=128)
```

**Use sparse format (already implemented):**

- Only non-zero features written
- Reduces file size by 30-50%

---

## 8. Deliverables Checklist

- [ ] Feature extraction working (<20 seconds)
- [ ] LIBSVM integrated and trained
- [ ] Test accuracy 60-65%
- [ ] Confusion matrix generated and analyzed
- [ ] Per-class accuracy computed
- [ ] Reconstruction quality visualized
- [ ] Feature space visualization (t-SNE)
- [ ] All results saved to `results/` directory

---

## 9. Report Content for Phase 4

Include in your Jupyter notebook:

**Section 4.1: Feature Extraction**

- Code snippet for extraction
- Timing results (should be <20s)
- Feature statistics (mean, std, sparsity)

**Section 4.2: SVM Training**

- Hyperparameters used
- Training time
- Cross-validation results (if done)

**Section 4.3: Classification Results**

- Overall accuracy (with comparison to baseline)
- Confusion matrix (figure)
- Per-class accuracy table
- Analysis of hardest/easiest classes

**Section 4.4: Qualitative Analysis**

- Sample reconstructions (figure)
- t-SNE visualization (figure)
- Discussion of learned features
- Comparison with supervised methods

---

## 10. References

**LIBSVM:**

- Official guide: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
- Parameter selection: https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf

**Feature Learning:**

- Coates et al., "An Analysis of Single-Layer Networks in Unsupervised Feature Learning" (2011)
- Hinton & Salakhutdinov, "Reducing the Dimensionality of Data with Neural Networks" (2006)

**SVM for Image Classification:**

- Support Vector Machines for Image Classification: https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html

---

**Congratulations on completing Phase 4! You now have a working end-to-end pipeline. Next: Final testing and report writing!**
