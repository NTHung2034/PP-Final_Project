#!/usr/bin/env python3
"""
cudaML SVM Training Script
Trains SVM classifier on extracted features using GPU-accelerated cudaML library
"""

import numpy as np
import cuml
from cuml.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os
import sys

def load_binary_data(feature_file, label_file, num_samples, feature_dim):
    """Load binary feature and label files"""
    features = np.fromfile(feature_file, dtype=np.float32)
    features = features.reshape(num_samples, feature_dim)
    
    labels = np.fromfile(label_file, dtype=np.uint8)
    
    print(f"Loaded features shape: {features.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    
    return features, labels

def print_confusion_matrix(cm, class_names):
    """Print formatted confusion matrix"""
    print("\n=== Confusion Matrix ===")
    print(f"{'':>12}", end='')
    for name in class_names:
        print(f"{name[:10]:>12}", end='')
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name[:10]:>12}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>12}", end='')
        print()

def print_per_class_accuracy(cm, class_names):
    """Print per-class accuracy"""
    print("\n=== Per-Class Accuracy ===")
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        acc = (cm[i, i] / total * 100) if total > 0 else 0
        print(f"{name:>15}: {acc:>6.2f}%")

def main():
    # Configuration
    base_dir = "../../models/saved_weights_gpu_naive/svm_features"
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    
    print("\n" + "="*70)
    print("CIFAR-10 cudaML SVM Training with RBF Kernel")
    print("="*70 + "\n")
    
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    # Hyperparameters
    C = 10.0
    gamma = 'auto'
    kernel = 'rbf'
    
    print(f"Configuration:")
    print(f"  Data directory: {base_dir}")
    print(f"  Kernel: {kernel}")
    print(f"  C: {C}")
    print(f"  Gamma: {gamma}")
    print()
    
    # Check if files exist
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        print("Please run feature extraction first.")
        return 1
    
    # Dataset dimensions (CIFAR-10 specific)
    num_train_samples = 50000
    num_test_samples = 10000
    feature_dim = 8192  # 128 * 8 * 8
    
    # ====== Step 1: Load Data ======
    print("=== Step 1: Loading Data ===")
    
    train_features_file = os.path.join(base_dir, "train_features.bin")
    train_labels_file = os.path.join(base_dir, "train_labels.bin")
    test_features_file = os.path.join(base_dir, "test_features.bin")
    test_labels_file = os.path.join(base_dir, "test_labels.bin")
    
    X_train, y_train = load_binary_data(train_features_file, train_labels_file, 
                                        num_train_samples, feature_dim)
    X_test, y_test = load_binary_data(test_features_file, test_labels_file,
                                      num_test_samples, feature_dim)
    
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    
    # ====== Feature Normalization ======
    print("\nNormalizing features...")
    
    # Compute mean and std from training set
    mean = np.mean(X_train, axis=0, dtype=np.float32)
    std = np.std(X_train, axis=0, dtype=np.float32)
    std[std < 1e-8] = 1.0  # Avoid division by zero
    
    # Normalize both train and test
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    print(f"Features normalized (mean=0, std=1)")
    print()
    
    # ====== Step 2: Train SVM ======
    print("=== Step 2: Training SVM with cudaML ===")
    print("Initializing SVM classifier...")
    
    svm_classifier = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        cache_size=2000,
        max_iter=-1,
        verbose=True
    )
    
    print("Training started...")
    train_start = time.time()
    
    svm_classifier.fit(X_train, y_train)
    
    train_end = time.time()
    train_time = train_end - train_start
    
    print(f"Training completed in {train_time:.2f}s")
    print()
    
    # ====== Step 3: Evaluation ======
    print("=== Step 3: Evaluation ===")
    
    # Batch size for prediction to avoid memory issues
    pred_batch_size = 1000
    
    # Training accuracy with batch processing
    print("Evaluating on training set...")
    train_pred = np.zeros(len(y_train), dtype=np.int32)
    for i in range(0, len(y_train), pred_batch_size):
        end_idx = min(i + pred_batch_size, len(y_train))
        train_pred[i:end_idx] = svm_classifier.predict(X_train[i:end_idx])
        if (i // pred_batch_size + 1) % 10 == 0:
            print(f"  Processed {end_idx}/{len(y_train)} samples")
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Test accuracy with batch processing
    print("Evaluating on test set...")
    test_pred = np.zeros(len(y_test), dtype=np.int32)
    for i in range(0, len(y_test), pred_batch_size):
        end_idx = min(i + pred_batch_size, len(y_test))
        test_pred[i:end_idx] = svm_classifier.predict(X_test[i:end_idx])
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    print_confusion_matrix(cm, class_names)
    print_per_class_accuracy(cm, class_names)
    
    # Classification report
    print("\n=== Classification Report ===")
    report = classification_report(y_test, test_pred, 
                                   target_names=class_names,
                                   digits=4)
    print(report)
    
    # ====== Step 4: Save Model ======
    print("\n=== Step 4: Saving Model ===")
    
    model_file = os.path.join(base_dir, "cuml_svm_model.pkl")
    try:
        import pickle
        with open(model_file, 'wb') as f:
            pickle.dump(svm_classifier, f)
        print(f"Model saved to: {model_file}")
    except Exception as e:
        print(f"Warning: Could not save model: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Training time: {train_time:.2f}s")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
