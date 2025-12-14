#!/usr/bin/env python3

import numpy as np
import cuml
from cuml.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def load_binary_data(feature_file, label_file, num_samples, feature_dim):
    features = np.fromfile(feature_file, dtype=np.float32)
    features = features.reshape(num_samples, feature_dim)
    
    labels = np.fromfile(label_file, dtype=np.uint8)
    
    print(f"Loaded features shape: {features.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    
    return features, labels

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - CIFAR-10 SVM Classification', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def get_per_class_accuracy(cm, class_names):
    result = "\n=== Per-Class Accuracy ===\n"
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        acc = (cm[i, i] / total * 100) if total > 0 else 0
        result += f"{name:>15}: {acc:>6.2f}%\n"
    return result

def main():
    # Configuration - determine implementation version
    impl_version = "naive"  # default
    if len(sys.argv) > 1:
        impl_version = sys.argv[1].lower()
    
    # Validate implementation version
    valid_versions = ["naive", "opt_v1", "opt_v2"]
    if impl_version not in valid_versions:
        print(f"Error: Invalid implementation version '{impl_version}'")
        print(f"Valid options: {', '.join(valid_versions)}")
        return 1
    
    # Set base directory based on implementation version
    version_dir_map = {
        "naive": "saved_weights_gpu_naive",
        "opt_v1": "saved_weights_gpu_opt_v1",
        "opt_v2": "saved_weights_gpu_opt_v2"
    }
    
    base_dir = f"/content/PP-Final_Project/models/{version_dir_map[impl_version]}/svm_features"
    
    print("\n" + "="*70)
    print(f"CIFAR-10 cudaML SVM Training with RBF Kernel [{impl_version.upper()}]")
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
    print(f"  Implementation: {impl_version.upper()}")
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
    std[std < 1e-8] = 1.0 
    
    # Normalize both train and test
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    print(f"Features normalized")
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
    
    # Visualize confusion matrix
    cm_plot_path = os.path.join(base_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_plot_path)
    
    # Per-class accuracy
    per_class_acc = get_per_class_accuracy(cm, class_names)
    print(per_class_acc)
    
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
    summary_text = "\n" + "="*70 + "\n"
    summary_text += "TRAINING SUMMARY\n"
    summary_text += "="*70 + "\n"
    summary_text += f"Configuration:\n"
    summary_text += f"  Implementation: {impl_version.upper()}\n"
    summary_text += f"  Kernel: {kernel}\n"
    summary_text += f"  C: {C}\n"
    summary_text += f"  Gamma: {gamma}\n\n"
    summary_text += f"Training time: {train_time:.2f}s\n"
    summary_text += f"Training accuracy: {train_accuracy * 100:.2f}%\n"
    summary_text += f"Test accuracy: {test_accuracy * 100:.2f}%\n"
    summary_text += "="*70 + "\n\n"
    summary_text += per_class_acc + "\n"
    summary_text += "\n=== Classification Report ===\n"
    summary_text += report + "\n"
    
    print(summary_text)
    
    # Save summary to text file
    summary_file = os.path.join(base_dir, "training_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    print(f"Training summary saved to: {summary_file}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

