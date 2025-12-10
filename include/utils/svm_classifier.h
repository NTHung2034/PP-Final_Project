#pragma once
#include "data/data_types.h"
#include <string>
#include <vector>

// Forward declaration to avoid including full LIBSVM headers
struct svm_model;
struct svm_parameter;
struct svm_problem;
struct svm_node;

namespace SVMClassifier
{
    /**
     * Simple SVM wrapper for CIFAR-10 classification using LIBSVM
     * - Trains RBF kernel SVM on extracted features
     * - Predicts labels for test features
     * - Evaluates classification accuracy
     */

    // Train SVM on features (in-memory)
    // features: N x feature_dim
    // labels: N labels (0-9 for CIFAR-10)
    // Returns: trained SVM model pointer (caller must free with free_model)
    svm_model *train_svm(
        const std::vector<std::vector<float>> &features,
        const std::vector<int> &labels,
        double C = 10.0,
        double gamma = 0.0001);

    // Predict labels for test features
    // Returns: predicted labels (0-9)
    std::vector<int> predict(
        const svm_model *model,
        const std::vector<std::vector<float>> &features);

    // Calculate classification accuracy
    float calculate_accuracy(
        const std::vector<int> &true_labels,
        const std::vector<int> &predicted_labels);

    // Calculate per-class accuracy
    std::vector<float> calculate_per_class_accuracy(
        const std::vector<int> &true_labels,
        const std::vector<int> &predicted_labels,
        int num_classes = 10);

    // Free SVM model
    void free_model(svm_model *model);

    // Save/load SVM model
    void save_model(const svm_model *model, const std::string &filename);
    svm_model *load_model(const std::string &filename);

} // namespace SVMClassifier
