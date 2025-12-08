#include "utils/svm_classifier.h"
#include "utils/logger.h"
#include "../../external/libsvm/svm.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace SVMClassifier
{

    svm_model *train_svm(
        const std::vector<std::vector<float>> &features,
        const std::vector<int> &labels,
        double C,
        double gamma)
    {
        if (features.empty() || labels.empty() || features.size() != labels.size())
        {
            LOG_ERROR("Invalid input: features or labels empty, or size mismatch");
            return nullptr;
        }

        int n_samples = features.size();
        int n_features = features[0].size();

        LOG_INFO("Training SVM with %d samples, %d features", n_samples, n_features);
        std::cout << "      SVM parameters: C=" << C << ", gamma=" << gamma << "\n";

        // Prepare SVM problem
        svm_problem *prob = new svm_problem;
        prob->l = n_samples;
        prob->y = new double[n_samples];
        prob->x = new svm_node *[n_samples];

        // Convert features to LIBSVM format
        for (int i = 0; i < n_samples; ++i)
        {
            prob->y[i] = labels[i];

            // Count non-zero features for sparse representation
            int non_zero = 0;
            for (float val : features[i])
            {
                if (std::abs(val) > 1e-8)
                    non_zero++;
            }

            // Allocate nodes (+1 for terminating node with index=-1)
            prob->x[i] = new svm_node[non_zero + 1];

            int node_idx = 0;
            for (int j = 0; j < n_features; ++j)
            {
                if (std::abs(features[i][j]) > 1e-8)
                {
                    prob->x[i][node_idx].index = j + 1; // LIBSVM uses 1-indexed features
                    prob->x[i][node_idx].value = features[i][j];
                    node_idx++;
                }
            }
            prob->x[i][non_zero].index = -1; // Terminating node
        }

        // Set SVM parameters
        svm_parameter *param = new svm_parameter;
        param->svm_type = C_SVC;  // Classification
        param->kernel_type = RBF; // RBF kernel
        param->degree = 3;        // For polynomial kernel (not used)
        param->gamma = gamma;     // RBF kernel parameter
        param->coef0 = 0;         // For polynomial/sigmoid kernel
        param->nu = 0.5;          // For nu-SVC
        param->cache_size = 200;  // MB
        param->C = C;             // Cost parameter
        param->eps = 1e-3;        // Stopping criterion
        param->p = 0.1;           // For epsilon-SVR
        param->shrinking = 1;     // Use shrinking heuristic
        param->probability = 0;   // No probability estimates
        param->nr_weight = 0;     // No class weights
        param->weight_label = nullptr;
        param->weight = nullptr;

        // Check parameters
        const char *error_msg = svm_check_parameter(prob, param);
        if (error_msg)
        {
            LOG_ERROR("SVM parameter error: %s", error_msg);
            // Cleanup
            for (int i = 0; i < n_samples; ++i)
                delete[] prob->x[i];
            delete[] prob->x;
            delete[] prob->y;
            delete prob;
            delete param;
            return nullptr;
        }

        // Train SVM
        std::cout << "      Training SVM (this may take a few minutes)...\n";
        svm_model *model = svm_train(prob, param);

        // Cleanup problem data (model contains its own copy)
        for (int i = 0; i < n_samples; ++i)
            delete[] prob->x[i];
        delete[] prob->x;
        delete[] prob->y;
        delete prob;
        delete param;

        LOG_INFO("SVM training completed");
        return model;
    }

    std::vector<int> predict(
        const svm_model *model,
        const std::vector<std::vector<float>> &features)
    {
        if (!model || features.empty())
        {
            LOG_ERROR("Invalid input: model or features null/empty");
            return {};
        }

        int n_samples = features.size();
        int n_features = features[0].size();
        std::vector<int> predictions(n_samples);

        for (int i = 0; i < n_samples; ++i)
        {
            // Count non-zero features
            int non_zero = 0;
            for (float val : features[i])
            {
                if (std::abs(val) > 1e-8)
                    non_zero++;
            }

            // Convert to LIBSVM format
            svm_node *x = new svm_node[non_zero + 1];
            int node_idx = 0;
            for (int j = 0; j < n_features; ++j)
            {
                if (std::abs(features[i][j]) > 1e-8)
                {
                    x[node_idx].index = j + 1;
                    x[node_idx].value = features[i][j];
                    node_idx++;
                }
            }
            x[non_zero].index = -1;

            // Predict
            double label = svm_predict(model, x);
            predictions[i] = static_cast<int>(label);

            delete[] x;
        }

        return predictions;
    }

    float calculate_accuracy(
        const std::vector<int> &true_labels,
        const std::vector<int> &predicted_labels)
    {
        if (true_labels.size() != predicted_labels.size())
        {
            LOG_ERROR("Label size mismatch");
            return 0.0f;
        }

        int correct = 0;
        for (size_t i = 0; i < true_labels.size(); ++i)
        {
            if (true_labels[i] == predicted_labels[i])
                correct++;
        }

        return static_cast<float>(correct) / true_labels.size();
    }

    std::vector<float> calculate_per_class_accuracy(
        const std::vector<int> &true_labels,
        const std::vector<int> &predicted_labels,
        int num_classes)
    {
        std::vector<int> correct(num_classes, 0);
        std::vector<int> total(num_classes, 0);

        for (size_t i = 0; i < true_labels.size(); ++i)
        {
            int true_label = true_labels[i];
            total[true_label]++;
            if (true_labels[i] == predicted_labels[i])
                correct[true_label]++;
        }

        std::vector<float> accuracies(num_classes);
        for (int i = 0; i < num_classes; ++i)
        {
            accuracies[i] = (total[i] > 0) ? (static_cast<float>(correct[i]) / total[i]) : 0.0f;
        }

        return accuracies;
    }

    void free_model(svm_model *model)
    {
        if (model)
        {
            svm_free_and_destroy_model(&model);
        }
    }

    void save_model(const svm_model *model, const std::string &filename)
    {
        if (model)
        {
            svm_save_model(filename.c_str(), model);
            LOG_INFO("SVM model saved to: %s", filename.c_str());
        }
    }

    svm_model *load_model(const std::string &filename)
    {
        svm_model *model = svm_load_model(filename.c_str());
        if (model)
        {
            LOG_INFO("SVM model loaded from: %s", filename.c_str());
        }
        else
        {
            LOG_ERROR("Failed to load SVM model from: %s", filename.c_str());
        }
        return model;
    }

} // namespace SVMClassifier
