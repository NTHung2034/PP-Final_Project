#include "data/cifar10_loader.h"
#include "config.h"
#include "../../external/svm.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>

using std::cout, std::string;

const string MODEL_DIR = string(PROJECT_ROOT_DIR) + "/models/saved_weights";
const string TRAIN_FEATURE_FILE = MODEL_DIR + "/train_features_cpu.bin";
const string TEST_FEATURE_FILE = MODEL_DIR + "/test_features_cpu.bin";
const string SVM_MODEL_FILE = MODEL_DIR + "/svm_model_cpu.bin";

// Forward declarations
svm_node *create_svm_nodes(const std::vector<float> &features);
svm_problem *create_svm_problem(const std::vector<std::vector<float>> &features, const std::vector<int> &labels);
void free_svm_problem(svm_problem *prob);
double calculate_gamma_auto(int num_features);

// Load features from binary file
bool load_features(const string &filepath, std::vector<std::vector<float>> &features, int &num_samples, int &feature_dim)
{
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Cannot open: " << filepath << "\n";
        return false;
    }

    in.read(reinterpret_cast<char *>(&num_samples), sizeof(int));
    in.read(reinterpret_cast<char *>(&feature_dim), sizeof(int));

    features.resize(num_samples);
    for (int i = 0; i < num_samples; ++i)
    {
        features[i].resize(feature_dim);
        in.read(reinterpret_cast<char *>(features[i].data()), feature_dim * sizeof(float));
    }

    in.close();
    return true;
}

// Convert to LIBSVM problem format
svm_problem *create_svm_problem(const std::vector<std::vector<float>> &features,
                                const std::vector<int> &labels)
{
    svm_problem *prob = new svm_problem;
    prob->l = features.size();
    prob->y = new double[prob->l];
    prob->x = new svm_node *[prob->l];

    for (int i = 0; i < prob->l; ++i)
    {
        prob->y[i] = labels[i];
        prob->x[i] = create_svm_nodes(features[i]);
    }

    return prob;
}

// Free LIBSVM problem
void free_svm_problem(svm_problem *prob)
{
    if (prob)
    {
        for (int i = 0; i < prob->l; ++i)
        {
            delete[] prob->x[i];
        }
        delete[] prob->x;
        delete[] prob->y;
        delete prob;
    }
}

// Calculate gamma=auto (1 / num_features)
double calculate_gamma_auto(int num_features)
{
    return 1.0 / num_features;
}

// Convert feature vector to LIBSVM node format
svm_node *create_svm_nodes(const std::vector<float> &features)
{
    int feature_dim = features.size();
    svm_node *nodes = new svm_node[feature_dim + 1];

    for (int j = 0; j < feature_dim; ++j)
    {
        nodes[j].index = j + 1;
        nodes[j].value = features[j];
    }
    nodes[feature_dim].index = -1; // Terminator

    return nodes;
}

int main()
{
    auto program_start = std::chrono::high_resolution_clock::now();

    cout << "SVM Training using LIBSVM (CPU)\n";
    cout << "================================\n\n";

    try
    {
        // Load training features
        cout << "Loading training features...\n";
        std::vector<std::vector<float>> train_features;
        int num_train, feature_dim;

        if (!load_features(TRAIN_FEATURE_FILE, train_features, num_train, feature_dim))
        {
            throw std::runtime_error("Failed to load training features. Run extract_features_cpu first.");
        }
        num_train = std::min(500, num_train);
        train_features.resize(num_train);
        cout << "  " << num_train << " samples x " << feature_dim << " features\n";

        // Load training labels
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        loader.load_train_data();
        std::vector<int> train_labels(num_train);
        for (int i = 0; i < num_train; ++i)
        {
            train_labels[i] = loader.train_labels()[i];
        }

        // Load test features
        cout << "Loading test features...\n";
        std::vector<std::vector<float>> test_features;
        int num_test, test_feat_dim;

        if (!load_features(TEST_FEATURE_FILE, test_features, num_test, test_feat_dim))
        {
            throw std::runtime_error("Failed to load test features. Run extract_features_cpu first.");
        }
        num_test = std::min(500, num_test);
        test_features.resize(num_test);
        cout << "  " << num_test << " samples x " << test_feat_dim << " features\n";

        // Load test labels
        CIFAR10Loader test_loader(CIFAR_BIN_DIR);
        test_loader.load_test_data();
        std::vector<int> test_labels(num_test);
        for (int i = 0; i < num_test; ++i)
        {
            test_labels[i] = test_loader.test_labels()[i];
        }

        // Setup SVM parameters
        svm_parameter param;
        param.svm_type = C_SVC;
        param.kernel_type = RBF;
        param.degree = 3;
        param.gamma = calculate_gamma_auto(feature_dim); // gamma = 1 / num_features
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 10.0;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = NULL;
        param.weight = NULL;

        cout << "\nSVM Parameters:\n";
        cout << "  Kernel: RBF\n";
        cout << "  C: " << param.C << "\n";
        cout << "  Gamma: " << param.gamma << " (auto)\n\n";

        // Create training problem
        cout << "Preparing training data...\n";
        svm_problem *prob = create_svm_problem(train_features, train_labels);

        // Check parameters
        const char *error_msg = svm_check_parameter(prob, &param);
        if (error_msg)
        {
            throw std::runtime_error(string("SVM parameter error: ") + error_msg);
        }

        // Train SVM
        cout << "Training SVM...\n";
        auto start = std::chrono::high_resolution_clock::now();

        svm_model *model = svm_train(prob, &param);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        cout << "Training completed in " << elapsed << "s\n\n";

        // Predict on training set
        cout << "Evaluating on training set...\n";
        int train_correct = 0;
        for (int i = 0; i < num_train; ++i)
        {
            svm_node *x = prob->x[i];
            double pred = svm_predict(model, x);
            if ((int)pred == train_labels[i])
            {
                train_correct++;
            }
        }
        double train_accuracy = 100.0 * train_correct / num_train;
        cout << "  Training accuracy: " << train_accuracy << "%\n\n";

        // Predict on test set
        cout << "Evaluating on test set...\n";
        int test_correct = 0;
        std::vector<int> predictions(num_test);

        // Confusion matrix
        int confusion[CIFAR_CLASSES][CIFAR_CLASSES] = {0};

        for (int i = 0; i < num_test; ++i)
        {
            svm_node *x = create_svm_nodes(test_features[i]);

            double pred = svm_predict(model, x);
            int pred_label = (int)pred;
            int true_label = test_labels[i];

            predictions[i] = pred_label;

            if (pred_label == true_label)
            {
                test_correct++;
            }

            confusion[true_label][pred_label]++;

            delete[] x;
        }

        double test_accuracy = 100.0 * test_correct / num_test;
        cout << "  Test accuracy: " << test_accuracy << "%\n\n";

        // Save predictions for Python analysis
        cout << "Saving predictions...\n";
        string pred_file = MODEL_DIR + "/test_predictions.txt";
        std::ofstream pred_out(pred_file);
        if (!pred_out.is_open())
        {
            cout << "Warning: Could not save predictions file\n";
        }
        else
        {
            pred_out << "Actual Predicted\n";
            for (int i = 0; i < num_test; ++i)
            {
                pred_out << test_labels[i] << " " << predictions[i] << "\n";
            }
            pred_out.close();
            cout << "Predictions saved: " << pred_file << "\n\n";
        }

        // Display confusion matrix
        const char *class_names[] = {"airplane", "automobile", "bird", "cat", "deer",
                                     "dog", "frog", "horse", "ship", "truck"};

        cout << "Confusion Matrix:\n";
        cout << "True\\Pred  ";
        for (int i = 0; i < CIFAR_CLASSES; ++i)
        {
            printf("%6s", class_names[i]);
        }
        cout << "\n";

        for (int i = 0; i < CIFAR_CLASSES; ++i)
        {
            printf("%-10s ", class_names[i]);
            for (int j = 0; j < CIFAR_CLASSES; ++j)
            {
                printf("%6d", confusion[i][j]);
            }
            cout << "\n";
        }
        cout << "\n";

        // Per-class accuracy
        cout << "Per-Class Accuracy:\n";
        for (int i = 0; i < CIFAR_CLASSES; ++i)
        {
            int total = 0;
            for (int j = 0; j < CIFAR_CLASSES; ++j)
            {
                total += confusion[i][j];
            }
            double acc = total > 0 ? 100.0 * confusion[i][i] / total : 0.0;
            printf("  %-12s: %6.2f%%\n", class_names[i], acc);
        }

        // Save results to file
        cout << "\nSaving results...\n";
        string results_file = MODEL_DIR + "/svm_results.txt";
        std::ofstream results_out(results_file);
        if (results_out.is_open())
        {
            results_out << "SVM Classification Results\n";
            results_out << "==========================\n\n";
            results_out << "Training samples: " << num_train << "\n";
            results_out << "Test samples: " << num_test << "\n";
            results_out << "Feature dimension: " << feature_dim << "\n\n";
            results_out << "SVM Parameters:\n";
            results_out << "  Kernel: RBF\n";
            results_out << "  C: " << param.C << "\n";
            results_out << "  Gamma: " << param.gamma << " (auto)\n\n";
            results_out << "Training time: " << elapsed << "s\n\n";
            results_out << "Training accuracy: " << train_accuracy << "%\n";
            results_out << "Test accuracy: " << test_accuracy << "%\n\n";
            results_out << "Confusion Matrix:\n";
            results_out << "True\\Pred  ";
            for (int i = 0; i < CIFAR_CLASSES; ++i)
            {
                results_out << class_names[i][0] << class_names[i][1] << "  ";
            }
            results_out << "\n";
            for (int i = 0; i < CIFAR_CLASSES; ++i)
            {
                results_out << class_names[i] << "  ";
                for (int j = 0; j < CIFAR_CLASSES; ++j)
                {
                    results_out << confusion[i][j] << "  ";
                }
                results_out << "\n";
            }
            results_out << "\nPer-Class Accuracy:\n";
            for (int i = 0; i < CIFAR_CLASSES; ++i)
            {
                int total = 0;
                for (int j = 0; j < CIFAR_CLASSES; ++j)
                {
                    total += confusion[i][j];
                }
                double acc = total > 0 ? 100.0 * confusion[i][i] / total : 0.0;
                results_out << "  " << class_names[i] << ": " << acc << "%\n";
            }
            results_out.close();
            cout << "Results saved to: " << results_file << "\n";
        }

        // Cleanup
        svm_free_and_destroy_model(&model);
        free_svm_problem(prob);

        // Calculate total program runtime
        auto program_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(program_end - program_start).count();

        cout << "\n=== SVM Training Complete ===\n";
        cout << "Total runtime: " << total_time << "s\n";

        // Append total runtime to results file
        std::ofstream results_append(results_file, std::ios::app);
        if (results_append.is_open())
        {
            results_append << "\nTotal program runtime: " << total_time << "s\n";
            results_append.close();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
