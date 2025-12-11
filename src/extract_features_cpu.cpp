#include "models/autoencoder_cpu.h"
#include "data/cifar10_loader.h"
#include "config.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using std::cout, std::string;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Please provide the name for the weight file as an argument when running";
        return 1;
    }

    string weights_file = argv[1];

    cout << "Feature Extraction (CPU)\n";
    cout << "Loading weights from " << weights_file << "\n";

    try
    {
        // Load model with trained weights
        AutoencoderCPU model;
        model.load_weights(weights_file);

        const int FEATURE_DIM = 8192; // 128 * 8 * 8
        const string MODEL_DIR = string(MODEL_SAVE_DIR);

        // ===== Extract Training Features =====
        cout << "\n\n=== Extracting Training Features ===\n";
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        loader.load_train_data();

        const int total_train = loader.train_size();
        const int train_batches = (total_train + BATCH_SIZE - 1) / BATCH_SIZE; // round up

        cout << "Processing " << total_train << " training images...\n";

        std::vector<std::vector<float>> train_features;
        train_features.reserve(total_train);

        auto start_time = std::chrono::high_resolution_clock::now();
        loader.reset();

        for (int b = 0; b < train_batches; ++b)
        {
            int current_batch_size = std::min(BATCH_SIZE, total_train - b * BATCH_SIZE);
            float *batch = loader.get_batch(current_batch_size);
            float *latent = model.extract_features(batch, current_batch_size);

            for (int i = 0; i < current_batch_size; ++i)
            {
                std::vector<float> feature_vec(FEATURE_DIM);
                std::copy(latent + i * FEATURE_DIM,
                          latent + (i + 1) * FEATURE_DIM,
                          feature_vec.begin());
                train_features.push_back(std::move(feature_vec));
            }

            // Print every 500 batches
            if ((b + 1) % 500 == 0)
            {
                cout << "  Processed " << train_features.size() << " / " << total_train << "\n";
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();

        cout << "Training features: " << train_features.size() << " x " << FEATURE_DIM << "\n";
        cout << "Time: " << elapsed << "s\n";

        // Save training features
        string train_output = MODEL_DIR + "/train_features_cpu.bin";
        std::ofstream train_out(train_output, std::ios::binary);
        int num_train = train_features.size();
        int feat_dim = FEATURE_DIM;
        train_out.write(reinterpret_cast<const char *>(&num_train), sizeof(int));
        train_out.write(reinterpret_cast<const char *>(&feat_dim), sizeof(int));
        for (const auto &vec : train_features)
        {
            train_out.write(reinterpret_cast<const char *>(vec.data()), vec.size() * sizeof(float));
        }
        train_out.close();
        cout << "Saved: " << train_output << "\n";

        // ===== Extract Test Features =====
        cout << "\n\n=== Extracting Test Features ===\n";
        loader.load_test_data();

        const int total_test = loader.test_size();
        const int test_batches = (total_test + BATCH_SIZE - 1) / BATCH_SIZE;

        cout << "Processing " << total_test << " test images...\n";

        std::vector<std::vector<float>> test_features;
        test_features.reserve(total_test);

        start_time = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < test_batches; ++b)
        {
            int current_batch_size = std::min(BATCH_SIZE, total_test - b * BATCH_SIZE);

            // Get test batch
            float *batch = loader.get_test_batch_at(b * BATCH_SIZE, current_batch_size);
            float *latent = model.extract_features(batch, current_batch_size);

            for (int i = 0; i < current_batch_size; ++i)
            {
                std::vector<float> feature_vec(FEATURE_DIM);
                std::copy(latent + i * FEATURE_DIM,
                          latent + (i + 1) * FEATURE_DIM,
                          feature_vec.begin());
                test_features.push_back(std::move(feature_vec));
            }

            if ((b + 1) % 100 == 0)
            {
                cout << "  Processed " << test_features.size() << " / " << total_test << "\n";
            }
        }

        end_time = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end_time - start_time).count();

        cout << "Test features: " << test_features.size() << " x " << FEATURE_DIM << "\n";
        cout << "Time: " << elapsed << "s\n";

        // Save test features
        string test_output = MODEL_DIR + "/test_features_cpu.bin";
        std::ofstream test_out(test_output, std::ios::binary);
        int num_test = test_features.size();
        test_out.write(reinterpret_cast<const char *>(&num_test), sizeof(int));
        test_out.write(reinterpret_cast<const char *>(&feat_dim), sizeof(int));
        for (const auto &vec : test_features)
        {
            test_out.write(reinterpret_cast<const char *>(vec.data()), vec.size() * sizeof(float));
        }
        test_out.close();
        cout << "Saved: " << test_output << "\n";

        cout << "\n=== Feature Extraction Complete ===\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
