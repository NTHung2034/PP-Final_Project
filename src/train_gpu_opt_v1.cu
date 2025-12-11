// CIFAR-10 Autoencoder Training - GPU Optimized v1
// Optimizations: Memory Pool, Shared Memory Tiling, Constant Memory for Biases
#include "models/autoencoder_gpu_opt_v1.cuh"
#include "data/cifar10_loader.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

// CUDA event-based timer for accurate GPU timing
class CUDATimer {
    cudaEvent_t start_event, stop_event;
public:
    CUDATimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    ~CUDATimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    void start() { cudaEventRecord(start_event, 0); }
    float stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms / 1000.0f;  // Return seconds
    }
};

void save_training_stats(const std::string& filepath, 
                         const std::vector<float>& losses,
                         const std::vector<float>& times,
                         int batch_size, float lr) {
    std::ofstream out(filepath);
    if (!out) return;
    
    float total_time = 0;
    for (float t : times) total_time += t;
    
    out << "=== GPU OPTIMIZED v1 TRAINING SUMMARY ===\n\n";
    out << "Optimizations: Memory Pool, Shared Memory Tiling, Constant Memory\n\n";
    out << "Configuration:\n";
    out << "  Batch size: " << batch_size << "\n";
    out << "  Learning rate: " << lr << "\n";
    out << "  Epochs: " << losses.size() << "\n\n";
    out << "Results:\n";
    out << "  Total time: " << std::fixed << std::setprecision(2) << total_time << "s\n";
    out << "  Avg time/epoch: " << (total_time / losses.size()) << "s\n";
    out << "  Initial loss: " << std::setprecision(6) << losses.front() << "\n";
    out << "  Final loss: " << losses.back() << "\n";
    out << "  Loss reduction: " << std::setprecision(2) << ((1.0f - losses.back() / losses.front()) * 100) << "%\n\n";
    out << "Epoch\tLoss\t\tTime(s)\t\timg/s\n";
    out << "-----\t--------\t-------\t\t-----\n";
    for (size_t i = 0; i < losses.size(); i++) {
        out << (i + 1) << "\t" << std::setprecision(6) << losses[i] << "\t"
            << std::setprecision(2) << times[i] << "\t\t"
            << (int)(50000 / times[i]) << "\n";
    }
    out.close();
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   CIFAR-10 AUTOENCODER - GPU OPTIMIZED v1 (Memory Optimization)  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // Check GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " | SM " << prop.major << "." << prop.minor 
              << " | " << (prop.totalGlobalMem >> 20) << " MB\n\n";

    // Config
    const int batch_size = BATCH_SIZE;
    const int epochs = EPOCHS;
    const float lr = LEARNING_RATE;
    const std::string save_dir = MODEL_SAVE_DIR_GPU_OPTIMIZED;
    
    std::cout << "Config: batch=" << batch_size << ", epochs=" << epochs << ", lr=" << lr << "\n";

    // Load data
    CIFAR10Loader loader(CIFAR_BIN_DIR);
    loader.load_train_data();
    int num_batches = loader.train_size() / batch_size;
    std::cout << "Data: " << loader.train_size() << " images, " << num_batches << " batches/epoch\n\n";

    // Create model (allocates all memory once)
    AutoencoderGPUOptV1 model(batch_size);
    
    // Training stats
    std::vector<float> epoch_losses;
    std::vector<float> epoch_times;
    
    // Training
    std::cout << "Epoch |   Loss   |  Time  |  img/s\n";
    std::cout << "------|----------|--------|--------\n";

    CUDATimer timer;

    for (int epoch = 0; epoch < epochs; epoch++) {
        timer.start();
        
        loader.shuffle();
        loader.reset();
        
        float epoch_loss = 0.0f;
        for (int b = 0; b < num_batches; b++) {
            float* batch = loader.get_batch(batch_size);
            epoch_loss += model.train_step(batch, batch_size, lr);
        }
        epoch_loss /= num_batches;
        
        float sec = timer.stop();
        
        epoch_losses.push_back(epoch_loss);
        epoch_times.push_back(sec);
        
        std::cout << std::setw(5) << (epoch + 1) << " | " 
                  << std::fixed << std::setprecision(6) << epoch_loss << " | "
                  << std::setprecision(1) << std::setw(5) << sec << "s | "
                  << std::setw(6) << (int)(loader.train_size() / sec) << "\n";
    }

    // Save weights and stats
    #ifdef _WIN32
        system(("if not exist \"" + save_dir + "\" mkdir \"" + save_dir + "\"").c_str());
    #else
        system(("mkdir -p " + save_dir).c_str());
    #endif
    model.save_weights(save_dir);
    save_training_stats(save_dir + "/training_summary.txt", epoch_losses, epoch_times, batch_size, lr);
    std::cout << "\nWeights and stats saved to: " << save_dir << "\n";

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    TRAINING COMPLETED!                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";


    return 0;
}
