// CIFAR-10 Autoencoder Training - GPU Optimized v2
// Optimizations: Kernel Fusion (Conv+ReLU+Bias), Multi-Stream Pipeline
#include "models/autoencoder_gpu_opt_v2.cuh"
#include "data/cifar10_loader.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

// Simple VRAM usage query
size_t get_vram_used_mb() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return (total - free) / (1024 * 1024);
}

// CUDA event-based timer for accurate GPU timing
class CUDATimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    CUDATimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~CUDATimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event, 0));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds / 1000.0f; // Convert to seconds
    }
};

// Training statistics tracker
class TrainingStatsV2 {
public:
    std::vector<float> epoch_losses;
    std::vector<float> epoch_times;
    float total_training_time = 0.0f;
    
    void add_epoch(float loss, float time) {
        epoch_losses.push_back(loss);
        epoch_times.push_back(time);
        total_training_time += time;
    }
    
    void print_epoch_summary(int epoch, float loss, float time, int total_epochs) {
        std::cout << "\r";
        std::cout << "Epoch [" << std::setw(3) << epoch << "/" << total_epochs << "] "
                  << "| Loss: " << std::fixed << std::setprecision(6) << std::setw(10) << loss
                  << " | Time: " << std::setprecision(2) << std::setw(6) << time << "s"
                  << " | Avg: " << std::setw(6) << (total_training_time / epoch_losses.size()) << "s/epoch"
                  << std::endl;
    }
    
    void print_final_summary() {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "TRAINING COMPLETED" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total training time: " << total_training_time << "s" << std::endl;
        std::cout << "Average time per epoch: " << (total_training_time / epoch_losses.size()) << "s" << std::endl;
        std::cout << std::setprecision(6);
        std::cout << "Initial loss: " << epoch_losses.front() << std::endl;
        std::cout << "Final loss: " << epoch_losses.back() << std::endl;
        std::cout << "Loss reduction: " << ((1.0f - epoch_losses.back() / epoch_losses.front()) * 100.0f) << "%" << std::endl;
        std::cout << std::string(70, '=') << "\n" << std::endl;
    }
    
    void save_to_file(const std::string& filepath) {
        std::ofstream out(filepath);
        if (!out) {
            std::cerr << "Warning: Could not save training summary to " << filepath << std::endl;
            return;
        }
        
        out << "=== GPU OPTIMIZED v2 AUTOENCODER TRAINING SUMMARY ===\n\n";
        out << "Optimizations: Kernel Fusion, Multi-Stream Pipeline\n\n";
        out << "Total Training Time: " << total_training_time << " seconds\n";
        out << "Average Time Per Epoch: " << (total_training_time / epoch_losses.size()) << " seconds\n";
        out << "Number of Epochs: " << epoch_losses.size() << "\n\n";
        
        out << "Epoch\tLoss\t\tTime(s)\n";
        out << "-----\t--------\t-------\n";
        for (size_t i = 0; i < epoch_losses.size(); i++) {
            out << (i + 1) << "\t" 
                << std::fixed << std::setprecision(6) << epoch_losses[i] << "\t"
                << std::setprecision(2) << epoch_times[i] << "\n";
        }
        
        out.close();
        std::cout << "Training summary saved to: " << filepath << std::endl;
    }
};

// Training function
void train_autoencoder_gpu_opt_v2(
    CIFAR10Loader& loader,
    int batch_size,
    int epochs,
    float learning_rate,
    const std::string& save_dir)
{
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "GPU OPTIMIZED v2 AUTOENCODER TRAINING" << std::endl;
    std::cout << "(Kernel Fusion + Multi-Stream Pipeline)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Training samples: " << loader.train_size() << std::endl;
    std::cout << "  Save directory: " << save_dir << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;
    
    // Create model (allocates all memory once via Memory Pool)
    AutoencoderGPUOptV2 model(batch_size);
    
    std::cout << "VRAM usage: " << get_vram_used_mb() << " MB\n\n";
    
    // Training statistics
    TrainingStatsV2 stats;
    CUDATimer gpu_timer;
    
    int num_batches = loader.train_size() / batch_size;
    std::cout << "Batches per epoch: " << num_batches << "\n" << std::endl;
    
    // Main training loop   
    for (int epoch = 0; epoch < epochs; epoch++) {
        gpu_timer.start();
        
        loader.shuffle();
        
        // Reset loss accumulator at epoch start
        model.reset_epoch_loss();
        int batches_processed = 0;
        
        // Batch loop with multi-stream pipeline
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            // Get batch data pointer from loader
            float* batch_data = loader.get_batch(batch_size);
            if (!batch_data) break;
            
            // Async load + compute pipeline (loss accumulated on GPU)
            model.async_load_input(batch_data, batch_size);
            model.forward_stream();
            model.backward_stream(learning_rate);
            
            batches_processed++;
            
            // Display progress (less frequently since we don't have per-batch loss)
            if ((batch_idx + 1) % 50 == 0 || batch_idx == num_batches - 1) {
                std::cout << "\r  Epoch [" << std::setw(3) << (epoch + 1) << "/" << epochs << "] "
                          << "Batch [" << std::setw(4) << (batch_idx + 1) << "/" << num_batches << "] "
                          << "Processing..."
                          << std::flush;
            }
        }
        
        // Sync and get accumulated loss at end of epoch
        CUDA_CHECK(cudaStreamSynchronize(model.get_compute_stream()));
        
        float epoch_time = gpu_timer.stop();
        float avg_epoch_loss = model.get_epoch_loss(batches_processed);
        
        stats.add_epoch(avg_epoch_loss, epoch_time);
        stats.print_epoch_summary(epoch + 1, avg_epoch_loss, epoch_time, epochs);
    }
    
    std::cout << std::endl;
    stats.print_final_summary();
    
    // Save weights and training summary
    #ifdef _WIN32
        system(("if not exist \"" + save_dir + "\" mkdir \"" + save_dir + "\"").c_str());
    #else
        system(("mkdir -p " + save_dir).c_str());
    #endif
    
    model.save_weights(save_dir);
    stats.save_to_file(save_dir + "/training_summary.txt");
    std::cout << "Final VRAM usage: " << get_vram_used_mb() << " MB\n";
}

int main(int argc, char** argv) {
    try {
        // Configuration - use constants from config.h
        const std::string save_dir = MODEL_SAVE_DIR_GPU_OPT_V2;
        const int batch_size = BATCH_SIZE;
        const int epochs = EPOCHS;
        const float learning_rate = LEARNING_RATE;
        
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║   CIFAR-10 AUTOENCODER - GPU OPTIMIZED v2 (Fusion + Streams)     ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
        // Check CUDA device
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            std::cerr << "Error: No CUDA devices found!" << std::endl;
            return 1;
        }
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total Global Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB\n" << std::endl;
        std::cout << "VRAM usage: " << get_vram_used_mb() << " MB\n\n";
        
        // Load dataset using CIFAR_BIN_DIR from config.h
        CIFAR10Loader loader(CIFAR_BIN_DIR);
        loader.load_train_data();
        
        // Train model
        train_autoencoder_gpu_opt_v2(
            loader,
            batch_size,
            epochs,
            learning_rate,
            save_dir
        );
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    TRAINING COMPLETED!                         ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
