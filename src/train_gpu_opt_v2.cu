// CIFAR-10 Autoencoder Training - GPU Optimized v2
// Optimizations: CUDA Streams, Kernel Fusion, Double Buffering Pipeline
#include "models/autoencoder_gpu_opt_v2.cuh"
#include "data/cifar10_loader.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>

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
class TrainingStats {
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
        out << "Optimizations: CUDA Streams, Kernel Fusion, Double Buffering Pipeline\n\n";
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
    std::cout << "(CUDA Streams + Kernel Fusion + Double Buffering Pipeline)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Training samples: " << loader.train_size() << std::endl;
    std::cout << "  Save directory: " << save_dir << std::endl;
    std::cout << "  Double buffering: ENABLED (overlap H2D transfer with compute)" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;
    
    // Create model (allocates all memory + streams)
    AutoencoderGPUOptV2 model(batch_size);
    
    // Training statistics
    TrainingStats stats;
    CUDATimer gpu_timer;
    
    int num_batches = loader.train_size() / batch_size;
    std::cout << "Batches per epoch: " << num_batches << "\n" << std::endl;
    
    // Main training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        gpu_timer.start();
        
        loader.shuffle();
        loader.reset();
        
        float epoch_loss = 0.0f;
        int batches_processed = 0;
        
        // === PIPELINED BATCH LOOP WITH DOUBLE BUFFERING ===
        // Load first batch synchronously to prime the pipeline
        float* first_batch = loader.get_batch(batch_size);
        if (!first_batch) continue;
        
        // Copy first batch to buffer 0 (synchronous)
        model.copy_input_async(first_batch, batch_size);
        model.swap_buffers();  // Now buffer 0 has first batch, current_buffer = 0
        
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            // Start async copy of NEXT batch while training CURRENT batch
            float* next_batch = nullptr;
            if (batch_idx < num_batches - 1) {
                next_batch = loader.get_batch(batch_size);
                if (next_batch) {
                    model.copy_input_async(next_batch, batch_size);  // Async copy to alternate buffer
                }
            }
            
            // Train current batch (overlaps with next batch transfer)
            float batch_loss = model.train_step(learning_rate);
            
            epoch_loss += batch_loss;
            batches_processed++;
            
            // Swap buffers for next iteration (waits for async copy to complete)
            if (next_batch) {
                model.swap_buffers();
            }
            
            // Display progress
            if ((batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1) {
                float avg_loss = epoch_loss / batches_processed;
                std::cout << "\r  Epoch [" << std::setw(3) << (epoch + 1) << "/" << epochs << "] "
                          << "Batch [" << std::setw(4) << (batch_idx + 1) << "/" << num_batches << "] "
                          << "Loss: " << std::fixed << std::setprecision(6) << avg_loss
                          << std::flush;
            }
        }
        
        float epoch_time = gpu_timer.stop();
        float avg_epoch_loss = epoch_loss / batches_processed;
        
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
}

int main(int argc, char** argv) {
    try {
        // Configuration - use constants from config.h
        const std::string save_dir = MODEL_SAVE_DIR_GPU_OPTIMIZED;
        const int batch_size = BATCH_SIZE;
        const int epochs = EPOCHS;
        const float learning_rate = LEARNING_RATE;
        
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║   CIFAR-10 AUTOENCODER - GPU OPTIMIZED v2 (Streams + Fusion)     ║\n";
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
        std::cout << "Total Global Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
        std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "Max Streams: " << prop.asyncEngineCount << "\n" << std::endl;
        
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
