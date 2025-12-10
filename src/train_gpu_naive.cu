#include "models/autoencoder_gpu_naive.cuh"
#include "data/cifar10_loader.h"
#include "config.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>

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
        
        out << "=== NAIVE GPU AUTOENCODER TRAINING SUMMARY ===\n\n";
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

// Save model weights
void save_model_weights(GPUAutoencoderNaive& model, const std::string& output_dir) {
    std::cout << "Saving model weights to: " << output_dir << std::endl;
    
    // Create directory if it doesn't exist (platform-specific)
    #ifdef _WIN32
        system(("if not exist \"" + output_dir + "\" mkdir \"" + output_dir + "\"").c_str());
    #else
        system(("mkdir -p " + output_dir).c_str());
    #endif
    
    // Save each layer's weights
    auto save_conv_weights = [&](GPUConvWeights* weights, const std::string& name) {
        std::string weight_file = output_dir + "/" + name + "_weights.bin";
        std::string bias_file = output_dir + "/" + name + "_bias.bin";
        
        // Copy weights from device to host
        std::vector<float> h_weights(weights->weight_size);
        std::vector<float> h_bias(weights->bias_size);
        
        CUDA_CHECK(cudaMemcpy(h_weights.data(), weights->d_weights, 
                             weights->weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_bias.data(), weights->d_bias, 
                             weights->bias_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Save to binary files
        std::ofstream wf(weight_file, std::ios::binary);
        wf.write(reinterpret_cast<const char*>(h_weights.data()), h_weights.size() * sizeof(float));
        wf.close();
        
        std::ofstream bf(bias_file, std::ios::binary);
        bf.write(reinterpret_cast<const char*>(h_bias.data()), h_bias.size() * sizeof(float));
        bf.close();
        
        std::cout << "  Saved " << name << " weights: " << weights->weight_size 
                  << " params, bias: " << weights->bias_size << " params" << std::endl;
    };
    
    save_conv_weights(model.conv1, "conv1");
    save_conv_weights(model.conv2, "conv2");
    save_conv_weights(model.conv3, "conv3");
    save_conv_weights(model.conv4, "conv4");
    save_conv_weights(model.conv5, "conv5");
    
    std::cout << "Model weights saved successfully!" << std::endl;
}

// Training function
void train_autoencoder_gpu_naive(
    CIFAR10Loader& loader,
    int batch_size,
    int epochs,
    float learning_rate,
    const std::string& save_dir)
{
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "NAIVE GPU AUTOENCODER TRAINING" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Training samples: " << loader.train_size() << std::endl;
    std::cout << "  Save directory: " << save_dir << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;
    
    // Initialize model
    GPUAutoencoderNaive model;
    model.allocatePoolingIndices(batch_size);
    
    // Preallocate activation buffers for training
    std::vector<GPUTensor*> activations;
    activations.push_back(new GPUTensor(batch_size, 256, 32, 32, true)); // Conv1
    activations.push_back(new GPUTensor(batch_size, 256, 16, 16, true)); // Pool1
    activations.push_back(new GPUTensor(batch_size, 128, 16, 16, true)); // Conv2
    activations.push_back(new GPUTensor(batch_size, 128, 8, 8, true));   // Pool2 (latent)
    activations.push_back(new GPUTensor(batch_size, 128, 8, 8, true));   // Conv3
    activations.push_back(new GPUTensor(batch_size, 128, 16, 16, true)); // Upsample1
    activations.push_back(new GPUTensor(batch_size, 256, 16, 16, true)); // Conv4
    activations.push_back(new GPUTensor(batch_size, 256, 32, 32, true)); // Upsample2
    activations.push_back(new GPUTensor(batch_size, 3, 32, 32, true));   // Output
    
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
        
        // Batch loop
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            // Get batch data pointer from loader
            float* batch_data = loader.get_batch(batch_size);
            if (!batch_data) break;
            
            // Create GPU tensor and copy data
            GPUTensor input(batch_size, 3, 32, 32, false);
            memcpy(input.h_data, batch_data, batch_size * 3 * 32 * 32 * sizeof(float));
            input.copyToDevice();
            
            // Forward-Backward-Update in one call
            float batch_loss = model.forward_backward_update(
                input, input, learning_rate, activations
            );
            
            epoch_loss += batch_loss;
            batches_processed++;
            
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
    save_model_weights(model, save_dir);
    stats.save_to_file(save_dir + "/training_summary.txt");
    
    // Cleanup
    for (auto* tensor : activations) {
        delete tensor;
    }
}

int main(int argc, char** argv) {
    try {
        // Configuration
        const std::string data_dir = "data";
        const std::string save_dir = "models/saved_weights_gpu_naive";
        const int batch_size = BATCH_SIZE;
        const int epochs = EPOCHS;
        const float learning_rate = LEARNING_RATE;
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║   CIFAR-10 AUTOENCODER - NAIVE GPU IMPLEMENTATION (PHASE 2)   ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
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
        
        // Load dataset using simple loader
        CIFAR10Loader loader(data_dir);
        loader.load_train_data();
        
        // Train model
        train_autoencoder_gpu_naive(
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
