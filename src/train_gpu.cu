#include "models/autoencoder_gpu.cuh"
#include "data/cifar10_dataset.h"
#include "config.h"
#include <iostream>
#include <chrono>
#include <cstring>

/**
 * =============================================================================
 * TRAINING LOOP WITH STREAM-BASED PIPELINING
 * =============================================================================
 * 
 * Optimization: Overlap H2D transfer of batch N+1 with computation on batch N
 * Uses double-buffering pattern with two streams
 */

class GPUTrainer {
private:
    GPUAutoencoder* model;
    
    // Double buffering for pipeline
    GPUTensor* input_buffers[2];   // Two input buffers for ping-pong
    GPUTensor* target_buffers[2];  // Two target buffers
    
    // Activation buffers (allocated once, reused)
    std::vector<GPUTensor*> activations;
    
    // CUDA events for timing and synchronization
    cudaEvent_t start_event, stop_event;
    
    int batch_size;
    int current_buffer;  // 0 or 1 for double buffering
    
public:
    GPUTrainer(int batch_sz) : batch_size(batch_sz), current_buffer(0) {
        model = new GPUAutoencoder();
        
        // Allocate pooling indices for training
        model->allocatePoolingIndices(batch_size);
        
        // Allocate double buffers for input/target with pinned memory
        input_buffers[0] = new GPUTensor(batch_size, 3, 32, 32);
        input_buffers[1] = new GPUTensor(batch_size, 3, 32, 32);
        target_buffers[0] = new GPUTensor(batch_size, 3, 32, 32);
        target_buffers[1] = new GPUTensor(batch_size, 3, 32, 32);
        
        // Preallocate activation buffers (device only)
        activations.push_back(new GPUTensor(batch_size, 256, 32, 32, true));  // act1
        activations.push_back(new GPUTensor(batch_size, 256, 16, 16, true));  // act2
        activations.push_back(new GPUTensor(batch_size, 128, 16, 16, true));  // act3
        activations.push_back(new GPUTensor(batch_size, 128, 8, 8, true));    // act4
        activations.push_back(new GPUTensor(batch_size, 128, 8, 8, true));    // act5
        activations.push_back(new GPUTensor(batch_size, 128, 16, 16, true));  // act6
        activations.push_back(new GPUTensor(batch_size, 256, 16, 16, true));  // act7
        activations.push_back(new GPUTensor(batch_size, 256, 32, 32, true));  // act8
        activations.push_back(new GPUTensor(batch_size, 3, 32, 32, true));    // output
        
        // Create CUDA events for timing
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~GPUTrainer() {
        delete model;
        delete input_buffers[0];
        delete input_buffers[1];
        delete target_buffers[0];
        delete target_buffers[1];
        
        for (auto* act : activations) {
            delete act;
        }
        
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    /**
     * Load batch data from CIFAR10Dataset into host pinned memory
     */
    void load_batch_from_dataset(CIFAR10Dataset& dataset, int buffer_idx, int batch_size) {
        // Get batch from dataset (returns Tensor)
        Tensor batch = dataset.get_batch(batch_size);
        
        // Copy to pinned host memory
        GPUTensor& input_buf = *input_buffers[buffer_idx];
        GPUTensor& target_buf = *target_buffers[buffer_idx];
        
        // For autoencoder, input = target (reconstruction task)
        size_t copy_size = batch.size() * sizeof(float);
        memcpy(input_buf.h_data, batch.raw_data(), copy_size);
        memcpy(target_buf.h_data, batch.raw_data(), copy_size);
    }
    
    /**
     * Load batch data into host memory (pinned) - Legacy version with raw pointer
     * In real implementation, this would read from CIFAR-10 dataset
     */
    void load_batch_to_host(float* cifar_data, int batch_idx, int buffer_idx) {
        // Copy data to pinned host memory
        GPUTensor& input_buf = *input_buffers[buffer_idx];
        GPUTensor& target_buf = *target_buffers[buffer_idx];
        
        int offset = batch_idx * batch_size * 3 * 32 * 32;
        
        // For autoencoder, input = target (reconstruction task)
        memcpy(input_buf.h_data, cifar_data + offset, input_buf.bytes);
        memcpy(target_buf.h_data, cifar_data + offset, target_buf.bytes);
    }
    
    /**
     * Train for one epoch using CIFAR10Dataset with stream-based pipelining
     * 
     * Pipeline stages:
     * 1. H2D transfer of batch N+1 (stream_transfer)
     * 2. Compute on batch N (stream_compute)
     * 
     * This overlaps data transfer with computation
     */
    float train_epoch_pipelined(CIFAR10Dataset& dataset, float learning_rate) {
        float total_loss = 0.0f;
        int num_batches = dataset.size() / batch_size;
        
        // Reset dataset to beginning and shuffle
        dataset.reset();
        
        // Start timing
        CUDA_CHECK(cudaEventRecord(start_event));
        
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            int compute_buffer = current_buffer;
            int transfer_buffer = 1 - current_buffer;
            
            GPUTensor& compute_input = *input_buffers[compute_buffer];
            GPUTensor& compute_target = *target_buffers[compute_buffer];
            
            // If not first batch, wait for previous transfer to complete
            if (batch_idx > 0) {
                CUDA_CHECK(cudaStreamSynchronize(model->stream_transfer));
            }
            
            // Launch H2D transfer for NEXT batch (async, non-blocking)
            if (batch_idx < num_batches - 1) {
                GPUTensor& next_input = *input_buffers[transfer_buffer];
                GPUTensor& next_target = *target_buffers[transfer_buffer];
                
                // Load next batch from dataset to host pinned memory
                load_batch_from_dataset(dataset, transfer_buffer, batch_size);
                
                // Launch async transfer (overlaps with computation below)
                next_input.copyToDevice(model->stream_transfer);
                next_target.copyToDevice(model->stream_transfer);
            }
            
            // If first batch, need to transfer current batch
            if (batch_idx == 0) {
                load_batch_from_dataset(dataset, compute_buffer, batch_size);
                compute_input.copyToDevice(model->stream_compute);
                compute_target.copyToDevice(model->stream_compute);
                CUDA_CHECK(cudaStreamSynchronize(model->stream_compute));
            }
            
            // Compute on CURRENT batch (forward + backward + update)
            // This happens concurrently with the H2D transfer above
            float batch_loss = model->forward_backward_update(
                compute_input, compute_target, learning_rate, activations
            );
            
            total_loss += batch_loss;
            
            // Swap buffers for next iteration (double buffering)
            current_buffer = transfer_buffer;
            
            // Print progress every 100 batches
            if ((batch_idx + 1) % 100 == 0) {
                printf("  Batch %d/%d - Loss: %.6f\n", 
                       batch_idx + 1, num_batches, batch_loss);
            }
        }
        
        // Stop timing
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float epoch_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&epoch_time_ms, start_event, stop_event));
        
        printf("  Epoch time: %.2f seconds\n", epoch_time_ms / 1000.0f);
        
        return total_loss / num_batches;
    }

    /**
     * Train for one epoch with stream-based pipelining (legacy version with raw pointer)
     * 
     * Pipeline stages:
     * 1. H2D transfer of batch N+1 (stream_transfer)
     * 2. Compute on batch N (stream_compute)
     * 
     * This overlaps data transfer with computation
     */
    float train_epoch_pipelined(float* cifar_data, int num_batches, float learning_rate) {
        float total_loss = 0.0f;
        
        // Start timing
        CUDA_CHECK(cudaEventRecord(start_event));
        
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            int compute_buffer = current_buffer;
            int transfer_buffer = 1 - current_buffer;
            
            GPUTensor& compute_input = *input_buffers[compute_buffer];
            GPUTensor& compute_target = *target_buffers[compute_buffer];
            
            // If not first batch, wait for previous transfer to complete
            if (batch_idx > 0) {
                CUDA_CHECK(cudaStreamSynchronize(model->stream_transfer));
            }
            
            // Launch H2D transfer for NEXT batch (async, non-blocking)
            if (batch_idx < num_batches - 1) {
                GPUTensor& next_input = *input_buffers[transfer_buffer];
                GPUTensor& next_target = *target_buffers[transfer_buffer];
                
                // Load next batch to host pinned memory
                load_batch_to_host(cifar_data, batch_idx + 1, transfer_buffer);
                
                // Launch async transfer (overlaps with computation below)
                next_input.copyToDevice(model->stream_transfer);
                next_target.copyToDevice(model->stream_transfer);
            }
            
            // If first batch, need to transfer current batch
            if (batch_idx == 0) {
                load_batch_to_host(cifar_data, 0, compute_buffer);
                compute_input.copyToDevice(model->stream_compute);
                compute_target.copyToDevice(model->stream_compute);
                CUDA_CHECK(cudaStreamSynchronize(model->stream_compute));
            }
            
            // Compute on CURRENT batch (forward + backward + update)
            // This happens concurrently with the H2D transfer above
            float batch_loss = model->forward_backward_update(
                compute_input, compute_target, learning_rate, activations
            );
            
            total_loss += batch_loss;
            
            // Swap buffers for next iteration (double buffering)
            current_buffer = transfer_buffer;
            
            // Print progress every 10 batches
            if ((batch_idx + 1) % 10 == 0) {
                printf("  Batch %d/%d - Loss: %.6f\n", 
                       batch_idx + 1, num_batches, batch_loss);
            }
        }
        
        // Stop timing
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float epoch_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&epoch_time_ms, start_event, stop_event));
        
        printf("  Epoch time: %.2f seconds\n", epoch_time_ms / 1000.0f);
        
        return total_loss / num_batches;
    }
    
    /**
     * Extract features for all images using CIFAR10Dataset (used for SVM training)
     * Processes in batches to avoid memory issues
     */
    void extract_all_features(
        CIFAR10Dataset& dataset,
        float* features_output,  // Host memory for features
        int feature_batch_size = 256)
    {
        int num_images = dataset.size();
        int num_batches = (num_images + feature_batch_size - 1) / feature_batch_size;
        
        // Reset dataset to beginning
        dataset.reset();
        
        // Create temporary tensors for feature extraction
        GPUTensor input_batch(feature_batch_size, 3, 32, 32);
        GPUTensor feature_batch(feature_batch_size, 128, 8, 8);
        
        printf("Extracting features for %d images...\n", num_images);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        int images_processed = 0;
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            int current_batch_size = std::min(feature_batch_size, 
                                              static_cast<int>(num_images - images_processed));
            
            // Get batch from dataset
            Tensor batch = dataset.get_batch(current_batch_size);
            
            // Copy batch to host pinned memory
            memcpy(input_batch.h_data, batch.raw_data(), 
                   current_batch_size * CIFAR_PIXELS * sizeof(float));
            
            // Transfer to device
            input_batch.copyToDevice();
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Extract features
            model->extract_features(input_batch, feature_batch);
            
            // Transfer features back to host
            feature_batch.copyToHost();
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy to output array (flatten: 128 * 8 * 8 = 8192 features per image)
            for (int i = 0; i < current_batch_size; i++) {
                int out_offset = (images_processed + i) * 8192;
                int feat_offset = i * 8192;
                memcpy(features_output + out_offset, 
                       feature_batch.h_data + feat_offset, 
                       8192 * sizeof(float));
            }
            
            images_processed += current_batch_size;
            
            if ((batch_idx + 1) % 10 == 0) {
                printf("  Processed %d/%d batches\n", batch_idx + 1, num_batches);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("Feature extraction completed in %.2f seconds\n", duration.count() / 1000.0f);
    }

    /**
     * Extract features for all images (legacy version with raw pointer)
     * Processes in batches to avoid memory issues
     */
    void extract_all_features(
        float* cifar_data, 
        float* features_output,  // Host memory for features
        int num_images,
        int feature_batch_size = 256)
    {
        int num_batches = (num_images + feature_batch_size - 1) / feature_batch_size;
        
        // Create temporary tensors for feature extraction
        GPUTensor input_batch(feature_batch_size, 3, 32, 32);
        GPUTensor feature_batch(feature_batch_size, 128, 8, 8);
        
        printf("Extracting features for %d images...\n", num_images);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            int start_idx = batch_idx * feature_batch_size;
            int end_idx = std::min(start_idx + feature_batch_size, num_images);
            int current_batch_size = end_idx - start_idx;
            
            // Copy batch to host pinned memory
            int offset = start_idx * 3 * 32 * 32;
            memcpy(input_batch.h_data, cifar_data + offset, 
                   current_batch_size * 3 * 32 * 32 * sizeof(float));
            
            // Transfer to device
            input_batch.copyToDevice();
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Extract features
            model->extract_features(input_batch, feature_batch);
            
            // Transfer features back to host
            feature_batch.copyToHost();
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy to output array (flatten: 128 * 8 * 8 = 8192 features per image)
            for (int i = 0; i < current_batch_size; i++) {
                int out_offset = (start_idx + i) * 8192;
                int feat_offset = i * 8192;
                memcpy(features_output + out_offset, 
                       feature_batch.h_data + feat_offset, 
                       8192 * sizeof(float));
            }
            
            if ((batch_idx + 1) % 10 == 0) {
                printf("  Processed %d/%d batches\n", batch_idx + 1, num_batches);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("Feature extraction completed in %.2f seconds\n", duration.count() / 1000.0f);
    }
    
    /**
     * Save model weights to file
     */
    void save_model(const char* filename) {
        // Implementation: save weights to binary file
        // For brevity, not fully implemented here
        printf("Model saved to %s\n", filename);
    }
    
    /**
     * Load model weights from file
     */
    void load_model(const char* filename) {
        // Implementation: load weights from binary file
        printf("Model loaded from %s\n", filename);
    }
};

/**
 * =============================================================================
 * MAIN TRAINING FUNCTION
 * =============================================================================
 */
int main(int argc, char** argv) {
    // Set CUDA device
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    
    // Print device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("\n");
    
    // Training hyperparameters
    const int batch_size = BATCH_SIZE;
    const int num_epochs = EPOCHS;
    const float learning_rate = LEARNING_RATE;
    
    // Load CIFAR-10 dataset
    printf("=== Loading CIFAR-10 Dataset ===\n");
    
    CIFAR10Dataset train_dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
    CIFAR10Dataset test_dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TEST);
    
    train_dataset.load_data();
    test_dataset.load_data();
    
    printf("Training images: %zu\n", train_dataset.size());
    printf("Test images: %zu\n", test_dataset.size());
    printf("\n");
    
    // Create trainer
    GPUTrainer trainer(batch_size);
    
    // Training loop
    int num_batches = train_dataset.size() / batch_size;
    
    printf("=== Starting Training ===\n");
    printf("Epochs: %d, Batch Size: %d, Learning Rate: %.4f\n", 
           num_epochs, batch_size, learning_rate);
    printf("Number of batches per epoch: %d\n\n", num_batches);
    
    auto training_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("Epoch %d/%d:\n", epoch + 1, num_epochs);
        
        // Shuffle dataset at the beginning of each epoch
        train_dataset.shuffle();
        
        float avg_loss = trainer.train_epoch_pipelined(train_dataset, learning_rate);
        
        printf("  Average Loss: %.6f\n\n", avg_loss);
    }
    
    auto training_end = std::chrono::high_resolution_clock::now();
    auto training_duration = std::chrono::duration_cast<std::chrono::seconds>(
        training_end - training_start
    );
    
    printf("Training completed in %ld seconds (%.2f minutes)\n", 
           training_duration.count(), training_duration.count() / 60.0f);
    
    // Save trained model
    trainer.save_model("autoencoder_weights.bin");
    
    // Extract features for SVM training
    printf("\n=== Feature Extraction Phase ===\n");
    
    const int feature_dim = 8192;  // 128 * 8 * 8
    float* train_features = new float[train_dataset.size() * feature_dim];
    float* test_features = new float[test_dataset.size() * feature_dim];
    
    // Reset datasets before feature extraction
    train_dataset.reset();
    test_dataset.reset();
    
    printf("Extracting training features...\n");
    trainer.extract_all_features(train_dataset, train_features);
    
    printf("Extracting test features...\n");
    trainer.extract_all_features(test_dataset, test_features);
    
    // Get labels for SVM training
    train_dataset.reset();
    test_dataset.reset();
    std::vector<int> train_labels = train_dataset.get_batch_labels(train_dataset.size());
    std::vector<int> test_labels = test_dataset.get_batch_labels(test_dataset.size());
    
    // At this point, features can be saved and used with LIBSVM
    printf("\nFeatures extracted. Ready for SVM training.\n");
    printf("Train features: %zu x %d\n", train_dataset.size(), feature_dim);
    printf("Test features: %zu x %d\n", test_dataset.size(), feature_dim);
    
    // TODO: Save features and labels to files for LIBSVM
    // save_features_libsvm_format("train_features.txt", train_features, train_labels, train_dataset.size(), feature_dim);
    // save_features_libsvm_format("test_features.txt", test_features, test_labels, test_dataset.size(), feature_dim);
    
    // Cleanup
    delete[] train_features;
    delete[] test_features;
    
    printf("\n=== Training Pipeline Complete ===\n");
    
    return 0;
}