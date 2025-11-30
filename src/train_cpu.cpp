
#include "models/autoencoder_cpu.h"
#include "data/cifar10_dataset.h"
#include "utils/logger.h"
#include "config.h"
#include <chrono>

int main()
{
    LOG_INIT();
    LOG_INFO("Starting CPU Training");

    // Load dataset
    CIFAR10Dataset dataset(CIFAR_BIN_DIR, CIFAR10Dataset::Mode::TRAIN);
    dataset.load_data();
    LOG_INFO("Loaded %zu training images", dataset.size());

    // Create model
    AutoencoderCPU model;

    // Training configuration
    const int batch_size = BATCH_SIZE;
    const int epochs = EPOCHS;
    const float lr = LEARNING_RATE;
    const int batches_per_epoch = dataset.size() / batch_size;

    LOG_INFO("Training config: epochs=%d, batch_size=%d, lr=%.4f",
             epochs, batch_size, lr);

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        dataset.shuffle();
        dataset.reset();

        double epoch_loss = 0.0;

        for (int batch = 0; batch < batches_per_epoch; ++batch)
        {
            // Get batch
            auto images = dataset.get_batch(batch_size);

            // Forward pass
            auto output = model.forward(images);

            // Compute loss
            float loss = model.compute_loss(output, images);
            epoch_loss += loss;

            // Backward pass + update
            model.backward(images, lr);

            // Log progress
            if (batch % 100 == 0)
            {
                LOG_INFO("Epoch %d/%d, Batch %d/%d, Loss: %.6f",
                         epoch + 1, epochs, batch, batches_per_epoch, loss);
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
                              epoch_end - epoch_start)
                              .count();

        epoch_loss /= batches_per_epoch;

        LOG_INFO("Epoch %d/%d completed in %ld seconds, Avg Loss: %.6f",
                 epoch + 1, epochs, epoch_time, epoch_loss);
    }

    // Save trained model
    model.save_weights(MODEL_SAVE_DIR "/cpu_encoder_weights.bin");
    LOG_INFO("Training complete! Model saved.");

    return 0;
}