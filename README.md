# Folder structure

cuda_autoencoder/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── data/
│   │   ├── cifar10_dataset.h      # Main dataset interface
│   │   ├── data_types.h           # Tensor and type definitions
│   │   └── data_utils.h           # Preprocessing utilities
│   ├── utils/
│   │   ├── memory_pool.h          # Efficient memory management
│   │   └── logger.h               # Logging system
│   └── config.h                   # Global configurations
├── src/
│   ├── data/
│   │   ├── cifar10_dataset.cpp    # Implementation
│   │   └── data_utils.cpp         # Normalization, etc.
│   ├── utils/
│   │   └── memory_pool.cpp
│   └── main_train.cpp             # Training entry point
├── external/
│   └── libsvm/                    # For Phase 4 (place holder)
├── data/
│   └── cifar-10-batches-bin/      # Dataset (auto-downloaded)
│       ├── data_batch_1.bin
│       ├── data_batch_2.bin
│       ├── data_batch_3.bin
│       ├── data_batch_4.bin
│       ├── data_batch_5.bin
│       └── test_batch.bin
├── models/
│   └── saved_weights/             # Trained model storage
└── scripts/
    └── download_cifar10.sh        # Data download script
