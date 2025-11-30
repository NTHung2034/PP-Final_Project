cmake_minimum_required(VERSION 3.18)
project(CUDAAutoencoder CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(ENABLE_CUDA "Enable CUDA support (requires NVIDIA GPU)" OFF)

# Find packages
find_package(OpenMP REQUIRED)

# Try to find CUDA, but make it optional
if(ENABLE_CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found - GPU support enabled")
        enable_language(CUDA)
        set(CUDA_AVAILABLE TRUE)
    else()
        message(WARNING "CUDA requested but not found - building CPU-only version")
        set(CUDA_AVAILABLE FALSE)
    endif()
else()
    message(STATUS "Building CPU-only version (set -DENABLE_CUDA=ON for GPU support)")
    set(CUDA_AVAILABLE FALSE)
endif()

# Include directories
include_directories(include)

# Source files
set(DATA_SOURCES
    src/data/cifar10_dataset.cpp
    src/data/data_utils.cpp
)

set(UTILS_SOURCES
    src/utils/logger.cpp
    src/utils/memory_pool.cpp
)

set(LAYER_SOURCES
    src/layers/conv2d_cpu.cpp
    src/layers/relu_cpu.cpp
    src/layers/maxpool_cpu.cpp
    src/layers/upsample_cpu.cpp
)

set(MAIN_SOURCES
    src/main_train.cpp
)

# Create executable
add_executable(train_autoencoder 
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
    ${LAYER_SOURCES}
    ${MAIN_SOURCES}
)

# Link libraries - OpenMP is always required
target_link_libraries(train_autoencoder 
    OpenMP::OpenMP_CXX
)

# Add CUDA support if available
if(CUDA_AVAILABLE)
    target_link_libraries(train_autoencoder CUDA::cudart)
    target_compile_definitions(train_autoencoder PRIVATE USE_CUDA)
endif()

# Compiler flags (MSVC and GCC/Clang compatible)
if(MSVC)
    target_compile_options(train_autoencoder PRIVATE
        /W4 /O2 /openmp
    )
else()
    target_compile_options(train_autoencoder PRIVATE
        -Wall -Wextra -O3 -march=native -fopenmp
    )
endif()

# Set output directories
set_target_properties(train_autoencoder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create data directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights)