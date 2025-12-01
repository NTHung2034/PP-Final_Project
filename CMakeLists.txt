# =============================================================================
# CUDA Autoencoder for CIFAR-10 Feature Learning
# =============================================================================
cmake_minimum_required(VERSION 3.18)
project(CUDAAutoencoder CXX C)

# =============================================================================
# C++ Standard Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Build Options
# =============================================================================
option(ENABLE_CUDA "Enable CUDA support (requires NVIDIA GPU)" OFF)

# =============================================================================
# Find Required Packages
# =============================================================================
find_package(OpenMP REQUIRED)

# CUDA Support (Optional)
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

# Project Directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define project root path for use in code
add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

######################################### set source files #########################################
set(DATA_SOURCES
    src/data/cifar10_dataset.cpp
    src/data/data_utils.cpp
)

set(UTILS_SOURCES
    src/utils/logger.cpp
    # src/utils/memory_pool.cpp  
)

set(LAYER_SOURCES
    src/layers/conv2d_cpu.cpp
    src/layers/relu_cpu.cpp
    src/layers/maxpool_cpu.cpp
    src/layers/upsample_cpu.cpp
)

set(MODEL_SOURCES
    src/models/autoencoder_cpu.cpp
)

set(MAIN_SOURCES
    src/main_train.cpp
)

######################################### Set up executables #########################################
add_executable(test_dataloader
    tests/test_dataloader.cpp
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
)

add_executable(test_layers
    tests/test_layers.cpp
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
    ${LAYER_SOURCES}
)

add_executable(test_autoencoder
    tests/test_autoencoder.cpp
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
    ${LAYER_SOURCES}
    ${MODEL_SOURCES}
)

add_executable(train_cpu 
    src/train_cpu.cpp
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
    ${LAYER_SOURCES}
    ${MODEL_SOURCES}
)

#####################################################################

# Link OpenMP
target_link_libraries(train_cpu OpenMP::OpenMP_CXX)
target_link_libraries(test_layers OpenMP::OpenMP_CXX)
target_link_libraries(test_autoencoder OpenMP::OpenMP_CXX)

# Add CUDA support if available
if(CUDA_AVAILABLE)     
    target_link_libraries(train_cpu CUDA::cudart)
    target_compile_definitions(train_cpu PRIVATE USE_CUDA)
endif()


if(MSVC)
    target_compile_options(train_cpu PRIVATE /W4 /O2 /openmp)
    target_compile_options(test_layers PRIVATE /W4 /O2 /openmp)
    target_compile_options(test_autoencoder PRIVATE /W4 /O2 /openmp)
else()
    target_compile_options(train_cpu PRIVATE -Wall -Wextra -O3 -march=native -fopenmp)
    target_compile_options(test_layers PRIVATE -Wall -Wextra -O3 -march=native -fopenmp)
    target_compile_options(test_autoencoder PRIVATE -Wall -Wextra -O3 -march=native -fopenmp)
endif()

# Create data directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights)