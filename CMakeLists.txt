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
# option(ENABLE_CUDA "Enable CUDA support (requires NVIDIA GPU)" OFF)

# =============================================================================
# Find Required Packages
# =============================================================================
# find_package(OpenMP REQUIRED)

# CUDA Support (Optional)
# if(ENABLE_CUDA)
#     find_package(CUDAToolkit)
#     if(CUDAToolkit_FOUND)
#         message(STATUS "CUDA Toolkit found - GPU support enabled")
#         enable_language(CUDA)
#         set(CUDA_AVAILABLE TRUE)
#     else()
#         message(WARNING "CUDA requested but not found - building CPU-only version")
#         set(CUDA_AVAILABLE FALSE)
#     endif()
# else()
#     message(STATUS "Building CPU-only version (set -DENABLE_CUDA=ON for GPU support)")
#     set(CUDA_AVAILABLE FALSE)
# endif()

# Project Directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define project root path for use in code
add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

######################################### set source files #########################################
set(DATA_SOURCES
    src/data/cifar10_loader.cpp
    # src/data/cifar10_dataset.cpp
    # src/data/data_utils.cpp
)

# set(UTILS_SOURCES
#     src/utils/logger.cpp
#     src/utils/image_utils.cpp
#     src/utils/memory_tracker.cpp
# )

set(LAYER_SOURCES
    src/layers/conv2d_cpu.cpp
    src/layers/relu_cpu.cpp
    src/layers/maxpool_cpu.cpp
    src/layers/upsample_cpu.cpp
)

set(MODEL_SOURCES
    src/models/autoencoder_cpu.cpp
)

# set(MAIN_SOURCES
#     src/main_train.cpp
# )

######################################### Set up executables #########################################
# =============================================================================
# Test Dataloader Only
# =============================================================================
add_executable(test_dataloader
    tests/test_dataloader.cpp
    ${DATA_SOURCES}
)

# =============================================================================
# CPU Training (v2 - raw pointer implementation)
# =============================================================================
add_executable(train_cpu_v2
    src/train_cpu_v2.cpp
    ${DATA_SOURCES}
    ${LAYER_SOURCES}
    ${MODEL_SOURCES}
)

# =============================================================================
# CPU Tests (raw pointer implementation)
# =============================================================================
add_executable(test_layers_cpu
    tests/tests_cpu/test_layers.cpp
    ${LAYER_SOURCES}
)

add_executable(test_autoencoder_cpu
    tests/tests_cpu/test_autoencoder.cpp
    ${LAYER_SOURCES}
    ${MODEL_SOURCES}
)


#####################################################################

# Link OpenMP (commented out for dataloader testing)
# target_link_libraries(test_layers_cpu OpenMP::OpenMP_CXX)
# target_link_libraries(test_autoencoder_cpu OpenMP::OpenMP_CXX)

# CPU Compile Options (commented out for dataloader testing)
# if(MSVC)
#     target_compile_options(test_layers_cpu PRIVATE /W4 /O2 /openmp)
#     target_compile_options(test_autoencoder_cpu PRIVATE /W4 /O2 /openmp)
# else()
#     target_compile_options(test_layers_cpu PRIVATE -Wall -Wextra -O3 -march=native -fopenmp)
#     target_compile_options(test_autoencoder_cpu PRIVATE -Wall -Wextra -O3 -march=native -fopenmp)
# endif()

# =============================================================================
# GPU Support (commented out for dataloader testing)
# =============================================================================
# if(CUDA_AVAILABLE)
#     set(GPU_LAYER_SOURCES
#         src/layers_gpu/conv2d_gpu.cu
#         src/layers_gpu/maxpool_gpu.cu
#         src/layers_gpu/relu_gpu.cu
#         src/layers_gpu/upsample_gpu.cu
#         src/layers_gpu/mse_loss_gpu.cu
#     )
#
#     set(GPU_MODEL_SOURCES
#         src/models/autoencoder_gpu.cu
#     )
#
#     set(GPU_LAYER_SOURCES_NAIVE
#         src/layers_gpu_naive/conv2d_gpu_naive.cu
#         src/layers_gpu_naive/maxpool_gpu_naive.cu
#         src/layers_gpu_naive/relu_gpu_naive.cu
#         src/layers_gpu_naive/upsample_gpu_naive.cu
#         src/layers_gpu_naive/mse_loss_gpu_naive.cu
#     )
#
#     set(GPU_MODEL_SOURCES_NAIVE
#         src/models/autoencoder_gpu_naive.cu
#     )
#
#     add_executable(train_gpu_optimized
#         src/train_gpu.cu
#         ${DATA_SOURCES}
#         ${UTILS_SOURCES}
#         ${GPU_LAYER_SOURCES}
#         ${GPU_MODEL_SOURCES}
#     )
#
#     add_executable(train_gpu_naive
#         src/train_gpu_naive.cu
#         ${DATA_SOURCES}
#         ${UTILS_SOURCES}
#         ${GPU_LAYER_SOURCES_NAIVE}
#         ${GPU_MODEL_SOURCES_NAIVE}
#     )
#
#     set_target_properties(train_gpu_optimized PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON
#         CUDA_ARCHITECTURES "75;80;89"  
#     )
#
#     target_include_directories(train_gpu_optimized PRIVATE 
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/src
#     )
#
#     target_link_libraries(train_gpu_optimized 
#         CUDA::cudart
#         OpenMP::OpenMP_CXX
#     )
#
#     target_compile_definitions(train_gpu_optimized PRIVATE USE_CUDA)
#
#     set_target_properties(train_gpu_naive PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON
#         CUDA_ARCHITECTURES "75;80;89"  
#     )
#
#     target_include_directories(train_gpu_naive PRIVATE 
#         ${CMAKE_SOURCE_DIR}/include
#         ${CMAKE_SOURCE_DIR}/src
#     )
#
#     target_link_libraries(train_gpu_naive 
#         CUDA::cudart
#         OpenMP::OpenMP_CXX
#     )
#
#     target_compile_definitions(train_gpu_naive PRIVATE USE_CUDA)
#
#     # GPU Test Executables
#     add_executable(test_layers_gpu
#         tests/tests_gpu/test_layers_gpu.cu
#         ${GPU_LAYER_SOURCES}
#     )
#
#     add_executable(test_autoencoder_gpu
#         tests/tests_gpu/test_autoencoder_gpu.cu
#         ${GPU_LAYER_SOURCES}
#         ${GPU_MODEL_SOURCES}
#     )
#
#     add_executable(test_layers_gpu_naive
#         tests/tests_gpu_naive/test_layers_gpu_naive.cu
#         ${GPU_LAYER_SOURCES_NAIVE}
#     )
#
#     add_executable(test_autoencoder_gpu_naive
#         tests/tests_gpu_naive/test_autoencoder_gpu_naive.cu
#         ${GPU_LAYER_SOURCES_NAIVE}
#         ${GPU_MODEL_SOURCES_NAIVE}
#     )
# endif()

# Create data and models directories
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights_gpu_naive)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights_gpu_optimized)