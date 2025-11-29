cmake_minimum_required(VERSION 3.18)
project(CUDAAutoencoder CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable CUDA (disabled by default for Phase 1)
option(ENABLE_CUDA "Enable CUDA support" OFF)

# Find packages
find_package(OpenMP REQUIRED)

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# Include directories
include_directories(include)

# Source files - Data Module
set(DATA_SOURCES
    src/data/cifar10_dataset.cpp
    src/data/data_utils.cpp
)

# Source files - Utils Module
set(UTILS_SOURCES
    src/utils/logger.cpp
    src/utils/memory_pool.cpp
)

# Source files - CPU Layers (Phase 1)
set(LAYERS_CPU_SOURCES
    src/layers/conv2d_cpu.cpp
    src/layers/relu_cpu.cpp
    src/layers/maxpool_cpu.cpp
    src/layers/upsample_cpu.cpp
    src/layers/loss_functions.cpp
)

# Source files - CPU Models (Phase 1)
set(MODELS_CPU_SOURCES
    src/models/autoencoder_cpu.cpp
)

# ============================================
# Phase 1: CPU Training Executable
# ============================================
add_executable(train_cpu
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
    ${LAYERS_CPU_SOURCES}
    ${MODELS_CPU_SOURCES}
    src/train_cpu.cpp
)

target_link_libraries(train_cpu 
    OpenMP::OpenMP_CXX
)

target_compile_options(train_cpu PRIVATE
    -Wall -Wextra -O3 -march=native
    -fopenmp
)

set_target_properties(train_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ============================================
# Test Data Loading Executable
# ============================================
add_executable(test_data_loading
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
    tests/test_data_loading.cpp
)

target_link_libraries(test_data_loading 
    OpenMP::OpenMP_CXX
)

target_compile_options(test_data_loading PRIVATE
    -Wall -Wextra -O3 -march=native
    -fopenmp
)

set_target_properties(test_data_loading PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ============================================
# Test CPU Layers Executable
# ============================================
add_executable(test_cpu_layers
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
    ${LAYERS_CPU_SOURCES}
    ${MODELS_CPU_SOURCES}
    tests/test_cpu_layers.cpp
)

target_link_libraries(test_cpu_layers 
    OpenMP::OpenMP_CXX
)

target_compile_options(test_cpu_layers PRIVATE
    -Wall -Wextra -O3 -march=native
    -fopenmp
)

set_target_properties(test_cpu_layers PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ============================================
# GPU Phase 2-3 (Conditional on CUDA)
# ============================================
if(ENABLE_CUDA)
    # GPU source files will be added in Phase 2
    message(STATUS "CUDA support enabled - GPU targets will be built")
    
    # Future: add_executable(train_gpu ...)
else()
    message(STATUS "CUDA support disabled - Building CPU-only targets (Phase 1)")
endif()

# ============================================
# Create necessary directories
# ============================================
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data/cifar-10-batches-bin)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/external)