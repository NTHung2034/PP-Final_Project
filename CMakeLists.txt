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
option(ENABLE_CUDA "Enable CUDA support (requires NVIDIA GPU)" ON)

# =============================================================================
# Find Required Packages
# =============================================================================
# find_package(OpenMP REQUIRED)

# CUDA Support
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
include_directories(${CMAKE_SOURCE_DIR}/src)

# Define project root path for use in code
add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

######################################### set source files #########################################
# set(UTILS_SOURCES
#     src/utils/logger.cpp
#     src/utils/image_utils.cpp
#     src/utils/memory_tracker.cpp
# )

set(DATA_SOURCES
    src/data/cifar10_loader.cpp
)

set(SVM_SOURCES
    external/svm.cu
    external/libsvm/svm.cpp
)

set(CPU_LAYER_SOURCES
    src/layers/conv2d_cpu.cpp
    src/layers/relu_cpu.cpp
    src/layers/maxpool_cpu.cpp
    src/layers/upsample_cpu.cpp
)

set(CPU_MODEL_SOURCES
    src/models/autoencoder_cpu.cpp
)

######################################### Set up executables #########################################
# =============================================================================
# Test Dataloader Only
# =============================================================================
add_executable(test_dataloader
    tests/test_dataloader.cpp
    ${DATA_SOURCES}
)

# =============================================================================
# CPU Tests 
# =============================================================================
add_executable(test_layers_cpu
    tests/tests_cpu/test_layers.cpp
    ${CPU_LAYER_SOURCES}
)

add_executable(test_autoencoder_cpu
    tests/tests_cpu/test_autoencoder.cpp
    ${CPU_LAYER_SOURCES}
    ${CPU_MODEL_SOURCES}
)

# =============================================================================
# CPU Training
# =============================================================================
add_executable(train_cpu
    src/train_cpu_v2.cpp
    ${DATA_SOURCES}
    ${CPU_LAYER_SOURCES}
    ${CPU_MODEL_SOURCES}
)

# =============================================================================
# GPU Support - Naive Implementation
# =============================================================================
if(CUDA_AVAILABLE)
    # CUDA architecture targets
    set(CUDA_ARCH "89;75;70;61;50")

    set(GPU_LAYER_SOURCES_NAIVE
        src/layers_gpu_naive/conv2d_gpu_naive.cu
        src/layers_gpu_naive/maxpool_gpu_naive.cu
        src/layers_gpu_naive/relu_gpu_naive.cu
        src/layers_gpu_naive/upsample_gpu_naive.cu
        src/layers_gpu_naive/mse_loss_gpu_naive.cu
    )

    set(GPU_MODEL_SOURCES_NAIVE
        src/models/autoencoder_gpu_naive.cu
    )

    # GPU Naive Training
    add_executable(train_gpu_naive
        src/train_gpu_naive.cu
        ${DATA_SOURCES}
        ${GPU_LAYER_SOURCES_NAIVE}
        ${GPU_MODEL_SOURCES_NAIVE}
    )
    set_target_properties(train_gpu_naive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )
    target_link_libraries(train_gpu_naive CUDA::cudart)
    target_compile_definitions(train_gpu_naive PRIVATE USE_CUDA)

    # GPU Naive Tests
    add_executable(test_layers_gpu_naive
        tests/tests_gpu_naive/test_layers_gpu_naive.cu
        ${GPU_LAYER_SOURCES_NAIVE}
    )
    
    add_executable(test_autoencoder_gpu_naive
        tests/tests_gpu_naive/test_autoencoder_gpu_naive.cu
        ${GPU_LAYER_SOURCES_NAIVE}
        ${GPU_MODEL_SOURCES_NAIVE}
    )
    
    set_target_properties(test_layers_gpu_naive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )
    set_target_properties(test_autoencoder_gpu_naive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )

    target_link_libraries(test_layers_gpu_naive CUDA::cudart)
    target_link_libraries(test_autoencoder_gpu_naive CUDA::cudart)

    # GPU Naive Step 4 - SVM Training
    add_executable(train_step4_naive
        src/train_step4_naive.cu
        ${DATA_SOURCES}
        ${GPU_LAYER_SOURCES_NAIVE}
        ${GPU_MODEL_SOURCES_NAIVE}
        ${SVM_SOURCES}
    )
    set_target_properties(train_step4_naive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )
    target_link_libraries(train_step4_naive CUDA::cudart)
    target_compile_definitions(train_step4_naive PRIVATE USE_CUDA)

    message(STATUS "GPU Naive targets: train_gpu_naive, train_step4_naive, test_layers_gpu_naive, test_autoencoder_gpu_naive")

    # -------------------------------------------------------------------------
    # GPU Optimized v1 - Memory Optimization (Shared Memory + Memory Pool)
    # -------------------------------------------------------------------------
    set(GPU_LAYER_SOURCES_OPT_V1
        src/layers_gpu_opt_v1/conv2d_gpu_opt_v1.cu
        src/layers_gpu_opt_v1/relu_gpu_opt_v1.cu
        src/layers_gpu_opt_v1/maxpool_gpu_opt_v1.cu
        src/layers_gpu_opt_v1/upsample_gpu_opt_v1.cu
        src/layers_gpu_opt_v1/mse_loss_gpu_opt_v1.cu
    )

    set(GPU_MODEL_SOURCES_OPT_V1
        src/models/autoencoder_gpu_opt_v1.cu
    )

    add_executable(train_gpu_opt_v1
        src/train_gpu_opt_v1.cu
        ${DATA_SOURCES}
        ${GPU_LAYER_SOURCES_OPT_V1}
        ${GPU_MODEL_SOURCES_OPT_V1}
    )
    set_target_properties(train_gpu_opt_v1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )
    target_link_libraries(train_gpu_opt_v1 CUDA::cudart)
    target_compile_definitions(train_gpu_opt_v1 PRIVATE USE_CUDA)

    # GPU Optimized v1 Tests
    add_executable(test_layers_gpu_opt_v1
        tests/tests_gpu_opt_v1/test_layers_gpu_opt_v1.cu
        ${GPU_LAYER_SOURCES_OPT_V1}
    )
    
    add_executable(test_autoencoder_gpu_opt_v1
        tests/tests_gpu_opt_v1/test_autoencoder_gpu_opt_v1.cu
        ${GPU_LAYER_SOURCES_OPT_V1}
        ${GPU_MODEL_SOURCES_OPT_V1}
    )
    
    set_target_properties(test_layers_gpu_opt_v1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )
    set_target_properties(test_autoencoder_gpu_opt_v1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )

    target_link_libraries(test_layers_gpu_opt_v1 CUDA::cudart)
    target_link_libraries(test_autoencoder_gpu_opt_v1 CUDA::cudart)

    message(STATUS "GPU Optimized v1 targets: train_gpu_opt_v1, test_layers_gpu_opt_v1, test_autoencoder_gpu_opt_v1")

    # -------------------------------------------------------------------------
    # GPU Optimized v2 - CUDA Streams + Kernel Fusion + Full Unrolling
    # -------------------------------------------------------------------------
    set(GPU_LAYER_SOURCES_OPT_V2
        src/layers_gpu_opt_v2/conv2d_gpu_opt_v2.cu
        src/layers_gpu_opt_v2/maxpool_gpu_opt_v2.cu
        src/layers_gpu_opt_v2/upsample_gpu_opt_v2.cu
        src/layers_gpu_opt_v2/mse_loss_gpu_opt_v2.cu
    )

    set(GPU_MODEL_SOURCES_OPT_V2
        src/models/autoencoder_gpu_opt_v2.cu
    )

    add_executable(train_gpu_opt_v2
        src/train_gpu_opt_v2.cu
        ${DATA_SOURCES}
        ${GPU_LAYER_SOURCES_OPT_V2}
        ${GPU_MODEL_SOURCES_OPT_V2}
    )
    set_target_properties(train_gpu_opt_v2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )
    target_link_libraries(train_gpu_opt_v2 CUDA::cudart)
    target_compile_definitions(train_gpu_opt_v2 PRIVATE USE_CUDA)

    # GPU Optimized v2 Tests
    add_executable(test_layers_gpu_opt_v2
        tests/tests_gpu_opt_v2/test_layers_gpu_opt_v2.cu
        ${GPU_LAYER_SOURCES_OPT_V2}
    )
    
    add_executable(test_autoencoder_gpu_opt_v2
        tests/tests_gpu_opt_v2/test_autoencoder_gpu_opt_v2.cu
        ${GPU_LAYER_SOURCES_OPT_V2}
        ${GPU_MODEL_SOURCES_OPT_V2}
    )
    
    set_target_properties(test_layers_gpu_opt_v2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )
    set_target_properties(test_autoencoder_gpu_opt_v2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
    )

    target_link_libraries(test_layers_gpu_opt_v2 CUDA::cudart)
    target_link_libraries(test_autoencoder_gpu_opt_v2 CUDA::cudart)

    message(STATUS "GPU Optimized v2 targets: train_gpu_opt_v2, test_layers_gpu_opt_v2, test_autoencoder_gpu_opt_v2")
endif()

# Create data and models directories
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights_gpu_naive)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights_gpu_optimized)
