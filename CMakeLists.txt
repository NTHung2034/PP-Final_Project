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
# =============================================================================
# CPU Test Executables
# =============================================================================
add_executable(test_dataloader
    tests/test_dataloader.cpp
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
)

add_executable(test_layers_cpu
    tests/tests_cpu/test_layers.cpp
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
    ${LAYER_SOURCES}
)

add_executable(test_autoencoder_cpu
    tests/tests_cpu/test_autoencoder.cpp
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
target_link_libraries(test_layers_cpu OpenMP::OpenMP_CXX)
target_link_libraries(test_autoencoder_cpu OpenMP::OpenMP_CXX)

# CPU Compile Options
if(MSVC)
    target_compile_options(train_cpu PRIVATE /W4 /O2 /openmp)
    target_compile_options(test_layers_cpu PRIVATE /W4 /O2 /openmp)
    target_compile_options(test_autoencoder_cpu PRIVATE /W4 /O2 /openmp)
else()
    target_compile_options(train_cpu PRIVATE -Wall -Wextra -O3 -march=native -fopenmp)
    target_compile_options(test_layers_cpu PRIVATE -Wall -Wextra -O3 -march=native -fopenmp)
    target_compile_options(test_autoencoder_cpu PRIVATE -Wall -Wextra -O3 -march=native -fopenmp)
endif()

# Add CUDA support if available
if(CUDA_AVAILABLE)
    # =============================================================================
    # GPU Layer Sources - Optimized Version (Phase 3)
    # =============================================================================
    set(GPU_LAYER_SOURCES
        src/layers_gpu/conv2d_gpu.cu
        src/layers_gpu/maxpool_gpu.cu
        src/layers_gpu/relu_gpu.cu
        src/layers_gpu/upsample_gpu.cu
        src/layers_gpu/mse_loss_gpu.cu
    )

    set(GPU_MODEL_SOURCES
        src/models/autoencoder_gpu.cu
    )

    # =============================================================================
    # GPU Layer Sources - Naive Version (Phase 2)
    # =============================================================================
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

    # =============================================================================
    # GPU Training Executable - Optimized (Phase 3)
    # =============================================================================
    add_executable(train_gpu_optimized
        src/train_gpu.cu
        ${DATA_SOURCES}
        ${UTILS_SOURCES}
        ${GPU_LAYER_SOURCES}
        ${GPU_MODEL_SOURCES}
    )

    # =============================================================================
    # GPU Training Executable - Naive (Phase 2)
    # =============================================================================
    add_executable(train_gpu_naive
        src/train_gpu_naive.cu
        ${DATA_SOURCES}
        ${UTILS_SOURCES}
        ${GPU_LAYER_SOURCES_NAIVE}
        ${GPU_MODEL_SOURCES_NAIVE}
    )

    # =============================================================================
    # Configure Optimized GPU Training Executable
    # =============================================================================
    set_target_properties(train_gpu_optimized PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;89"  
    )

    target_include_directories(train_gpu_optimized PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(train_gpu_optimized 
        CUDA::cudart
        OpenMP::OpenMP_CXX
    )

    target_compile_definitions(train_gpu_optimized PRIVATE USE_CUDA)

    if(MSVC)
        target_compile_options(train_gpu_optimized PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2 /openmp>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/O2>
        )
    else()
        target_compile_options(train_gpu_optimized PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -march=native -fopenmp>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-O3,-fopenmp>
        )
    endif()

    # =============================================================================
    # Configure Naive GPU Training Executable
    # =============================================================================
    set_target_properties(train_gpu_naive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;89"  
    )

    target_include_directories(train_gpu_naive PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(train_gpu_naive 
        CUDA::cudart
        OpenMP::OpenMP_CXX
    )

    target_compile_definitions(train_gpu_naive PRIVATE USE_CUDA)

    if(MSVC)
        target_compile_options(train_gpu_naive PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2 /openmp>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/O2>
        )
    else()
        target_compile_options(train_gpu_naive PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -march=native -fopenmp>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-O3,-fopenmp>
        )
    endif()

    message(STATUS "GPU training executables 'train_gpu_optimized' and 'train_gpu_naive' will be built")

    # =============================================================================
    # GPU Test Executables - Optimized Version
    # =============================================================================
    
    # Test GPU Layers (Optimized)
    add_executable(test_layers_gpu
        tests/tests_gpu/test_layers_gpu.cu
        ${GPU_LAYER_SOURCES}
    )

    set_target_properties(test_layers_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;89"  
    )

    target_include_directories(test_layers_gpu PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(test_layers_gpu CUDA::cudart)
    target_compile_definitions(test_layers_gpu PRIVATE USE_CUDA)

    # Test GPU Autoencoder (Optimized)
    add_executable(test_autoencoder_gpu
        tests/tests_gpu/test_autoencoder_gpu.cu
        ${GPU_LAYER_SOURCES}
        ${GPU_MODEL_SOURCES}
    )

    set_target_properties(test_autoencoder_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;89"  
    )

    target_include_directories(test_autoencoder_gpu PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(test_autoencoder_gpu CUDA::cudart)
    target_compile_definitions(test_autoencoder_gpu PRIVATE USE_CUDA)

    # =============================================================================
    # GPU Test Executables - Naive Version
    # =============================================================================
    
    # Test GPU Layers (Naive)
    add_executable(test_layers_gpu_naive
        tests/tests_gpu_naive/test_layers_gpu_naive.cu
        ${GPU_LAYER_SOURCES_NAIVE}
    )

    set_target_properties(test_layers_gpu_naive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;89"  
    )

    target_include_directories(test_layers_gpu_naive PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(test_layers_gpu_naive CUDA::cudart)
    target_compile_definitions(test_layers_gpu_naive PRIVATE USE_CUDA)

    # Test GPU Autoencoder (Naive)
    add_executable(test_autoencoder_gpu_naive
        tests/tests_gpu_naive/test_autoencoder_gpu_naive.cu
        ${GPU_LAYER_SOURCES_NAIVE}
        ${GPU_MODEL_SOURCES_NAIVE}
    )

    set_target_properties(test_autoencoder_gpu_naive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;89"  
    )

    target_include_directories(test_autoencoder_gpu_naive PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(test_autoencoder_gpu_naive CUDA::cudart)
    target_compile_definitions(test_autoencoder_gpu_naive PRIVATE USE_CUDA)

    # =============================================================================
    # Compile Options for All GPU Tests
    # =============================================================================

    if(MSVC)
        target_compile_options(test_layers_gpu PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/O2>
        )
        target_compile_options(test_autoencoder_gpu PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/O2>
        )
        target_compile_options(test_layers_gpu_naive PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/O2>
        )
        target_compile_options(test_autoencoder_gpu_naive PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/O2>
        )
    else()
        target_compile_options(test_layers_gpu PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-O3>
        )
        target_compile_options(test_autoencoder_gpu PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-O3>
        )
        target_compile_options(test_layers_gpu_naive PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-O3>
        )
        target_compile_options(test_autoencoder_gpu_naive PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-O3>
        )
    endif()

    message(STATUS "GPU test executables will be built:")
    message(STATUS "  - test_layers_gpu (optimized)")
    message(STATUS "  - test_autoencoder_gpu (optimized)")
    message(STATUS "  - test_layers_gpu_naive (naive)")
    message(STATUS "  - test_autoencoder_gpu_naive (naive)")
     
    target_link_libraries(train_cpu CUDA::cudart)
    target_compile_definitions(train_cpu PRIVATE USE_CUDA)
endif()


# Create data and models directories
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights_gpu_naive)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights_gpu_optimized)