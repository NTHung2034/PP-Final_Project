cmake_minimum_required(VERSION 3.18)
project(CUDAAutoencoder CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(include)

# Source files
set(DATA_SOURCES
    src/data/cifar10_dataset.cpp
    src/data/data_utils.cpp
)

set(UTILS_SOURCES
    src/utils/logger.cpp
)

set(MAIN_SOURCES
    src/main_train.cpp
)

# Create executable
add_executable(train_autoencoder 
    ${DATA_SOURCES}
    ${UTILS_SOURCES}
    ${MAIN_SOURCES}
)

# Link libraries
target_link_libraries(train_autoencoder 
    OpenMP::OpenMP_CXX
    CUDA::cudart  # For future GPU phases
)

# Compiler flags
target_compile_options(train_autoencoder PRIVATE
    -Wall -Wextra -O3 -march=native
    -fopenmp
)

# Set output directories
set_target_properties(train_autoencoder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create data directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models/saved_weights)