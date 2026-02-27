cmake_minimum_required(VERSION 3.13)
project(NVIDIASHMEM_Example LANGUAGES CXX) # Specify required languages

# Find the NVSHMEM package.
# ISHMEMROOT environment variable can be used to hint the location.
find_package(NVSHMEM CONFIG REQUIRED)

# Add your executable target
add_executable(my_nvshmem_app app.cpp)

# Link the NVSHMEM library to your target
target_link_libraries(my_nvshmem_app PUBLIC nvshmem::nvshmem_host)

find_package(CUDAToolkit 12.0 REQUIRED)
target_link_libraries(my_nvshmem_app PUBLIC CUDA::cudart)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI COMPONENTS CXX)

target_link_libraries(my_nvshmem_app PUBLIC MPI::MPI_CXX)
