cmake_minimum_required(VERSION 3.13)
project(INTELHMEM_Example LANGUAGES CXX) # Specify required languages

# Find the Intel SHMEM package.
# ISHMEMROOT environment variable can be used to hint the location.
find_package(ISHMEM CONFIG REQUIRED)

# Add your executable target
add_executable(my_ishmem_app app.cpp)

# Link the Intel SHMEM library to your target
target_link_libraries(my_ishmem_app PUBLIC ISHMEM::ISHMEM)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI COMPONENTS CXX)

target_link_libraries(my_ishmem_app PUBLIC MPI::MPI_CXX)
