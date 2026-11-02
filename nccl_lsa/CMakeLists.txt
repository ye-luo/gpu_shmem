cmake_minimum_required(VERSION 3.13)
project(NCCL_LSA_Example LANGUAGES CXX)

find_package(CUDAToolkit 12.0 REQUIRED)

# Find NCCL
find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{NCCL_DIR}
    PATH_SUFFIXES include)

find_library(NCCL_LIBRARY nccl
    HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{NCCL_DIR}
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

# Add your executable target
add_executable(my_nccl_lsa_app app.cpp)

# Link NCCL and CUDA libraries to your target
target_include_directories(my_nccl_lsa_app PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(my_nccl_lsa_app PUBLIC ${NCCL_LIBRARY} CUDA::cudart)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI COMPONENTS CXX)

target_link_libraries(my_nccl_lsa_app PUBLIC MPI::MPI_CXX)
