cmake_minimum_required(VERSION 3.13)
project(NCCL_LSA_Example LANGUAGES CXX)

# Find the NCCL package using CMake Config mode.
# NCCL_ROOT or CMAKE_PREFIX_PATH environment variable can be used to hint the location.
find_package(NCCL CONFIG REQUIRED)

# Add your executable target
add_executable(my_nccl_lsa_app app.cpp)

# Link the NCCL library to your target
if(TARGET NCCL::nccl)
  target_link_libraries(my_nccl_lsa_app PUBLIC NCCL::nccl)
elseif(TARGET nccl::nccl)
  target_link_libraries(my_nccl_lsa_app PUBLIC nccl::nccl)
elseif(TARGET nccl)
  target_link_libraries(my_nccl_lsa_app PUBLIC nccl)
else()
  target_link_libraries(my_nccl_lsa_app PUBLIC NCCL::nccl)
endif()

find_package(CUDAToolkit 12.0 REQUIRED)
target_link_libraries(my_nccl_lsa_app PUBLIC CUDA::cudart)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI COMPONENTS CXX)

target_link_libraries(my_nccl_lsa_app PUBLIC MPI::MPI_CXX)
