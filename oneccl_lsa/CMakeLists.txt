cmake_minimum_required(VERSION 3.13)
project(oneCCL_LSA_Example LANGUAGES CXX)

# Find the Intel oneCCL package.
# CCL_ROOT environment variable can be used to hint the location.
find_package(oneCCL CONFIG REQUIRED)

# Add your executable target
add_executable(my_oneccl_lsa_app app.cpp)

# Link the oneCCL library to your target
target_link_libraries(my_oneccl_lsa_app PUBLIC oneCCL)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI COMPONENTS CXX)

target_link_libraries(my_oneccl_lsa_app PUBLIC MPI::MPI_CXX)
