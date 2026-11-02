#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#ifndef NCCL_WIN_COLL_SYMMETRIC
#define NCCL_WIN_COLL_SYMMETRIC (1 << 0)
#endif

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__,                   \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("[NCCL ERROR] %s:%d: %s\n", __FILE__, __LINE__,                   \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define N 10

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  int my_pe = 0;
  int npes = 0;
  MPI_Comm_rank(mpi_comm, &my_pe);
  MPI_Comm_size(mpi_comm, &npes);

  if (npes < 2) {
    if (my_pe == 0) {
      std::cerr << "This example requires at least two PEs." << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  /* Assign CUDA & OpenMP device per rank */
  int num_gpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
  int local_rank = my_pe % num_gpus;
  CUDA_CHECK(cudaSetDevice(local_rank));
  omp_set_default_device(local_rank);

  /* Initialize NCCL communicator with unique ID exchange via MPI */
  ncclUniqueId id;
  if (my_pe == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, npes, id, my_pe));

  /* Allocate symmetric memory accessible by all PEs */
  int *src = nullptr;
  int *dst = nullptr;
  NCCL_CHECK(ncclMemAlloc((void **)&src, N * sizeof(int)));
  NCCL_CHECK(ncclMemAlloc((void **)&dst, N * sizeof(int)));

  /* Register symmetric memory windows (LSA) */
  ncclWindow_t win_src;
  ncclWindow_t win_dst;
  NCCL_CHECK(ncclCommWindowRegister(comm, src, N * sizeof(int), &win_src,
                                    NCCL_WIN_COLL_SYMMETRIC));
  NCCL_CHECK(ncclCommWindowRegister(comm, dst, N * sizeof(int), &win_dst,
                                    NCCL_WIN_COLL_SYMMETRIC));

  // Initialize source and destination arrays
#pragma omp target teams distribute parallel for is_device_ptr(src, dst)
  for (int i = 0; i < N; ++i) {
    src[i] = my_pe * 100 + i;
    dst[i] = -1; // Initialize dst to a different value
  }

  // make sure dst is initialized before receiving data.
  MPI_Barrier(mpi_comm);

  /* Perform a put operation: my_pe sends its src data to the next PE's dst
   * buffer */
  int target_pe = (my_pe + 1) % npes;

  // Resolve target PE's dst buffer device pointer via NCCL LSA symmetric memory
  int *target_dst = nullptr;
  NCCL_CHECK(
      ncclGetPeerDevicePointer(win_dst, 0, target_pe, (void **)&target_dst));

  if (target_dst == nullptr) {
    std::cerr << "[PE " << my_pe << "] ERROR: target_dst is NULL for target PE "
              << target_pe << "!\n"
              << "LSA (Load/Store Access) direct peer pointer is unavailable.\n"
              << "Please verify NVLink/P2P connectivity between devices or environment settings."
              << std::endl;
    MPI_Abort(mpi_comm, 1);
  }

  // Directly put source data into target PE's destination buffer
#pragma omp target teams distribute parallel for is_device_ptr(target_dst, src)
  for (int i = 0; i < N; ++i) {
    target_dst[i] = src[i];
  }

  /* Use a barrier to ensure all communication is complete across all PEs */
  MPI_Barrier(mpi_comm);

  /* Verify the data was sent correctly by the current PE (target PE's 'dst'
     should contain the data from my PE's 'src') */
#pragma omp target teams distribute parallel for is_device_ptr(target_dst)
  for (int i = 0; i < N; ++i) {
    if (target_dst[i] != my_pe * 100 + i) {
      printf("[ERROR] Target PE %d validation check failed at index %d : "
             "expected %d, got %d\n",
             target_pe, i, my_pe * 100 + i, target_dst[i]);
    }
  }

  /* Verify the data was received correctly by the next PE (current PE's 'dst'
     should contain the data from the previous PE's 'src') */
  int previous_pe = (my_pe - 1 + npes) % npes;
#pragma omp target teams distribute parallel for is_device_ptr(dst)
  for (int i = 0; i < N; ++i) {
    if (dst[i] != previous_pe * 100 + i) {
      printf("[ERROR] My PE %d validation check failed at index %d : expected "
             "%d, got %d\n",
             my_pe, i, previous_pe * 100 + i, dst[i]);
    }
  }

  if (my_pe == 0) {
    std::cout << "[SUCCESS] Data transfer verified on all PEs." << std::endl;
  }

  /* Free allocated symmetric memory and finalize the library */
  NCCL_CHECK(ncclCommWindowDeregister(comm, win_src));
  NCCL_CHECK(ncclCommWindowDeregister(comm, win_dst));
  NCCL_CHECK(ncclMemFree(src));
  NCCL_CHECK(ncclMemFree(dst));
  NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();

  return 0;
}
