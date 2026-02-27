#include "mpi.h"
#include <cstdio>
#include <iostream>
#include <nvshmem_host.h>

#define N 10

int main() {
  MPI_Init(NULL, NULL);
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;

  /* Initialize NVSHMEM */
  attr.mpi_comm = &mpi_comm;
  nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int my_pe = nvshmem_my_pe();
  int npes = nvshmem_n_pes();

  if (npes < 2) {
    std::cerr << "This example requires at least two PEs." << std::endl;
    nvshmemx_hostlib_finalize();
    return 1;
  }

  /* Allocate objects on the symmetric heap, accessible by all PEs */
  int *src = (int *)nvshmem_malloc(N * sizeof(int));
  int *dst = (int *)nvshmem_malloc(N * sizeof(int));

  // Initialize source and destination arrays
#pragma omp target teams distribute parallel for is_device_ptr(src, dst)
  for (int i = 0; i < N; ++i) {
    src[i] = my_pe * 100 + i;
    dst[i] = -1; // Initialize dst to a different value
  }

  // make sure dst is initialized before receiving data.
  nvshmem_barrier_all();

  /* Perform a put operation: my_pe sends its src data to the next PE's dst
   * buffer */
  int target_pe = (my_pe + 1) % npes;

  // nvshmem_int_put(destination_addr, source_addr, count, target_pe_id)
  nvshmem_int_put(dst, src, N, target_pe);

  /* Use a barrier to ensure all communication is complete across all PEs */
  nvshmem_barrier_all();

  /* Verify the data was sent correctly by the current PE (target PE's 'dst'
     should contain the data from my PE's 'src') */
  int *target_dst = (int *)nvshmem_ptr(dst, target_pe);
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
  nvshmem_free(src);
  nvshmem_free(dst);
  nvshmemx_hostlib_finalize();
  MPI_Finalize();

  return 0;
}
