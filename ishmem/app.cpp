#include "mpi.h"
#include <cstdio>
#include <iostream>
#include <ishmem.h>
#include <ishmemx.h>

#define N 10

int main() {
  int provided;
  // FUNNELED or MPI_Init hit segfault.
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  ishmemx_attr_t attr;

  /* Initialize NVSHMEM */
  attr.mpi_comm = &mpi_comm;
  attr.runtime = ISHMEMX_RUNTIME_MPI;
  attr.initialize_runtime = false;
  ishmemx_init_attr(&attr);

  int my_pe = ishmem_my_pe();
  int npes = ishmem_n_pes();

  if (npes < 2) {
    std::cerr << "This example requires at least two PEs." << std::endl;
    ishmem_finalize();
    return 1;
  }

  /* Allocate objects on the symmetric heap, accessible by all PEs */
  int *src = (int *)ishmem_malloc(N * sizeof(int));
  int *dst = (int *)ishmem_malloc(N * sizeof(int));

  // Initialize source and destination arrays
#pragma omp target teams distribute parallel for is_device_ptr(src, dst)
  for (int i = 0; i < N; ++i) {
    src[i] = my_pe * 100 + i;
    dst[i] = -1; // Initialize dst to a different value
  }

  // make sure dst is initialized before receiving data.
  ishmem_barrier_all();

  /* Perform a put operation: my_pe sends its src data to the next PE's dst
   * buffer */
  int target_pe = (my_pe + 1) % npes;

  // ishmem_int_put(destination_addr, source_addr, count, target_pe_id)
  ishmem_int_put(dst, src, N, target_pe);

  /* Use a barrier to ensure all communication is complete across all PEs */
  ishmem_barrier_all();

  /* Verify the data was sent correctly by the current PE (target PE's 'dst'
     should contain the data from my PE's 'src') */
  int *target_dst = (int *)ishmem_ptr(dst, target_pe);
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
  ishmem_free(src);
  ishmem_free(dst);
  ishmem_finalize();
  MPI_Finalize();

  return 0;
}
