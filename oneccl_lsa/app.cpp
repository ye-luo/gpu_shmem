#include "mpi.h"
#include <cstdio>
#include <iostream>
#include <sycl/sycl.hpp>
#include "oneapi/ccl.hpp"

#define N 10

int main() {
  int provided;
  // FUNNELED or MPI_Init hit segfault.
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
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

  /* Initialize oneCCL */
  ccl::init();

  /* Create KVS and share its address among all ranks via MPI broadcast */
  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type main_addr;
  if (my_pe == 0) {
    kvs = ccl::create_main_kvs();
    main_addr = kvs->get_address();
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, mpi_comm);
  } else {
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, mpi_comm);
    kvs = ccl::create_kvs(main_addr);
  }

  /* Create SYCL queue on GPU and oneCCL communicator */
  sycl::queue q(sycl::gpu_selector_v);
  auto dev = ccl::create_device(q.get_device());
  auto ctx = ccl::create_context(q.get_context());
  auto comm = ccl::create_communicator(npes, my_pe, dev, ctx, kvs);

  /* Allocate objects on device memory */
  int *src = sycl::malloc_device<int>(N, q);
  int *dst = sycl::malloc_device<int>(N, q);

  /* Register symmetric memory windows (accessible by all PEs) */
  ccl::window_t win_src = ccl::comm_window_register_v2(comm, src, N * sizeof(int), 0);
  ccl::window_t win_dst = ccl::comm_window_register_v2(comm, dst, N * sizeof(int), 0);

  // Initialize source and destination arrays
#pragma omp target teams distribute parallel for is_device_ptr(src, dst)
  for (int i = 0; i < N; ++i) {
    src[i] = my_pe * 100 + i;
    dst[i] = -1; // Initialize dst to a different value
  }

  // make sure dst is initialized before receiving data.
  ccl::barrier(comm).wait();

  /* Perform a put operation: my_pe sends its src data to the next PE's dst
   * buffer */
  int target_pe = (my_pe + 1) % npes;

  // Resolve target PE's dst buffer device pointer via oneCCL LSA symmetric memory
  int *target_dst = nullptr;
  ccl::get_peer_device_pointer(win_dst, 0, target_pe, (void **)&target_dst);

  // Directly put source data into target PE's destination buffer
#pragma omp target teams distribute parallel for is_device_ptr(target_dst, src)
  for (int i = 0; i < N; ++i) {
    target_dst[i] = src[i];
  }

  /* Use a barrier to ensure all communication is complete across all PEs */
  ccl::barrier(comm).wait();

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
  ccl::comm_window_deregister(comm, win_src);
  ccl::comm_window_deregister(comm, win_dst);
  sycl::free(src, q);
  sycl::free(dst, q);
  MPI_Finalize();

  return 0;
}
