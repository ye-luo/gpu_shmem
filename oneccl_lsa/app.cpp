#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include "oneapi/ccl.h"

#define ONECCL_CHECK(cmd)                                                      \
  do {                                                                         \
    onecclResult_t r = cmd;                                                    \
    if (r != onecclSuccess) {                                                  \
      printf("[ONECCL ERROR] %s:%d: %s\n", __FILE__, __LINE__,                 \
             onecclGetErrorString(r));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define N 10

int main(int argc, char *argv[]) {
  int provided;
  // FUNNELED or MPI_Init hit segfault.
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
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

  /* Assign device per rank */
  int num_gpus = omp_get_num_devices();
  std::cout << "num_gpus " << num_gpus << std::endl;
  int local_rank = (num_gpus > 0) ? (my_pe % num_gpus) : 0;
  omp_set_default_device(local_rank);
  ONECCL_CHECK(onecclSetDevice(local_rank));

  /* Initialize oneCCL communicator with unique ID exchange via MPI */
  onecclUniqueId id;
  if (my_pe == 0) {
    ONECCL_CHECK(onecclGetUniqueId(&id));
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);

  onecclComm_t comm;
  ONECCL_CHECK(onecclCommInitRank(&comm, npes, id, my_pe));

  /* Allocate symmetric memory accessible by all PEs */
  int *src = nullptr;
  int *dst = nullptr;
  ONECCL_CHECK(onecclMemAlloc((void **)&src, N * sizeof(int)));
  ONECCL_CHECK(onecclMemAlloc((void **)&dst, N * sizeof(int)));

  /* Register symmetric memory windows (LSA) */
  onecclWindow_t win_src = nullptr;
  onecclWindow_t win_dst = nullptr;
  ONECCL_CHECK(onecclCommWindowRegister(comm, src, N * sizeof(int), &win_src,
                                       ONECCL_WINDOW_DEFAULT));
  ONECCL_CHECK(onecclCommWindowRegister(comm, dst, N * sizeof(int), &win_dst,
                                       ONECCL_WINDOW_DEFAULT));

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

  // Resolve target PE's dst buffer device pointer via oneCCL LSA symmetric memory
  int *target_dst = nullptr;
  ONECCL_CHECK(onecclGetPeerDevicePointer(win_dst, 0, target_pe,
                                         (void **)&target_dst));

  if (target_dst == nullptr) {
    std::cerr << "[PE " << my_pe << "] ERROR: target_dst is NULL for target PE "
              << target_pe << "!\n"
              << "LSA direct peer pointer is unavailable.\n"
              << "Please verify hardware P2P/XeLink connectivity or environment settings "
              << "(e.g., CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0)." << std::endl;
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
  ONECCL_CHECK(onecclCommWindowDeregister(comm, win_src));
  ONECCL_CHECK(onecclCommWindowDeregister(comm, win_dst));
  ONECCL_CHECK(onecclMemFree(src));
  ONECCL_CHECK(onecclMemFree(dst));
  ONECCL_CHECK(onecclCommDestroy(comm));
  MPI_Finalize();

  return 0;
}
