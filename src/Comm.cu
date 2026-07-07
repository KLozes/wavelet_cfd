#include "Comm.cuh"
#include <cuda_runtime.h>

#ifdef USE_NVSHMEM
// ---------------------------------------------------------------------------
// Real NVSHMEM backend (compiled only where the toolkit is available).
// ---------------------------------------------------------------------------
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace comm {

  static int g_rank = 0, g_size = 1;

  void init(int *argc, char ***argv) {
    MPI_Init(argc, argv);
    MPI_Comm mpi = MPI_COMM_WORLD;
    MPI_Comm_rank(mpi, &g_rank);
    MPI_Comm_size(mpi, &g_size);
    // one GPU per PE within the node (rank % gpus-per-node)
    int nGpu = 1; cudaGetDeviceCount(&nGpu);
    cudaSetDevice(g_rank % nGpu);
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &mpi;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    g_rank = nvshmem_my_pe();
    g_size = nvshmem_n_pes();
  }
  void finalize() { nvshmem_finalize(); MPI_Finalize(); }
  int  rank() { return g_rank; }
  int  size() { return g_size; }
  void barrier() { nvshmem_barrier_all(); }

  void *mallocSym(size_t bytes) { return nvshmem_malloc(bytes); }
  void  freeSym(void *ptr) { nvshmem_free(ptr); }

  // Reductions go through a small symmetric scratch + team reduce, then are
  // copied back to the caller's (host-accessible) buffer.
  static void reduce(real *v, int n, bool isMax) {
    real *src = (real*)nvshmem_malloc(n*sizeof(real));
    real *dst = (real*)nvshmem_malloc(n*sizeof(real));
    cudaMemcpy(src, v, n*sizeof(real), cudaMemcpyDefault);
#ifdef USE_DOUBLE
    if (isMax) nvshmem_double_max_reduce(NVSHMEM_TEAM_WORLD, dst, src, n);
    else       nvshmem_double_min_reduce(NVSHMEM_TEAM_WORLD, dst, src, n);
#else
    if (isMax) nvshmem_float_max_reduce(NVSHMEM_TEAM_WORLD, dst, src, n);
    else       nvshmem_float_min_reduce(NVSHMEM_TEAM_WORLD, dst, src, n);
#endif
    cudaMemcpy(v, dst, n*sizeof(real), cudaMemcpyDefault);
    nvshmem_free(src); nvshmem_free(dst);
  }
  void allreduceMin(real *v, int n) { reduce(v, n, false); }
  void allreduceMax(real *v, int n) { reduce(v, n, true); }

}

#else
// ---------------------------------------------------------------------------
// Loopback backend: single process / single PE.  Everything is local; the
// collectives are the identity.  This is what builds on a box without NVSHMEM.
// ---------------------------------------------------------------------------
namespace comm {

  void init(int *, char ***) {}
  void finalize() {}
  int  rank() { return 0; }
  int  size() { return 1; }
  void barrier() {}

  void *mallocSym(size_t bytes) {
    void *p = nullptr;
    cudaMallocManaged(&p, bytes);
    return p;
  }
  void freeSym(void *ptr) { cudaFree(ptr); }

  void allreduceMin(real *, int) {}   // identity at P=1
  void allreduceMax(real *, int) {}

}

#endif
