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
  void run(int argc, char **argv, void (*fn)(int, char **)) { fn(argc, argv); }

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
// Loopback backend: single process emulating P PEs as P host threads on one
// GPU.  Builds with plain nvcc (no NVSHMEM/MPI).  Collectives are thread
// barriers + a shared reduction buffer; symmetric alloc is managed memory (all
// PEs share one address space, so a remote PE's buffer is a plain pointer).
// This lets the domain-decomposition logic be validated at P>1 on a dev box.
// ---------------------------------------------------------------------------
#include <thread>
#include <vector>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>

namespace comm {

  static int g_P = 1;
  static thread_local int tl_rank = 0;
  static pthread_barrier_t g_barrier;
  static std::vector<real> g_red;     // [P * MAXN] reduction scratch

  static const int MAXN = 8;

  void init(int *argc, char ***argv) {
    g_P = 1;
    for (int a = 1; a + 1 < *argc; a++)
      if (strcmp((*argv)[a], "--np") == 0) g_P = atoi((*argv)[a+1]);
    if (g_P < 1) g_P = 1;
    pthread_barrier_init(&g_barrier, nullptr, g_P);
    g_red.assign((size_t)g_P * MAXN, 0);
  }
  void finalize() { pthread_barrier_destroy(&g_barrier); }
  int  rank() { return tl_rank; }
  int  size() { return g_P; }
  void barrier() { pthread_barrier_wait(&g_barrier); }

  void run(int argc, char **argv, void (*fn)(int, char **)) {
    if (g_P == 1) { fn(argc, argv); return; }
    std::vector<std::thread> th;
    for (int r = 0; r < g_P; r++)
      th.emplace_back([=]() { tl_rank = r; fn(argc, argv); });
    for (auto &t : th) t.join();
  }

  void *mallocSym(size_t bytes) {
    void *p = nullptr;
    cudaMallocManaged(&p, bytes);
    return p;
  }
  void freeSym(void *ptr) { cudaFree(ptr); }

  // in-place all-reduce of n reals: publish, barrier, everyone reduces the P
  // published rows, barrier.  (n <= MAXN.)
  static void reduce(real *v, int n, bool isMax) {
    if (g_P == 1) return;
    for (int j = 0; j < n; j++) g_red[(size_t)tl_rank*MAXN + j] = v[j];
    pthread_barrier_wait(&g_barrier);
    for (int j = 0; j < n; j++) {
      real acc = g_red[j];
      for (int r = 1; r < g_P; r++) {
        real x = g_red[(size_t)r*MAXN + j];
        acc = isMax ? (x > acc ? x : acc) : (x < acc ? x : acc);
      }
      v[j] = acc;
    }
    pthread_barrier_wait(&g_barrier);
  }
  void allreduceMin(real *v, int n) { reduce(v, n, false); }
  void allreduceMax(real *v, int n) { reduce(v, n, true); }

}

#endif
