#include "Comm.cuh"
#include <cuda_runtime.h>
#include <vector>

#ifdef USE_MPI
// ---------------------------------------------------------------------------
// CUDA-aware MPI backend (compiled only where an MPI is available).
// ---------------------------------------------------------------------------
#include <mpi.h>

namespace comm {

  static int g_rank = 0, g_size = 1;

  void init(int *argc, char ***argv) {
    int provided;
    MPI_Init_thread(argc, argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    int nGpu = 1; cudaGetDeviceCount(&nGpu);   // one GPU per rank within the node
    cudaSetDevice(g_rank % nGpu);
  }
  void finalize() { MPI_Finalize(); }
  int  rank() { return g_rank; }
  int  size() { return g_size; }
  void barrier() { MPI_Barrier(MPI_COMM_WORLD); }
  void run(int argc, char **argv, void (*fn)(int, char **)) { fn(argc, argv); }

#ifdef USE_DOUBLE
  static const MPI_Datatype MPI_REAL_T = MPI_DOUBLE;
#else
  static const MPI_Datatype MPI_REAL_T = MPI_FLOAT;
#endif
  void allreduceMin(real *v, int n) { MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_REAL_T, MPI_MIN, MPI_COMM_WORLD); }
  void allreduceMax(real *v, int n) { MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD); }
  void allreduceSum(double *v, int n) { MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); }

  void neighborExchange(int nNbr, const int *nbr, void **sbuf, const size_t *sbytes,
                        void **rbuf, const size_t *rbytes) {
    std::vector<MPI_Request> req; req.reserve(2*nNbr);
    for (int n = 0; n < nNbr; n++) {
      MPI_Request r; MPI_Irecv(rbuf[n], (int)rbytes[n], MPI_BYTE, nbr[n], 0, MPI_COMM_WORLD, &r); req.push_back(r);
    }
    for (int n = 0; n < nNbr; n++) {
      MPI_Request r; MPI_Isend(sbuf[n], (int)sbytes[n], MPI_BYTE, nbr[n], 0, MPI_COMM_WORLD, &r); req.push_back(r);
    }
    if (!req.empty()) MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);
  }

}

#else
// ---------------------------------------------------------------------------
// Loopback backend: single process emulating P PEs as P host threads on one GPU
// (no MPI).  Runs the identical message-passing code path used with MPI, so it
// validates the decomposition + halo at P>1 on a single-GPU box.
// ---------------------------------------------------------------------------
#include <thread>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>

namespace comm {

  static int g_P = 1;
  static thread_local int tl_rank = 0;
  static pthread_barrier_t g_barrier;
  static std::vector<real>   g_red;    // [P * MAXN] min/max reduction scratch
  static std::vector<double> g_redD;   // [P * MAXN] double-sum reduction scratch

  static const int MAXN = 8;

  // shared (src,dst) mailbox for neighborExchange: g_mail[src*P + dst]
  struct MailSlot { void *ptr; size_t bytes; };
  static MailSlot *g_mail = nullptr;

  void init(int *argc, char ***argv) {
    g_P = 1;
    for (int a = 1; a + 1 < *argc; a++)
      if (strcmp((*argv)[a], "--np") == 0) g_P = atoi((*argv)[a+1]);
    if (g_P < 1) g_P = 1;
    pthread_barrier_init(&g_barrier, nullptr, g_P);
    g_red.assign((size_t)g_P * MAXN, 0);
    g_redD.assign((size_t)g_P * MAXN, 0);
    g_mail = new MailSlot[(size_t)g_P * g_P];
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

  void allreduceSum(double *v, int n) {
    if (g_P == 1) return;
    for (int j = 0; j < n; j++) g_redD[(size_t)tl_rank*MAXN + j] = v[j];
    pthread_barrier_wait(&g_barrier);
    for (int j = 0; j < n; j++) {
      double acc = 0;
      for (int r = 0; r < g_P; r++) acc += g_redD[(size_t)r*MAXN + j];
      v[j] = acc;
    }
    pthread_barrier_wait(&g_barrier);
  }

  // shared-memory rendezvous: publish my sends into the mailbox, barrier, then
  // copy the messages addressed to me out of it (device-to-device).
  void neighborExchange(int nNbr, const int *nbr, void **sbuf, const size_t *sbytes,
                        void **rbuf, const size_t *rbytes) {
    if (g_P == 1) return;
    for (int n = 0; n < nNbr; n++)
      g_mail[(size_t)tl_rank*g_P + nbr[n]] = { sbuf[n], sbytes[n] };
    pthread_barrier_wait(&g_barrier);
    for (int n = 0; n < nNbr; n++) {
      MailSlot m = g_mail[(size_t)nbr[n]*g_P + tl_rank];
      size_t nb = rbytes[n] < m.bytes ? rbytes[n] : m.bytes;
      if (nb) cudaMemcpy(rbuf[n], m.ptr, nb, cudaMemcpyDefault);
    }
    pthread_barrier_wait(&g_barrier);
  }

}

#endif
