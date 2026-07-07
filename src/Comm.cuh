#ifndef COMM_H
#define COMM_H

#include <cstddef>
#include "Settings.cuh"

//
// SPMD communication layer for the multi-GPU (domain-decomposed) build.  The
// solver talks only to this interface; two backends selected at compile time:
//
//   default (loopback):  single process, P PEs emulated as P host threads on one
//     GPU.  neighborExchange is a shared mailbox + barriers, collectives are
//     barrier reductions.  Builds with plain nvcc (no MPI), so it runs and is
//     validated on a machine with no cluster -- and, because it runs the SAME
//     message-passing code path, it validates the actual MPI structure.
//
//   -DUSE_MPI:  real multi-GPU/multi-node over CUDA-aware MPI.  neighborExchange
//     is MPI_Irecv/MPI_Isend/MPI_Waitall on the (device) buffers, collectives are
//     MPI_Allreduce.  Compiled only where an MPI is available.
//
namespace comm {

  void init(int *argc, char ***argv);   // bring up the runtime; pick this PE's GPU
  void finalize();
  int  rank();                          // this PE's id in [0, size)
  int  size();                          // number of PEs
  void barrier();                       // all-PE synchronization

  // Run the per-rank body once per PE.  MPI: each process is one PE, so this just
  // calls fn.  Loopback: spawns `size()` host threads (one per logical PE, each
  // with its own thread-local rank).
  void run(int argc, char **argv, void (*fn)(int, char **));

  // In-place all-PE reductions of a host-accessible vector of n reals.
  void allreduceMin(real *v, int n);
  void allreduceMax(real *v, int n);

  // Point-to-point neighbor exchange: this PE sends sbuf[n] (sbytes[n] bytes) to
  // nbrRank[n] and receives from nbrRank[n] into rbuf[n] (rbytes[n] bytes), for
  // all nNbr neighbors, completing together.  Buffers are device memory.
  //   MPI:       MPI_Irecv/MPI_Isend/MPI_Waitall (CUDA-aware).
  //   loopback:  a shared (src,dst) mailbox + barriers, cudaMemcpy per pair.
  // The halo + topology directory exchanges are built on this one primitive.
  void neighborExchange(int nNbr, const int *nbrRank,
                        void **sbuf, const size_t *sbytes,
                        void **rbuf, const size_t *rbytes);

}

#endif
