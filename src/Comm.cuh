#ifndef COMM_H
#define COMM_H

#include <cstddef>
#include "Settings.cuh"

//
// SPMD communication layer for the multi-GPU (domain-decomposed) build.
// Only compiled/linked into the wave3d_mgpu target.  Two backends, chosen at
// compile time:
//
//   default (loopback):   single process, single PE.  Collectives are the
//     identity, symmetric allocation falls back to cudaMallocManaged.  This is
//     the path that builds and runs on a machine with no NVSHMEM/MPI (e.g. the
//     GTX 1650 dev box) and is used to validate the decomposition logic at P=1
//     bit-for-bit against the single-GPU wave3d binary.
//
//   -DUSE_NVSHMEM:  real GPU-initiated communication over the NVSHMEM symmetric
//     heap.  Compiled only where the NVSHMEM toolkit is present; bootstrapped
//     via MPI (nvshmemx_init_attr).  This is the actual multi-GPU target.
//
// The rest of the solver is backend-agnostic: it calls only this interface.
//
namespace comm {

  void init(int *argc, char ***argv);   // bring up the runtime; pick this PE's GPU
  void finalize();
  int  rank();                          // this PE's id in [0, size)
  int  size();                          // number of PEs
  void barrier();                       // all-PE synchronization

  // Run the per-rank body once per PE.  NVSHMEM: each process is one PE, so this
  // just calls fn.  Loopback: spawns `size()` host threads (one per logical PE,
  // each with its own thread-local rank) so P subdomains can be exercised in a
  // single process on a single GPU for correctness validation.
  void run(int argc, char **argv, void (*fn)(int, char **));

  // Symmetric-heap allocation: the SAME number of bytes on every PE, at a
  // symmetric address, so a remote PE's buffer is addressable for one-sided
  // get/put.  (loopback: plain cudaMallocManaged.)
  void *mallocSym(size_t bytes);
  void  freeSym(void *ptr);

  // In-place all-PE reductions of a host-accessible vector of n reals.
  void allreduceMin(real *v, int n);
  void allreduceMax(real *v, int n);

}

#endif
