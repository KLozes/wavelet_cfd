# Vendored dependency for the multi-GPU (MPI) build

One git submodule, pinned to a release tag:

| Submodule         | Upstream                          | Pinned tag  | Purpose                                  |
|-------------------|-----------------------------------|-------------|------------------------------------------|
| `extern/openmpi`  | `github.com/open-mpi/ompi`        | `v5.0.10`   | CUDA-aware MPI: the neighbor halo exchange + collectives, and the `mpirun` launcher |

Added `shallow = true`.  After a fresh clone:

    git submodule update --init          # top-level shallow clone
    # (extern/build.sh runs the recursive init OpenMPI needs)

## Why this is here

The solver's multi-GPU path (`-DUSE_MGPU`) talks only to the `comm::` abstraction
(`src/Comm.{cuh,cu}`).  It has two backends:

- **loopback** (default) — single process, P threads on one GPU, no external
  deps.  It runs the *same* message-passing code path (`neighborExchange` +
  allreduce) as the MPI backend, just over a shared-memory mailbox instead of the
  network, so it builds and runs on a box without MPI and is what the P=2
  correctness validation used.
- **MPI** (`-DUSE_MPI`) — the real multi-GPU/multi-node backend, CUDA-aware
  `MPI_Isend`/`MPI_Irecv`/`MPI_Allreduce` on device buffers.  Requires this
  library.

## Building

Toolchain the build needs (NOT vendored; install once):

    sudo apt-get install -y autoconf automake libtool m4 flex

plus a CUDA toolkit (`nvcc` on `PATH`).  Then:

    extern/build.sh                       # -> extern/openmpi/install
    make wave3d_mgpu USE_MPI=1
    extern/openmpi/install/bin/mpirun -np 2 ./wave3d_mgpu --case 2 --nlvls 3

`extern/build.sh` targets a **single-node CUDA-aware P2P** OpenMPI config (no
InfiniBand / UCX) — appropriate for a dev box.  For a real cluster, add the
matching transports (UCX + GDRCopy) in the script.

## Caveat

This was set up on a single GTX 1650 (sm_75, no NVLink) where the toolchain
above is not installed, so the MPI build has **not been compiled or run here** —
the backend code in `src/Comm.cu` under `#ifdef USE_MPI` is unverified until
built on suitable hardware.  The loopback backend runs the identical messaging
code path, so its P=2 validation is strong evidence the MPI path is correct up to
the transport itself.
