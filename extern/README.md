# Vendored dependencies for the multi-GPU (NVSHMEM) build

Two git submodules, pinned to release tags:

| Submodule         | Upstream                          | Pinned tag  | Purpose                                  |
|-------------------|-----------------------------------|-------------|------------------------------------------|
| `extern/nvshmem`  | `github.com/NVIDIA/nvshmem`       | `v3.7.1-0`  | GPU-initiated one-sided comm (the halo exchange + collectives) |
| `extern/openmpi`  | `github.com/open-mpi/ompi`        | `v5.0.10`   | CUDA-aware MPI, used to **bootstrap** NVSHMEM (`nvshmemx_init_attr`) and as the `mpirun` launcher |

Both are added `shallow = true`.  After a fresh clone:

    git submodule update --init          # top-level shallow clones
    # (extern/build.sh runs the recursive init OpenMPI needs)

## Why these are here

The solver's multi-GPU path (`-DUSE_MGPU`) talks only to the `comm::` abstraction
(`src/Comm.{cuh,cu}`).  It has two backends:

- **loopback** (default) — single process, P threads on one GPU, no external
  deps.  This is what builds and runs on a box without NVSHMEM/MPI, and is what
  the P=2 correctness validation used.
- **NVSHMEM** (`-DUSE_NVSHMEM`) — the real thing, requiring these two libraries.

## Building

Toolchain the build needs (NOT vendored; install once):

    sudo apt-get install -y cmake autoconf automake libtool m4 flex

plus a CUDA toolkit (`nvcc` on `PATH`).  Then:

    extern/build.sh                       # -> extern/{openmpi,nvshmem}/install
    make wave3d_mgpu USE_NVSHMEM=1
    extern/openmpi/install/bin/mpirun -np 2 ./wave3d_mgpu --case 2 --nlvls 3

`extern/build.sh` targets a **single-node P2P** NVSHMEM config (no InfiniBand /
IBGDA / UCX) — appropriate for a dev box.  For a real cluster, enable the matching
transports in the script and set `NVSHMEM_ARCH` (80 for A100, 90 for H100).

## Caveat

This was set up on a single GTX 1650 (sm_75, no NVLink) where the toolchain
above is not installed, so the NVSHMEM build has **not been compiled or run
here** — the backend code in `src/Comm.cu` under `#ifdef USE_NVSHMEM` is
unverified until built on suitable hardware.
