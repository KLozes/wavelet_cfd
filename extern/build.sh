#!/usr/bin/env bash
#
# Build the vendored OpenMPI + NVSHMEM submodules into extern/{openmpi,nvshmem}/install,
# then the multi-GPU solver is built with:
#     make wave3d_mgpu USE_NVSHMEM=1
# and launched with:
#     extern/openmpi/install/bin/mpirun -np <P> ./wave3d_mgpu --case ...
#
# Prerequisites (Ubuntu/Debian), NOT installed by this script:
#     sudo apt-get install -y cmake autoconf automake libtool m4 flex
# plus a CUDA toolkit (nvcc on PATH).  For a real multi-node/IB deployment you
# also want UCX + GDRCopy + a PMIx and to enable the matching NVSHMEM transports
# below; this default config targets single-node P2P (what a dev box can test).
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$(nproc)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ARCH="${NVSHMEM_ARCH:-75}"          # sm_75 = GTX 1650; use 80 (A100) / 90 (H100) on real HW
MPI_PREFIX="$ROOT/extern/openmpi/install"
NVSHMEM_PREFIX="$ROOT/extern/nvshmem/install"

echo ">>> [1/2] OpenMPI (CUDA-aware) -> $MPI_PREFIX"
cd "$ROOT/extern/openmpi"
git submodule update --init --recursive        # openpmix, prrte, ...
[ -x ./configure ] || ./autogen.pl             # git repo ships no configure
./configure --prefix="$MPI_PREFIX" --with-cuda="$CUDA_HOME" \
            --disable-mpi-fortran --enable-mca-no-build=btl-uct
make -j"$JOBS"
make install

echo ">>> [2/2] NVSHMEM (MPI bootstrap, single-node P2P) -> $NVSHMEM_PREFIX"
cd "$ROOT/extern/nvshmem"
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX="$NVSHMEM_PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCUDA_HOME="$CUDA_HOME" \
  -DNVSHMEM_MPI_SUPPORT=1 -DMPI_HOME="$MPI_PREFIX" \
  -DNVSHMEM_PMIX_SUPPORT=0 -DNVSHMEM_SHMEM_SUPPORT=0 \
  -DNVSHMEM_IBGDA_SUPPORT=0 -DNVSHMEM_IBRC_SUPPORT=0 -DNVSHMEM_UCX_SUPPORT=0 \
  -DNVSHMEM_IBDEVX_SUPPORT=0 -DNVSHMEM_LIBFABRIC_SUPPORT=0
cmake --build build -j"$JOBS"
cmake --install build

cat <<EOF

Done.  Build + run the NVSHMEM solver:
    make wave3d_mgpu USE_NVSHMEM=1
    $MPI_PREFIX/bin/mpirun -np 2 ./wave3d_mgpu --case 2 --nlvls 3
(Makefile picks up NVSHMEM_HOME=$NVSHMEM_PREFIX and MPI_HOME=$MPI_PREFIX by default.)
EOF
