#!/usr/bin/env bash
#
# Build the vendored CUDA-aware OpenMPI into extern/openmpi/install, then the
# multi-GPU solver is built with:
#     make wave3d_mgpu USE_MPI=1
# and launched with:
#     extern/openmpi/install/bin/mpirun -np <P> ./wave3d_mgpu --case ...
#
# Prerequisites (Ubuntu/Debian), NOT installed by this script:
#     sudo apt-get install -y autoconf automake libtool m4 flex
# plus a CUDA toolkit (nvcc on PATH).  For a real multi-node/IB deployment you
# also want UCX + GDRCopy + a PMIx; this default config targets single-node
# CUDA-aware P2P (what a dev box can test).
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$(nproc)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MPI_PREFIX="$ROOT/extern/openmpi/install"

echo ">>> OpenMPI (CUDA-aware) -> $MPI_PREFIX"
cd "$ROOT/extern/openmpi"
git submodule update --init --recursive        # openpmix, prrte, ...
[ -x ./configure ] || ./autogen.pl             # git repo ships no configure
./configure --prefix="$MPI_PREFIX" --with-cuda="$CUDA_HOME" \
            --disable-mpi-fortran --enable-mca-no-build=btl-uct
make -j"$JOBS"
make install

cat <<EOF

Done.  Build + run the multi-GPU (MPI) solver:
    make wave3d_mgpu USE_MPI=1
    $MPI_PREFIX/bin/mpirun -np 2 ./wave3d_mgpu --case 2 --nlvls 3
(Makefile picks up MPI_HOME=$MPI_PREFIX by default.)
EOF
