# Build for both solvers, which share the MultiLevelSparseGrid wavelet AMR core:
#   wave3d   - the compressible Euler flow solver        (16 fields/block)
#   wavesdf  - the narrowband signed distance field gen   (2 fields/block)
# Both executables are placed in this (root) directory and write to ./output.
#
# The shared core is compiled separately per executable (obj/wave3d, obj/wavesdf)
# so each can set its own block-capacity cap (NCELLS_MAX): the SDF stores only 2
# fields/block vs the Euler solver's 16, so it fits ~8x more narrowband blocks in
# the same VRAM and is built with a larger cap for fine-resolution narrowbands.

NVCC = nvcc

SRC_DIR = src
OBJ_DIR = obj

# CUDA 13 / Thrust need >= c++17; sm_75 = GTX 1650
ARCH      = sm_75
STD       = c++17
NVCCFLAGS = -O2 -std=$(STD) -arch=$(ARCH)
LDFLAGS   = -lpng -lz

# per-executable cell cap (blocks = NCELLS_MAX/blockSizeTot).  wave3d gets 64M
# cells (30 fields x 4B -> ~7.7 GB, fits the 16 GB card) for high-resolution
# adaptive runs (e.g. 8192^2 circular Sod).  wave3d_dp keeps the 8M default
# (doubles: 64M would need ~15.4 GB).  wavesdf runs the grid in lean mode (skips the flow solver's
# cFlagsList/nbrIdxList/prntIdxList/imageDataX); with the fp32 Sdf that is ~310
# B/block, so 384M cells (6M blocks, ~2.2 GB) fit on a 3 GB card -- enough for a
# clean res 2048 (~1.9M blocks).
WAVE3D_DEFS  = -DNCELLS_MAX=64000000
WAVESDF_DEFS = -DNCELLS_MAX=384000000
# wavewsdf (the wavelet / BVH-oracle SDF) stores the 1-jet per cell (value +
# gradient = 16 B/cell).  Its surface-fit octree is ~1000x sparser than a
# narrowband, so 64M cells (1M blocks, ~1 GB) is ample even at high res and fits a
# 3 GB card.
WAVEWSDF_DEFS = -DNCELLS_MAX=64000000

# headers (no automatic dependency tracking, so rebuild on any header change)
HDRS = $(wildcard $(SRC_DIR)/*.cuh) $(wildcard $(SRC_DIR)/*.h)

WAVE3D_SRCS  = HashTable MultiLevelSparseGrid MultiLevelSparseGridKernels \
               CompressibleSolver CompressibleSolverKernels Main
WAVESDF_SRCS = HashTable MultiLevelSparseGrid MultiLevelSparseGridKernels \
               SignedDistanceSolver SignedDistanceSolverKernels MainSdf
WAVEWSDF_SRCS = HashTable MultiLevelSparseGrid MultiLevelSparseGridKernels \
               WaveletSdfSolver WaveletSdfSolverKernels DualContourGpu NodalOctree MainWaveSdf

WAVE3D_OBJS  = $(patsubst %,$(OBJ_DIR)/wave3d/%.cu.o,$(WAVE3D_SRCS))
WAVESDF_OBJS = $(patsubst %,$(OBJ_DIR)/wavesdf/%.cu.o,$(WAVESDF_SRCS))
WAVEWSDF_OBJS = $(patsubst %,$(OBJ_DIR)/wavewsdf/%.cu.o,$(WAVEWSDF_SRCS))
# double-precision Euler build (convergence studies: float roundoff floors
# long-time acoustic errors around 1e-5 relative)
WAVE3D_DP_DEFS = -DUSE_DOUBLE
WAVE3D_DP_OBJS = $(patsubst %,$(OBJ_DIR)/wave3d_dp/%.cu.o,$(WAVE3D_SRCS))

# multi-resolution DGSEM solver (leaf-only AMR, one block = one p=3 element of
# 4^3 LGL nodes).  16M nodes = 250k elements; 17 node-fields ~ 1.1 GB fp32.
# wavedg3d_dp: fp64 for conservation/convergence studies (halved node cap).
WAVEDG_DEFS    = -DNCELLS_MAX=32000000 -DDG_ORDER=3
WAVEDG_DP_DEFS = -DNCELLS_MAX=8000000 -DDG_ORDER=3 -DUSE_DOUBLE
WAVEDG_SRCS = HashTable MultiLevelSparseGrid MultiLevelSparseGridKernels \
              DgSolver DgSolverKernels DgMain
WAVEDG_OBJS    = $(patsubst %,$(OBJ_DIR)/wavedg3d/%.cu.o,$(WAVEDG_SRCS))
WAVEDG_DP_OBJS = $(patsubst %,$(OBJ_DIR)/wavedg3d_dp/%.cu.o,$(WAVEDG_SRCS))

# multi-GPU (domain-decomposed) Euler build.  Same solver + the Comm layer and a
# comm-aware main; -DUSE_MGPU turns on the decomposition paths.  By default it
# builds the LOOPBACK backend (single process/PE, no external deps) so it runs on
# a box without MPI and can be A/B'd against wave3d at P=1.
# modest cap: the loopback backend allocates a full fieldData per PE-thread in
# one process, so keep P*fieldData within the dev GPU (8M cells -> ~0.5 GB/PE).
# Physics is cap-independent, so this still A/B's against the 64M wave3d build.
WAVE3D_MGPU_DEFS = -DNCELLS_MAX=8000000 -DUSE_MGPU
WAVE3D_MGPU_SRCS = HashTable MultiLevelSparseGrid MultiLevelSparseGridKernels \
                   CompressibleSolver CompressibleSolverKernels Comm MainMgpu
WAVE3D_MGPU_OBJS = $(patsubst %,$(OBJ_DIR)/wave3d_mgpu/%.cu.o,$(WAVE3D_MGPU_SRCS))

# CUDA-aware MPI backend (opt-in: `make wave3d_mgpu USE_MPI=1`).  Uses the
# vendored OpenMPI (extern/build.sh installs it there); override MPI_HOME to point
# at a system MPI instead.  Default target stays the loopback backend (no MPI).
MPI_HOME ?= $(CURDIR)/extern/openmpi/install
WAVE3D_MGPU_LDFLAGS =
ifeq ($(USE_MPI),1)
  WAVE3D_MGPU_DEFS   += -DUSE_MPI -I$(MPI_HOME)/include
  # -rpath embeds the vendored lib dir so the binary finds libmpi.so at load
  # without needing LD_LIBRARY_PATH set.
  WAVE3D_MGPU_LDFLAGS = -L$(MPI_HOME)/lib -lmpi -Xlinker -rpath -Xlinker $(MPI_HOME)/lib
endif

all: wave3d wavesdf wavewsdf wavedg3d

wave3d: $(WAVE3D_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wavesdf: $(WAVESDF_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wavewsdf: $(WAVEWSDF_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wave3d_dp: $(WAVE3D_DP_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wavedg3d: $(WAVEDG_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wavedg3d_dp: $(WAVEDG_DP_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wave3d_mgpu: $(WAVE3D_MGPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) $(WAVE3D_MGPU_LDFLAGS)

# ---- build rules (one per executable, so each gets its own NCELLS_MAX) ------
$(OBJ_DIR)/wave3d/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wave3d
	$(NVCC) $(NVCCFLAGS) $(WAVE3D_DEFS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wavesdf/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wavesdf
	$(NVCC) $(NVCCFLAGS) $(WAVESDF_DEFS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wavewsdf/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wavewsdf
	$(NVCC) $(NVCCFLAGS) $(WAVEWSDF_DEFS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wave3d_dp/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wave3d_dp
	$(NVCC) $(NVCCFLAGS) $(WAVE3D_DP_DEFS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wavedg3d/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wavedg3d
	$(NVCC) $(NVCCFLAGS) $(WAVEDG_DEFS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wavedg3d_dp/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wavedg3d_dp
	$(NVCC) $(NVCCFLAGS) $(WAVEDG_DP_DEFS) -I./$(SRC_DIR) -dc $< -o $@

# --default-stream per-thread: the loopback backend runs one host thread per
# logical PE, so give each thread its own default stream (independent syncs).
$(OBJ_DIR)/wave3d_mgpu/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wave3d_mgpu
	$(NVCC) $(NVCCFLAGS) --default-stream per-thread $(WAVE3D_MGPU_DEFS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wave3d $(OBJ_DIR)/wavesdf $(OBJ_DIR)/wavewsdf $(OBJ_DIR)/wave3d_dp $(OBJ_DIR)/wave3d_mgpu $(OBJ_DIR)/wavedg3d $(OBJ_DIR)/wavedg3d_dp:
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) wave3d wavesdf wavewsdf wave3d_dp wave3d_mgpu wavedg3d wavedg3d_dp

.PHONY: all clean
