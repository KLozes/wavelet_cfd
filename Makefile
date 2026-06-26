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

# per-executable cell cap (blocks = NCELLS_MAX/blockSizeTot).  wave3d keeps the
# default (8M cells).  wavesdf runs the grid in lean mode (skips the flow solver's
# cFlagsList/nbrIdxList/prntIdxList/imageDataX); with the fp32 Sdf that is ~310
# B/block, so 384M cells (6M blocks, ~2.2 GB) fit on a 3 GB card -- enough for a
# clean res 2048 (~1.9M blocks).
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

all: wave3d wavesdf wavewsdf

wave3d: $(WAVE3D_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wavesdf: $(WAVESDF_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wavewsdf: $(WAVEWSDF_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

# ---- build rules (one per executable, so each gets its own NCELLS_MAX) ------
$(OBJ_DIR)/wave3d/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wave3d
	$(NVCC) $(NVCCFLAGS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wavesdf/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wavesdf
	$(NVCC) $(NVCCFLAGS) $(WAVESDF_DEFS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wavewsdf/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wavewsdf
	$(NVCC) $(NVCCFLAGS) $(WAVEWSDF_DEFS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wave3d $(OBJ_DIR)/wavesdf $(OBJ_DIR)/wavewsdf:
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) wave3d wavesdf wavewsdf

.PHONY: all clean
