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
# default (8M cells ~0.5 GB at 16 fields); wavesdf is 2 fields so 64M cells
# (~1M blocks, ~0.9 GB) fits comfortably on the 4 GB card.
WAVESDF_DEFS = -DNCELLS_MAX=64000000

# headers (no automatic dependency tracking, so rebuild on any header change)
HDRS = $(wildcard $(SRC_DIR)/*.cuh) $(wildcard $(SRC_DIR)/*.h)

WAVE3D_SRCS  = HashTable MultiLevelSparseGrid MultiLevelSparseGridKernels \
               CompressibleSolver CompressibleSolverKernels Main
WAVESDF_SRCS = HashTable MultiLevelSparseGrid MultiLevelSparseGridKernels \
               SignedDistanceSolver SignedDistanceSolverKernels MainSdf

WAVE3D_OBJS  = $(patsubst %,$(OBJ_DIR)/wave3d/%.cu.o,$(WAVE3D_SRCS))
WAVESDF_OBJS = $(patsubst %,$(OBJ_DIR)/wavesdf/%.cu.o,$(WAVESDF_SRCS))

all: wave3d wavesdf

wave3d: $(WAVE3D_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wavesdf: $(WAVESDF_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

# ---- build rules (one per executable, so each gets its own NCELLS_MAX) ------
$(OBJ_DIR)/wave3d/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wave3d
	$(NVCC) $(NVCCFLAGS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wavesdf/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)/wavesdf
	$(NVCC) $(NVCCFLAGS) $(WAVESDF_DEFS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/wave3d $(OBJ_DIR)/wavesdf:
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) wave3d wavesdf

.PHONY: all clean
