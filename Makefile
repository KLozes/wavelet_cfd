# Combined build for both solvers:
#   wave3d        - the compressible flow solver (src/)
#   narrowbandSDF - the narrowband signed distance field solver (sdf/)
# Both executables are placed in this (root) directory and write to ./output.

CC   = g++
NVCC = nvcc

SRC_DIR = src
SDF_DIR = sdf
OBJ_DIR = obj

# shared compile options (CUDA 13 / Thrust need >= c++17; sm_75 = GTX 1650)
ARCH       = sm_75
STD        = c++17
CPPFLAGS   = -O2 -Wextra -std=$(STD)
NVCCFLAGS  = -O2 -std=$(STD) -arch=$(ARCH)
LDFLAGS    = -lpng -lz

# ---- compressible solver (src/*.cu, src/*.cpp) -> wave3d --------------------
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS  = $(wildcard $(SRC_DIR)/*.cu)
SRC_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.cpp.o,$(CPP_SRCS)) \
           $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_SRCS))

# ---- narrowband SDF solver (sdf/*.cu + shared HashTable) -> narrowbandSDF ---
SDF_CU   = $(wildcard $(SDF_DIR)/*.cu)
SDF_OBJS = $(patsubst $(SDF_DIR)/%.cu,$(OBJ_DIR)/sdf_%.cu.o,$(SDF_CU)) \
           $(OBJ_DIR)/HashTable.cu.o

# headers (no automatic dependency tracking, so rebuild on any header change)
HDRS = $(wildcard $(SRC_DIR)/*.cuh) $(wildcard $(SDF_DIR)/*.h) $(wildcard $(SDF_DIR)/*.cuh)

all: narrowbandSDF wave3d

wave3d: $(SRC_OBJS)
	$(NVCC) $(NVCCFLAGS) $(SRC_OBJS) -o $@ $(LDFLAGS)

narrowbandSDF: $(SDF_OBJS)
	$(NVCC) $(NVCCFLAGS) $(SDF_OBJS) -o $@ $(LDFLAGS)

# ---- build rules -----------------------------------------------------------
$(OBJ_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp $(HDRS) | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) -I./$(SRC_DIR) -c $< -o $@

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu $(HDRS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I./$(SRC_DIR) -dc $< -o $@

$(OBJ_DIR)/sdf_%.cu.o: $(SDF_DIR)/%.cu $(HDRS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I./$(SRC_DIR) -I./$(SDF_DIR) -dc $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) wave3d narrowbandSDF

.PHONY: all clean
