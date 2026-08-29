# Build for the solvers, which share the MultiLevelSparseGrid wavelet AMR core:
#   wave3d   - the compressible flow solver              (15 fields/block)
#   wavesdf  - the narrowband signed distance field gen   (2 fields/block)
#   wavefem  - CutFEM linear elasticity on a cut/immersed STL body
# archived (still builds via `make wavewsdf`, not part of `all`):
#   wavewsdf - the wavelet / BVH-oracle SDF + dual contour
# All executables are placed in this (root) directory and write to ./output.
#
# Sources are split one directory per solver, plus src/common for the shared
# core (grid, hash table, comm, geometry/BVH helpers):
#   src/common      - Settings/Util/Vec3f, HashTable, MultiLevelSparseGrid, Comm,
#                     Stl/Features/Bvh/BvhQuery
#   src/fv          - CompressibleSolver + Main/MainMgpu       (wave3d*)
#   src/sdf         - SignedDistanceSolver + MainSdf           (wavesdf)
#   src/dg          - standalone cut-cell DG gates (dg*_test only)
#   src/fem         - CutFemSolver + FemMain                   (wavefem*)
#   src/archive     - retired solvers, kept buildable but out of `all`
#     archive/waveletsdf - WaveletSdfSolver, DualContour, NodalOctree (wavewsdf)
# Each executable sees only src/common and its own directory on the include
# path, so a solver cannot reach into another solver's headers.
#
# The shared core is compiled separately per executable (obj/<exe>/...) so each
# can set its own block-capacity cap (NCELLS_MAX): the SDF stores only 2
# fields/block vs the Euler solver's 16, so it fits ~8x more narrowband blocks in
# the same VRAM and is built with a larger cap for fine-resolution narrowbands.

NVCC = nvcc

SRC_DIR = src
OBJ_DIR = obj

# CUDA 13 / Thrust need >= c++17; sm_89 = RTX 4080 SUPER (was sm_75 = GTX 1650)
ARCH      = sm_89
STD       = c++17
NVCCFLAGS = -O2 -std=$(STD) -arch=$(ARCH)
LDFLAGS   = -lpng -lz -lcusolver -lcusparse

# include paths: shared core + the solver's own directory
INC_COMMON    = -I./$(SRC_DIR)/common
WAVE3D_INC    = $(INC_COMMON) -I./$(SRC_DIR)/fv
WAVESDF_INC   = $(INC_COMMON) -I./$(SRC_DIR)/sdf
WAVEWSDF_INC  = $(INC_COMMON) -I./$(SRC_DIR)/archive/waveletsdf
WAVEFEM_INC   = $(INC_COMMON) -I./$(SRC_DIR)/fem

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
# wavefem materializes the FULL bounding-box background grid before pruning it
# down to the blocks the body actually touches, so the cap must cover the DENSE
# grid, not the active mesh: res^3 cells at the longest axis.  256M cells allows
# res ~ 512 on a cube-ish bbox.  The FEM data itself (24x24 per CUT element,
# 12 reals per stabilized face) scales with the SURFACE, so it is never the
# binding constraint.
WAVEFEM_DEFS  = -DNCELLS_MAX=256000000

# headers (no automatic dependency tracking, so rebuild on any header change)
HDRS = $(wildcard $(SRC_DIR)/*/*.cuh) $(wildcard $(SRC_DIR)/*/*.h) \
       $(wildcard $(SRC_DIR)/*/*/*.cuh) $(wildcard $(SRC_DIR)/*/*/*.h)

COMMON_SRCS  = common/HashTable common/MultiLevelSparseGrid common/MultiLevelSparseGridKernels

WAVE3D_SRCS  = $(COMMON_SRCS) \
               fv/CompressibleSolver fv/CompressibleSolverKernels fv/Main
WAVESDF_SRCS = $(COMMON_SRCS) \
               sdf/SignedDistanceSolver sdf/SignedDistanceSolverKernels sdf/MainSdf
WAVEWSDF_SRCS = $(COMMON_SRCS) \
               archive/waveletsdf/WaveletSdfSolver archive/waveletsdf/WaveletSdfSolverKernels \
               archive/waveletsdf/DualContourGpu archive/waveletsdf/NodalOctree \
               archive/waveletsdf/MainWaveSdf

WAVE3D_OBJS  = $(patsubst %,$(OBJ_DIR)/wave3d/%.cu.o,$(WAVE3D_SRCS))
WAVESDF_OBJS = $(patsubst %,$(OBJ_DIR)/wavesdf/%.cu.o,$(WAVESDF_SRCS))
WAVEWSDF_OBJS = $(patsubst %,$(OBJ_DIR)/wavewsdf/%.cu.o,$(WAVEWSDF_SRCS))
# double-precision Euler build (convergence studies: float roundoff floors
# long-time acoustic errors around 1e-5 relative)
WAVE3D_DP_DEFS = -DUSE_DOUBLE
WAVE3D_DP_OBJS = $(patsubst %,$(OBJ_DIR)/wave3d_dp/%.cu.o,$(WAVE3D_SRCS))
# collapsed 2-D build: blocks are blockSize x blockSize x 1 instead of cubic, so
# a pseudo-2D run allocates and integrates 1/blockSize of the cells.  Separate
# object dir -- blockSizeTot is constexpr, so every object differs from the 3-D
# build.  Only valid for genuinely 2-D cases (nBlocksZ == 1).
WAVE3D_2D_DEFS = -DUSE_DOUBLE -DCOLLAPSE_2D
WAVE3D_2D_OBJS = $(patsubst %,$(OBJ_DIR)/wave3d_2d/%.cu.o,$(WAVE3D_SRCS))

# single-precision pseudo-2D build: the fp32 gate/validation twin of wave3d_2d
WAVE3D_2D_SP_DEFS = -DCOLLAPSE_2D
WAVE3D_2D_SP_OBJS = $(patsubst %,$(OBJ_DIR)/wave3d_2d_sp/%.cu.o,$(WAVE3D_SRCS))

# CutFEM linear elasticity (steady, matrix-free CG -- the only implicit solver
# here).  wavefem_dp is the fp64 build: the convergence study needs errors well
# below the ~1e-5 relative floor fp32 CG leaves.
WAVEFEM_SRCS = $(COMMON_SRCS) \
               fem/CutFemSolver fem/CutFemSolverKernels fem/CutFemIga fem/CutFemSbm fem/FemMain
WAVEFEM_OBJS    = $(patsubst %,$(OBJ_DIR)/wavefem/%.cu.o,$(WAVEFEM_SRCS))
WAVEFEM_DP_DEFS = $(WAVEFEM_DEFS) -DUSE_DOUBLE
WAVEFEM_DP_OBJS = $(patsubst %,$(OBJ_DIR)/wavefem_dp/%.cu.o,$(WAVEFEM_SRCS))

# multi-GPU (domain-decomposed) Euler build.  Same solver + the Comm layer and a
# comm-aware main; -DUSE_MGPU turns on the decomposition paths.  By default it
# builds the LOOPBACK backend (single process/PE, no external deps) so it runs on
# a box without MPI and can be A/B'd against wave3d at P=1.
# modest cap: the loopback backend allocates a full fieldData per PE-thread in
# one process, so keep P*fieldData within the dev GPU (8M cells -> ~0.5 GB/PE).
# Physics is cap-independent, so this still A/B's against the 64M wave3d build.
WAVE3D_MGPU_DEFS = -DNCELLS_MAX=8000000 -DUSE_MGPU
WAVE3D_MGPU_SRCS = $(COMMON_SRCS) common/Comm \
                   fv/CompressibleSolver fv/CompressibleSolverKernels fv/MainMgpu
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

all: wave3d wavesdf wavefem

wave3d: $(WAVE3D_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wavesdf: $(WAVESDF_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wavewsdf: $(WAVEWSDF_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wave3d_dp: $(WAVE3D_DP_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wave3d_2d: $(WAVE3D_2D_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wave3d_2d_sp: $(WAVE3D_2D_SP_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

wavefem: $(WAVEFEM_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp $^ -o $@ $(LDFLAGS)

wavefem_dp: $(WAVEFEM_DP_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp $^ -o $@ $(LDFLAGS)

wave3d_mgpu: $(WAVE3D_MGPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) $(WAVE3D_MGPU_LDFLAGS)

# ---- build rules (one per executable, so each gets its own NCELLS_MAX) ------
# obj/<exe>/ mirrors the src/<dir>/ layout, so the recipe makes the subdirectory.
$(OBJ_DIR)/wave3d/%.cu.o: $(SRC_DIR)/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(WAVE3D_DEFS) $(WAVE3D_INC) -dc $< -o $@

$(OBJ_DIR)/wavesdf/%.cu.o: $(SRC_DIR)/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(WAVESDF_DEFS) $(WAVESDF_INC) -dc $< -o $@

$(OBJ_DIR)/wavewsdf/%.cu.o: $(SRC_DIR)/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(WAVEWSDF_DEFS) $(WAVEWSDF_INC) -dc $< -o $@

$(OBJ_DIR)/wave3d_dp/%.cu.o: $(SRC_DIR)/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(WAVE3D_DP_DEFS) $(WAVE3D_INC) -dc $< -o $@

$(OBJ_DIR)/wave3d_2d/%.cu.o: $(SRC_DIR)/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(WAVE3D_2D_DEFS) $(WAVE3D_INC) -dc $< -o $@

$(OBJ_DIR)/wave3d_2d_sp/%.cu.o: $(SRC_DIR)/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(WAVE3D_2D_SP_DEFS) $(WAVE3D_INC) -dc $< -o $@

# -Xcompiler -fopenmp: the Qp path (CutFemIga.cu) parallelizes its host assembly /
# CG over cores.  The p=1 sources have no OpenMP pragmas, so their generated code
# is unchanged; only libgomp is linked in.
$(OBJ_DIR)/wavefem/%.cu.o: $(SRC_DIR)/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp $(WAVEFEM_DEFS) $(WAVEFEM_INC) -dc $< -o $@

$(OBJ_DIR)/wavefem_dp/%.cu.o: $(SRC_DIR)/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp $(WAVEFEM_DP_DEFS) $(WAVEFEM_INC) -dc $< -o $@

# --default-stream per-thread: the loopback backend runs one host thread per
# logical PE, so give each thread its own default stream (independent syncs).
$(OBJ_DIR)/wave3d_mgpu/%.cu.o: $(SRC_DIR)/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) --default-stream per-thread $(WAVE3D_MGPU_DEFS) $(WAVE3D_INC) -dc $< -o $@

# higher-order (Qp) CutFEM verification drivers: standalone single-file host
# tests (fp64) for the Saye cut quadrature + Qp matrix-free operator being
# developed toward the wavefem higher-order upgrade.  Each is self-checking and
# prints PASS/FAIL or a convergence table; not part of `all`.
#   saye_test - O(h^{p+1}) cut volume/area on a sphere + torus (SayeQuad.h)
#   qp_test   - Qp basis: diff matrix, off-node eval/grad, GLL quadrature
#   qpe_test  - Qp element operator patch test (symmetry, rigid null, const strain)
#   qp_mms    - end-to-end Qp CutFEM MMS convergence (bulk + Nitsche + ghost pen.)
# -Xcompiler -fopenmp: qp_mms parallelizes its host CG operator apply / reductions
# (the other three have no OpenMP pragmas, so the flag is harmless there).
FEMTEST_FLAGS = -O2 -std=$(STD) -arch=$(ARCH) -Xcompiler -fopenmp -DUSE_DOUBLE $(WAVEFEM_INC)

saye_test: $(SRC_DIR)/fem/SayeTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) $< -o $@
qp_test: $(SRC_DIR)/fem/QpTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) $< -o $@
qpe_test: $(SRC_DIR)/fem/QpElemTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) $< -o $@
qp_mms: $(SRC_DIR)/fem/QpMms.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) $< -o $@
sbm_shift_test: $(SRC_DIR)/fem/SbmShiftTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) $< -o $@
sbm_mms: $(SRC_DIR)/fem/SbmMms.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) $< -o $@

# cut-cell DG gate: the discrete divergence theorem on a cut element.  Needs the
# fem include path only for LagrangeBasis.h (the DGSEM nodal basis); the cut
# quadrature itself is in common/.
dgcut_test: $(SRC_DIR)/dg/DgCutTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) -DDG_ORDER=3 $< -o $@

# S1 gate: cut-element operators (boundary-derived moments -> free-stream)
dgcutelem_test: $(SRC_DIR)/dg/DgCutElemTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) -DDG_ORDER=3 $< -o $@

# S2 gate (scheme): cut-element Euler RHS -- free stream + stagnant state
dgcutrhs_test: $(SRC_DIR)/dg/DgCutRhsTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) -DDG_ORDER=3 $< -o $@

# state redistribution gate: conservation, polynomial exactness, contractivity
dgsrd_test: $(SRC_DIR)/dg/DgSrdTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) -DDG_ORDER=3 $< -o $@

# ES1 gate: entropy-stable (skew-hybridized SBP) cut-element operators -- the
# hybridized SBP property, Taylor & Chan's Eq 47, free stream, and discrete
# entropy conservation of the flux-differenced RHS at P1/P2
dges_test: $(SRC_DIR)/dg/DgEsCutTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) -DDG_ORDER=3 $< -o $@

# IGA compressible flow, step 1: 1-D Euler + classic FEM shock capturing (Sod gate)
iga_euler1d: $(SRC_DIR)/fem/IgaEuler1dTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) $< -o $@

# IGA compressible flow, step 2: 2-D Euler + cut-cell cylinder (vortex/fsp/cyl gates)
iga_euler2d: $(SRC_DIR)/fem/IgaEuler2dTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) -Xcompiler -fopenmp $< -o $@

femtests: saye_test qp_test qpe_test qp_mms sbm_shift_test sbm_mms

clean:
	rm -rf $(OBJ_DIR) wave3d wavesdf wavewsdf wave3d_dp wave3d_2d wave3d_mgpu \
	       wavefem wavefem_dp ktau_test ktau_test_sp saye_test qp_test qpe_test qp_mms sbm_shift_test sbm_mms

.PHONY: all clean femtests

# k~-tau~ SST closure gates: the wall function is the integral of Eq. (34), the
# Theta identity, the constant mu_t below the image point, and the Eq. (24)
# near-wall balance with the Appendix-A non-conservative tau~ diffusion fluxes.
ktau_test: $(SRC_DIR)/fv/KtauTest.cu $(HDRS) $(SRC_DIR)/fv/KtauSst.h
	$(NVCC) -O2 -std=$(STD) -arch=$(ARCH) -DUSE_DOUBLE $(WAVE3D_INC) $< -o $@

# ... and the SAME gates in single precision, which is what wave3d actually builds
ktau_test_sp: $(SRC_DIR)/fv/KtauTest.cu $(HDRS) $(SRC_DIR)/fv/KtauSst.h
	$(NVCC) -O2 -std=$(STD) -arch=$(ARCH) $(WAVE3D_INC) $< -o $@

# Tier-0 cut-element Jacobian + runtime-rule-mismatch probe (no solver needed)
dgcutjac_test: $(SRC_DIR)/dg/DgCutJacTest.cu $(HDRS)
	$(NVCC) $(FEMTEST_FLAGS) -DDG_ORDER=3 $< -o $@
