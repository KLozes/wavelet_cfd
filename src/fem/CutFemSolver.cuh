#ifndef CUT_FEM_SOLVER_H
#define CUT_FEM_SOLVER_H

#include <vector>
#include <string>

#include "MultiLevelSparseGrid.cuh"
#include "CutFem.cuh"

//
// wavefem: CutFEM for linear elasticity on a block-sparse background grid
//   (Hansbo, Larson & Larsson, arXiv:1703.04377).
//
// ---------------------------------------------------------------------------
//  The CutFEM paradigm, as realized here
// ---------------------------------------------------------------------------
//
// * BACKGROUND MESH.  The MultiLevelSparseGrid in leafMode at a SINGLE level:
//   one block = blockSize^3 cells, one cell = one trilinear (Q1) hexahedral
//   element.  Nothing about the mesh knows about Omega -- the STL surface cuts
//   elements arbitrarily.
//
// * ACTIVE MESH.  K_h = { K : K \cap Omega != 0 } (2.13): every cell whose
//   trilinear level set goes negative somewhere.  Blocks holding no such cell
//   are never kept, so storage follows the BODY, not its bounding box -- this
//   is the sense in which the mesh is adaptive.  V_h is the restriction of the
//   background Q1 space to N_h(Omega) = union K_h (2.17).
//
// * WEAK BOUNDARY CONDITIONS.  Dirichlet data by Nitsche (2.21); Neumann data
//   in the load (2.27).  No dof is eliminated, so an element holding only a
//   sliver of Omega is harmless to the FORMULATION.
//
// * STABILIZATION.  The sliver harms CONDITIONING and coercivity, which the
//   ghost penalty j_h (2.18) repairs.  For p = 1 only l = 1 survives:
//
//       j_h(v,w) = sum_{F in F_h(dOmega)} h^3 ( [dv/dn], [dw/dn] )_F
//
//   Every face here is axis aligned with both elements at the same level, and
//   a trilinear function's normal derivative is independent of the normal
//   coordinate, so [dv/dn] on F is the BILINEAR function whose four nodal
//   values are the second difference across the face,
//
//       q_m = ( v_farR,m - 2 v_face,m + v_farL,m ) / h .
//
//   Hence  ( [dv/dn], [dw/dn] )_F = q_v^T M q_w  with M the 4x4 unit-square
//   bilinear mass matrix, and the whole face term is a 12-node closed form --
//   no quadrature at all (femFaceApplyKernel).
//
// * SOLVER.  Matrix free.  Only CUT elements carry a stored 24x24 matrix
//   (their quadrature is irregular and expensive); all interior elements share
//   one reference matrix in constant memory, scaled by h.  Jacobi-
//   preconditioned CG -- the paper's diagonal scaling (5.4), which together
//   with j_h delivers the kappa(A) <~ h^-2 of Thm 4.1.
//
// ---------------------------------------------------------------------------
//  Conventions
// ---------------------------------------------------------------------------
//   * corner numbering n = a + 2b + 4c (x,y,z bits), matching q1Shape;
//   * a node is keyed by its integer grid coordinates, so nodes shared between
//     blocks are identified automatically;
//   * one refinement level everywhere => NO hanging nodes: the node dofs are
//     the unknowns and there is no constraint operator.
//

static constexpr i32 nFemFields = 0;    // the solver owns all of its storage

class CutFemSolver : public MultiLevelSparseGrid {
public:

  // ---- problem ------------------------------------------------------------
  FemProblem prob;
  LevelSet   ls;
  real domainOrigin[3] = {0,0,0};   // world position of grid corner (0,0,0)

  // ---- higher order (Qp) --------------------------------------------------
  // femOrder 1 = the p=1 GPU path (everything below).  femOrder >= 2 dispatches
  // to runQp() (CutFemQp.cu): a self-contained host-assembled Qp CutFEM using
  // Saye cut quadrature on a level set sampled at the Qp solution nodes.  It
  // reuses buildMesh() (the sparse octree + oracle) but does its own Qp dof
  // numbering, assembly and CG.  Cartesian only for now.
  i32         femOrder = 1;
  i32         femBasis = 0;    // solution basis inside runQp():
                               // 0 = C^0 Lagrange Q_p (--basis fem, the default
                               //     and the only validated path),
                               // 1 = C^{p-1} uniform B-spline (--basis iga):
                               //     immersed isogeometric / finite-cell.  Same
                               //     geometry (level set + Saye), ~p^3 fewer dofs.
  i32         femMethod = 0;   // 0 = cut-cell (Saye) -- the default;
                               // 1 = GSBM shifted boundary (runSbm, CutFemSbm.cu):
                               //     surrogate domain of FULL cells, shifted
                               //     Nitsche on Gamma~, no cut quadrature.
  std::string outTag;          // output basename (set by FemMain)
  i32         wantVtu = 1;

  // ---- method parameters --------------------------------------------------
  real gammaD;     // Nitsche penalty beta (paper Sec 5: 1000*p^2, p=1 -> 1000)
  real gammaA;     // ghost-penalty strength gamma_a (paper: (2mu+lam)*1e-4);
                   // 0 disables the stabilization entirely (A/B test)
  i32  stabMode;   // 0 = (2.20) gamma_a h^-2 j_h on every stabilized face
                   // 1 = (2.25) that strength only where the cut element
                   //     carries Dirichlet data, weaker gamma_a j_h elsewhere
                   //     ("preferred in practice", Remark 2.1)
  i32  cutSub;     // cut-cell sub-division per direction (1 or 2)

  // ---- cyclic periodicity (cylindrical sector) ----------------------------
  //
  // The two theta faces of the sector are NOT cut surfaces: the computational
  // domain spans exactly one pitch with an integer number of cells, so the
  // boundary lands on cell faces and the node columns at j = 0 and j = nTheta
  // are geometric images of one another under a rotation by the pitch.
  //
  // They are tied algebraically, not by the immersed machinery:
  //     u(j = nTheta) = R(pitch) u(j = 0),
  // R being the rotation about the machine axis.  The displacement is kept in
  // CARTESIAN components -- which is what lets the elasticity operator stay the
  // verified Cartesian one -- so the partner relation carries that rotation
  // rather than being an identity.  Every node therefore maps to a real dof
  // with a flag saying whether R is applied, which is all a single-rotation
  // constraint needs (no general constraint matrix).
  i32  spdCheck = 0;
  i32  rangeTest = 0;
  i32  periodic = 0;
  i32  nThetaCells = 0;
  real pitchAngle = 0;

  // ---- linear solver ------------------------------------------------------
  i32    cgMaxIt;
  real   cgTol;
  i32    cgIters;
  double cgRes;

  // ---- mesh / dof structure (managed; rebuilt by setupDofs) ----------------
  i32  nElem = 0, nCut = 0, nFace = 0, nNode = 0, nReal = 0;
  i32  nCutTrue = 0;              // elements the boundary actually cuts (nCut is
                                  // the size of the stored-matrix bank, which in
                                  // cylindrical mode is every element)

  real *phiBlk  = nullptr;   // [nBlocks*(blockSize+1)^3]   nodal level set per block
  real *nrmBlk  = nullptr;   // [3*nBlocks*(blockSize+1)^3] nodal unit normal (comp. frame)

  i32  *eNode   = nullptr;   // [8*nElem] global node id per corner
  real *eX0     = nullptr;   // [3*nElem] element lower corner (grid frame)
  real *eH      = nullptr;   // [nElem]   element size
  i32  *eCut    = nullptr;   // [nElem]   cut index, or -1 for a full interior element
  i32  *eNbr    = nullptr;   // [6*nElem] face-neighbour element or -1 (-x,+x,-y,+y,-z,+z)

  i32  *cutElem = nullptr;   // [nCut]     owning element
  real *cutPhi  = nullptr;   // [8*nCut]   nodal level set
  real *cutNrm  = nullptr;   // [24*nCut]  nodal unit normal per corner (comp. frame)
  real *cutFpred= nullptr;   // [nCut]     1-jet-predicted cut-volume fraction
  real *cutCoh  = nullptr;   // [nCut]     normal coherence (1 smooth, ->0 crease)
  real *cutK    = nullptr;   // [576*nCut] 24x24 element matrix (volume + Nitsche)
  real *cutF    = nullptr;   // [24*nCut]  element load (volume + Neumann + Nitsche)

  i32  *fNode   = nullptr;   // [12*nFace] 4 far-L, 4 on-face, 4 far-R
  real *fCoef   = nullptr;   // [nFace]    scalar in front of q^T M q

  real *nodeX   = nullptr;   // [3*nNode]
  real *phiNode = nullptr;   // [nNode]
  i32  *nMap    = nullptr;   // [nNode] node -> real dof
  i32  *nRot    = nullptr;   // [nNode] 1 if the pitch rotation applies

  // ---- vectors: unknowns are 3*nReal, node-space scratch is 3*nNode -------
  real *uh = nullptr, *rhs = nullptr, *diag = nullptr;
  real *cgR = nullptr, *cgZ = nullptr, *cgP = nullptr, *cgQ = nullptr;
  real *xn = nullptr, *yn = nullptr;
  double *acc = nullptr;     // [8] managed reduction accumulators
  i32 *fracHist = nullptr;   // [16] cut-volume-fraction decade histogram

  // ---- diagnostics --------------------------------------------------------
  double errL2 = 0, errEnergy = 0, normL2 = 0, normEnergy = 0;
  double volOmega = 0, areaGamma = 0, areaDirich = 0;
  double diagMin = 0, diagMax = 0;
  double slivMin = 1e30, slivMinTheta = 1e30;
  i32 nDiagBad = 0;
  i32 nComp = 0, nCompFree = 0, compMax = 0, freeMax = 0;
  i32 nCleanComp = 1;   // drop free-floating (unanchored) components before solve
  i32 nCrease = 0;
  double cohWorst = 1;
  real creaseCos = 0.9;   // corner-normal coherence below this => cell under-resolved
  double volExact = 0, areaExact = 0;     // from the STL itself (FemMain)
  double meshMs = 0, setupMs = 0, assembleMs = 0, solveMs = 0;

  // lean=true drops the facilities the p=1 / Qp paths never touch (nbrIdxList,
  // prntIdxList, cFlagsList).  Pass lean=false to keep the 27-entry per-block
  // neighbour table -- what the grid-native IGA layout gathers through.  It must
  // be a CONSTRUCTOR argument because sortBlocks() decides whether to populate
  // the table long before femBasis could be assigned to the object.
  CutFemSolver(real *domainSize_, i32 *baseGridSize_, bool lean_ = true) :
    MultiLevelSparseGrid(domainSize_, baseGridSize_, 1, nFemFields, lean_) {
    leafMode  = 1;         // leaf-only: no exterior ring, no parent blocks
    sortCurve = 1;         // space-filling-curve memory order
    gammaD    = 1000;      // p = 1
    gammaA    = -1;        // < 0 -> set from the material in initialize()
    stabMode  = 0;
    cutSub    = 1;
    cgMaxIt   = 20000;
    cgTol     = 1e-10;
    cgIters   = 0;
    cgRes     = 0;
    cudaMallocManaged(&acc, 8*sizeof(double));
    cudaMallocManaged(&fracHist, 16*sizeof(i32));
  }

  ~CutFemSolver(void) { freeMesh(); cudaFree(phiBlk); cudaFree(nrmBlk); cudaFree(acc); cudaFree(fracHist); }

  // ---- driver -------------------------------------------------------------
  void initialize(void);            // reference element matrix, default gamma_a
  void run(void);                   // mesh -> dofs -> assemble -> solve
  void runQp(void);                 // higher-order path (CutFemQp.cu)
  void runSbm(void);                // shifted-boundary path (CutFemSbm.cu)
  void runDensity(void);            // ersatz tanh(phi) density-mask path (CutFemSbm.cu)

  void buildMesh(void);             // dense base grid -> prune to the active set
  void setupDofs(void);             // elements, nodes, neighbours, stabilized faces
  void assemble(void);              // cut-element matrices + global load + diagonal
  void solveCg(void);
  void spdProbe(i32 nTrial = 12);
  void computeErrors(void);         // MMS: L2 and energy norms over Omega

  void applyA(const real *x, real *y);
  double dot(const real *a, const real *b, i32 n);
  void sortFieldData(void) override {}   // no grid-resident fields

  // ---- output -------------------------------------------------------------
  void writeVtu(const char *fileName);         // active elements + displacement
  void writeSurfaceVtu(const char *fileName);  // the cut interface, von Mises
  void report(void);

  // ---- internals ----------------------------------------------------------
  void freeMesh(void);
  void allocVectors(void);
  real cellSize(void) const { return domainSize[0]/(real)baseGridSize[0]; }
};

#endif
