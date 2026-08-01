//
// Cut-cell preprocessing for wavedg3d.
//
// Classifies every leaf element against the immersed geometry, builds the dense
// cut operators once (src/common/CutElem.h), and uploads them.  Runs ONCE after
// the wall band is frozen -- the operators cost NNLS moment fitting per element,
// which cannot be repeated per adaptation.
//
// The state of a cut element stays in the ordinary nodal fieldData; only the
// OPERATORS live here.  See the cut-cell block in DgSolver.cuh for why.
//

#include <cmath>
#include <cstdio>
#include <vector>

#include <cstring>

#include "DgSolver.cuh"
#include "DgSolverKernels.cuh"
#include "Poly.h"
#include "PolyFit.h"
#include "SayeQuad.h"
#include "CutQuadCompress.h"
#include "CutElem.h"

// host mirror of the solver's level-set: fluid is phi > 0 (OUTSIDE the body),
// while Saye's convention is that the ACTIVE region is phi < 0, so the sign is
// flipped at the sampling step and nowhere else.
static double dgHostIbPhi(const DgSolver &g, double x, double y) {
  double dx = x - (double)g.ibX, dy = y - (double)g.ibY;
  return sqrt(dx*dx + dy*dy) - (double)g.ibR;
}

// ONE BLOCK == ONE DG ELEMENT and a block holds blockSize cells, so the element
// size carries a blockSize factor.  Omitting it put every sample point 4x too
// close to the origin and classified the whole grid as fluid.
static void hostElemSizeLocal(const DgSolver &g, i32 lvl, double h[3]) {
  h[0] = (double)g.domainSize[0] / ((double)(g.baseGridSize[0]/blockSize) * powi(2, lvl));
  h[1] = (double)g.domainSize[1] / ((double)(g.baseGridSize[1]/blockSize) * powi(2, lvl));
  h[2] = g.pseudo2D ? (double)g.domainSize[2]
                    : (double)g.domainSize[2] / ((double)(g.baseGridSize[2]/blockSize) * powi(2, lvl));
}

void DgSolver::buildCutElems(void) {
  if (!cutOn) return;
  cudaDeviceSynchronize();

  const i32 N = dgOrder, nb = CutBasis::count(N);
  cutNb = nb;

  double w[NNODE], xi[NNODE];
  dgGetHostOps(w, xi, gauss);            // solution-point coordinates on [-1,1]

  // ---- pass 1: which blocks are cut, and build their operators ------------
  std::vector<SayeNode> arena(1<<21), scratch(1<<18);
  SayeArena ar; ar.buf = arena.data(); ar.cap = 1<<21; ar.top = 0;
  SayeCfg cfg = SayeCfg::def();
  cfg.ng = 10;                           // 5 is NOT enough for the GCL -- measured

  std::vector<i32>       blkOf;
  std::vector<CutElemOps> ops;
  i32 nSolid = 0, nGeomBad = 0;

  for (i32 b = 0; b < hashTable.nKeys; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
    double h[3]; hostElemSizeLocal(*this, lvl, h);

    // sample the level set at the element's OWN solution points: blockSize ==
    // p+1, so the nodes the solver already stores are exactly what fitPoly3
    // wants -- no separate geometry stencil.
    std::vector<real> v((size_t)NNODE*NNODE*NNODE);
    bool anyF = false, anyS = false;
    for (i32 k = 0; k < NNODE; k++)
    for (i32 j = 0; j < NNODE; j++)
    for (i32 i = 0; i < NNODE; i++) {
      double X = (ib + 0.5*(xi[i]+1.0))*h[0];
      double Y = (jb + 0.5*(xi[j]+1.0))*h[1];
      double f = -dgHostIbPhi(*this, X, Y);   // Saye active = phi<0 = FLUID
      v[i + NNODE*(j + NNODE*k)] = (real)f;
      if (f < 0) anyF = true; else anyS = true;
    }
    if (!anyS) continue;                 // wholly fluid: ordinary Cartesian element
    if (!anyF) { nSolid++; continue; }   // wholly solid: not evolved

    PolyND phi = fitPoly3(dgOrder, v.data());
    CutElemOps E;
    if (!cutElemBuild(phi, N, E, ar, cfg, scratch)) continue;
    if (E.bndIncons > 1e-6) nGeomBad++;
    blkOf.push_back(b);
    ops.push_back(std::move(E));
  }
  {
    // diagnostic: how the scan classified things
    i32 nBlk=0; for (i32 b=0;b<hashTable.nKeys;b++) if (bLocList[b]!=kEmpty) nBlk++;
    printf("cut    : scanned %d blocks, ibX=%.4f ibY=%.4f ibR=%.4f, %d solid, %zu cut\n",
           nBlk, (double)ibX, (double)ibY, (double)ibR, nSolid, ops.size());
  }
  nCutElem = (i32)ops.size();
  if (nCutElem == 0) { printf("cut    : no cut elements\n"); cutOn = 0; return; }

  // ---- pass 2: flatten into device pools ---------------------------------
  std::vector<i32> volOff(nCutElem+1,0), walOff(nCutElem+1,0), facOff(6*nCutElem+1,0);
  for (i32 c = 0; c < nCutElem; c++) {
    volOff[c+1] = volOff[c] + (i32)ops[c].vol.size();
    walOff[c+1] = walOff[c] + (i32)ops[c].wall.size();
    for (i32 f = 0; f < 6; f++) facOff[6*c+f+1] = facOff[6*c+f] + (i32)ops[c].face[f].size();
  }
  auto devI = [&](const std::vector<i32> &h_, i32 **d){
    cudaMallocManaged(d, (h_.size()?h_.size():1)*sizeof(i32));
    if (h_.size()) memcpy(*d, h_.data(), h_.size()*sizeof(i32)); };
  auto devS = [&](i32 nTot, SayeNode **d){
    cudaMallocManaged(d, (nTot?nTot:1)*sizeof(SayeNode)); };

  devI(volOff, &cutVolOff); devI(walOff, &cutWalOff); devI(facOff, &cutFacOff);
  devS(volOff[nCutElem], &cutVolP);
  devS(walOff[nCutElem], &cutWalP);
  devS(facOff[6*nCutElem], &cutFacP);
  cudaMallocManaged(&cutBlk,  nCutElem*sizeof(i32));
  cudaMallocManaged(&cutCen,  (size_t)nCutElem*4*sizeof(real));
  cudaMallocManaged(&cutQual, (size_t)nCutElem*sizeof(real));
  cudaMallocManaged(&cutMinv, (size_t)nCutElem*nb*nb*sizeof(real));
  cudaMallocManaged(&blkCut,  nBlocksMax*sizeof(i32));
  for (i32 b = 0; b < nBlocksMax; b++) blkCut[b] = -1;

  size_t nVolT = 0, nWalT = 0, nFacT = 0;
  for (i32 c = 0; c < nCutElem; c++) {
    const CutElemOps &E = ops[c];
    cutBlk[c] = blkOf[c];  blkCut[blkOf[c]] = c;
    cutQual[c] = (real)E.bndIncons;
    cutCen[4*c+0]=(real)E.B.c[0]; cutCen[4*c+1]=(real)E.B.c[1];
    cutCen[4*c+2]=(real)E.B.c[2]; cutCen[4*c+3]=(real)E.B.s;
    for (const SayeNode &s : E.vol)  cutVolP[nVolT++] = s;
    for (const SayeNode &s : E.wall) cutWalP[nWalT++] = s;
    for (i32 f = 0; f < 6; f++) for (const SayeNode &s : E.face[f]) cutFacP[nFacT++] = s;
    // DENSE inverse mass: the kernel applies M^-1 as a matvec rather than
    // carrying a triangular solve, which is the wrong shape for a GPU thread.
    std::vector<double> col(nb);
    for (i32 j = 0; j < nb; j++) {
      for (i32 i = 0; i < nb; i++) col[i] = (i==j) ? 1.0 : 0.0;
      E.massSolve(col.data());
      for (i32 i = 0; i < nb; i++) cutMinv[(size_t)c*nb*nb + (size_t)i*nb + j] = (real)col[i];
    }
  }

  double vmin = 1e300, vmax = 0, qmax = 0;
  for (const CutElemOps &E : ops) {
    vmin = fmin(vmin, E.volume); vmax = fmax(vmax, E.volume);
    qmax = fmax(qmax, E.bndIncons);
  }
  printf("cut    : %d cut elements (%d solid blocks skipped), %d modes/elem\n",
         nCutElem, nSolid, nb);
  printf("       : rule pools  vol %zu  wall %zu  face %zu pts   M^-1 %.1f MB\n",
         nVolT, nWalT, nFacT, (double)nCutElem*nb*nb*sizeof(real)/1048576.0);
  printf("       : cut volume fraction %.3e .. %.3e   worst bndIncons %.2e (%d > 1e-6)\n",
         vmin, vmax, qmax, nGeomBad);
  if (nGeomBad)
    printf("       : WARNING %d element(s) geometry-limited -- free stream is only\n"
           "         preserved to their bndIncons; refine the wall band\n", nGeomBad);
  cudaDeviceSynchronize();
}
