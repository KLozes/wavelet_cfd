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
  i32 nSolid = 0, nGeomBad = 0, nSnapF = 0, nSnapS = 0;

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
    if (!anyS) { ibClassList[b] = IB_FLUID; continue; }   // ordinary Cartesian element
    if (!anyF) { ibClassList[b] = IB_DEAD; nSolid++; continue; }  // not evolved

    PolyND phi = fitPoly3(dgOrder, v.data());
    // SHARED-FACE CANONICALIZATION.  The two fits RESTRICT IDENTICALLY to a
    // shared face in exact arithmetic (fitPoly3 interpolates; the face's
    // (p+1)^2 Lobatto samples are the same physical nodes; a tensor degree-p
    // restriction is unique in those values) -- but in floating point each side
    // evaluates from different 3-D coefficients, and at a TANGENCY the ulp
    // differences flip Saye branches: the measured mirror-pair asymmetry
    // (bndIncons 8.5e-6 vs 7.8e-3 on identical geometry) was exactly that.
    // The block loop ascends, so an already-built cut neighbour OWNS the shared
    // face: flip its rule into our frame and pass it as an override.
    std::vector<SayeNode> ovStore[6];
    const std::vector<SayeNode> *ovPtr[6] = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
    {
      const i32 faceOff2[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
      for (i32 fc = 0; fc < 6; fc++) {
        i32 o[3] = {1+faceOff2[fc][0], 1+faceOff2[fc][1], 1+faceOff2[fc][2]};
        i32 nn = nbrIdxList[27*b + o[0] + 3*o[1] + 9*o[2]];
        if (nn == bEmpty || nn < 0) continue;
        // already-built cut neighbour?
        i32 cnb = -1;
        for (size_t cc = 0; cc < blkOf.size(); cc++) if (blkOf[cc] == nn) { cnb = (i32)cc; break; }
        if (cnb < 0) continue;
        const i32 d2 = fc/2, myS = fc%2, nbF = 2*d2 + (1-myS);   // their matching face
        ovStore[fc] = ops[cnb].face[nbF];
        for (SayeNode &sn : ovStore[fc]) sn.x[d2] = (real)1.0 - sn.x[d2];
        ovPtr[fc] = &ovStore[fc];
      }
    }
    CutElemOps E;
    if (!cutElemBuild(phi, N, E, ar, cfg, scratch, 1e-6, ovPtr)) continue;
    // SNAPPED elements: the cut machinery half-saw a sub-resolution feature and
    // its rules were irreparably inconsistent.  Dropping the feature keeps the
    // GCL exact; the cost is the pocket volume, which is below resolution.
    if (E.snap == 1) { ibClassList[b] = IB_FLUID; nSnapF++; continue; }
    if (E.snap == 2) { ibClassList[b] = IB_DEAD;  nSnapS++; nSolid++; continue; }
    if (E.bndIncons > 1e-6) nGeomBad++;
    if (getenv("CUT_DUMPQ"))
      printf("  cut elem (%2d,%2d) h=(%.4f,%.4f,%.4f)  vol %.4f  wall %.4f  "
             "bndIncons %.2e  deg %d\n", ib, jb, h[0], h[1], h[2],
             E.volume, E.wallArea, E.bndIncons, E.B.N);
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
  cudaMallocManaged(&cutFacA, (size_t)6*nCutElem*sizeof(real));
  cudaMallocManaged(&cutBlk,  nCutElem*sizeof(i32));
  cudaMallocManaged(&cutNbOf, nCutElem*sizeof(i32));
  cudaMallocManaged(&cutNbLo, nCutElem*sizeof(i32));
  cudaMallocManaged(&cutM11,  (size_t)nCutElem*CUT_NBMAX_H*CUT_NBMAX_H*sizeof(real));
  memset(cutM11, 0, (size_t)nCutElem*CUT_NBMAX_H*CUT_NBMAX_H*sizeof(real));
  cudaMallocManaged(&cutCen,  (size_t)nCutElem*4*sizeof(real));
  cudaMallocManaged(&cutQual, (size_t)nCutElem*sizeof(real));
  cudaMallocManaged(&cutMinv, (size_t)nCutElem*CUT_NBMAX_H*CUT_NBMAX_H*sizeof(real));
  memset(cutMinv, 0, (size_t)nCutElem*CUT_NBMAX_H*CUT_NBMAX_H*sizeof(real));
  cudaMallocManaged(&blkCut,  nBlocksMax*sizeof(i32));
  for (i32 b = 0; b < nBlocksMax; b++) blkCut[b] = -1;

  size_t nVolT = 0, nWalT = 0, nFacT = 0;
  for (i32 c = 0; c < nCutElem; c++) {
    const CutElemOps &E = ops[c];
    cutBlk[c] = blkOf[c];  blkCut[blkOf[c]] = c;
    cutNbOf[c] = E.B.nb;
    cutNbLo[c] = E.nbLo;
    for (i32 i = 0; i < E.nbLo; i++) for (i32 j = 0; j < E.nbLo; j++)
      cutM11[(size_t)c*CUT_NBMAX_H*CUT_NBMAX_H + (size_t)i*CUT_NBMAX_H + j] =
        (real)E.M11inv[(size_t)i*E.nbLo + j];
    cutQual[c] = (real)E.bndIncons;
    cutCen[4*c+0]=(real)E.B.c[0]; cutCen[4*c+1]=(real)E.B.c[1];
    cutCen[4*c+2]=(real)E.B.c[2]; cutCen[4*c+3]=(real)E.B.s;
    for (const SayeNode &s : E.vol)  cutVolP[nVolT++] = s;
    for (const SayeNode &s : E.wall) cutWalP[nWalT++] = s;
    for (i32 f = 0; f < 6; f++) {
      double a = 0;
      for (const SayeNode &s : E.face[f]) { cutFacP[nFacT++] = s; a += (double)s.w; }
      cutFacA[6*c+f] = (real)a;
    }
    // DENSE inverse mass: the kernel applies M^-1 as a matvec rather than
    // carrying a triangular solve, which is the wrong shape for a GPU thread.
    const i32 nbE = E.B.nb;               // may be < nb on a degenerate sliver
    std::vector<double> col(nbE);
    for (i32 j = 0; j < nbE; j++) {
      for (i32 i = 0; i < nbE; i++) col[i] = (i==j) ? 1.0 : 0.0;
      E.massSolve(col.data());
      for (i32 i = 0; i < nbE; i++)
        cutMinv[(size_t)c*CUT_NBMAX_H*CUT_NBMAX_H + (size_t)i*CUT_NBMAX_H + j] = (real)col[i];
    }
  }

  // ---- the cut path REPLACES the IB machinery ----------------------------
  // Cut elements integrate as ordinary fluid (the wall lives in their flux
  // terms), solid blocks are dead, and the FRIB/ghost/donor machinery must not
  // run at all -- it would re-classify these blocks as IB_GHOST and the RK
  // stage would skip them.
  for (i32 c = 0; c < nCutElem; c++) ibClassList[cutBlk[c]] = IB_FLUID;
  if (ibOn) { printf("cut    : disabling the FRIB/ghost IB machinery (ibOn 1 -> 0)\n"); ibOn = 0; }

  double vmin = 1e300, vmax = 0, qmax = 0;
  for (const CutElemOps &E : ops) {
    vmin = fmin(vmin, E.volume); vmax = fmax(vmax, E.volume);
    qmax = fmax(qmax, E.bndIncons);
  }
  i32 nRed = 0; for (i32 c = 0; c < nCutElem; c++) if (cutNbOf[c] < nb) nRed++;
  printf("cut    : %d cut elements (%d solid blocks skipped), %d modes/elem (%d degree-reduced)\n",
         nCutElem, nSolid, nb, nRed);
  if (nSnapF || nSnapS)
    printf("       : %d snapped to FLUID, %d snapped to SOLID (sub-resolution features dropped)\n",
           nSnapF, nSnapS);
  printf("       : rule pools  vol %zu  wall %zu  face %zu pts   M^-1 %.1f MB\n",
         nVolT, nWalT, nFacT, (double)nCutElem*nb*nb*sizeof(real)/1048576.0);
  printf("       : cut volume fraction %.3e .. %.3e   worst bndIncons %.2e (%d > 1e-6)\n",
         vmin, vmax, qmax, nGeomBad);
  if (nGeomBad)
    printf("       : WARNING %d element(s) geometry-limited -- free stream is only\n"
           "         preserved to their bndIncons; refine the wall band\n", nGeomBad);
  cudaDeviceSynchronize();
}

// one RHS apply with the current mask, then per-class max |RHS| -- the debug
// probe that localises which term of the cut RHS misbehaves.
void DgSolver::probeCutRhs(void) {
  cudaDeviceSynchronize();
  for (i32 q = 0; q < 5; q++)
    cudaMemset(getField(D_RHS+q), 0, (size_t)nBlocksMax*blockSizeTot*sizeof(real));
  dgRhsKernel<<<cudaGridSize, DG_EPB*blockSizeTot>>>(*this, (real)0);
  size_t shm = (5*blockSizeTot + 10*CUT_NBMAX_H + 2)*sizeof(real);
  dgRhsCutKernel<<<nCutElem, blockSizeTot, shm>>>(*this, (real)0);
  cudaDeviceSynchronize();
  double mCut=0, mNbr=0, mFar=0; i32 bCut=-1,bNbr=-1;
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty || ibClassList[b] != IB_FLUID) continue;
    bool isCut = blkCut[b] >= 0, isNbr = false;
    if (!isCut) for (i32 o = 0; o < 27 && !isNbr; o++) {
      i32 nn = nbrIdxList[27*b+o];
      if (nn != bEmpty && nn >= 0 && blkCut[nn] >= 0) isNbr = true;
    }
    for (i32 nd = 0; nd < blockSizeTot; nd++) for (i32 q = 0; q < 5; q++) {
      double v = fabs((double)getField(D_RHS+q)[(size_t)b*blockSizeTot+nd]);
      if (isCut)      { if (v>mCut){mCut=v;bCut=b;} }
      else if (isNbr) { if (v>mNbr){mNbr=v;bNbr=b;} }
      else            { if (v>mFar) mFar=v; }
    }
  }
  printf("[cutprobe] mask=%2d  max|RHS|  cut=%.3e (b%d)  nbr=%.3e (b%d)  far=%.3e\n",
         cutDbgMask, mCut, bCut, mNbr, bNbr, mFar);
}

// ===========================================================================
//  STATE REDISTRIBUTION in the solver.
//
//  Built once, right after the cut operators: the SRD element set is every
//  fluid block within two face-hops of a cut element (two hops because a small
//  cut element's merge neighbourhood can need to grow past another cut
//  element).  Applied on the HOST after every RK stage -- fieldData is managed
//  memory, and at wall-band size (tens of elements) the cost is microseconds;
//  port to a kernel only if a profile ever says so.
//
//  LIMITATION (by design, documented): block indices are captured at build
//  time, so the wall band must be STATIC and unsorted -- the same constraint
//  the cut operators already impose.
// ===========================================================================

#include "StateRedistribution.h"
#include "LagrangeBasis.h"

struct DgSrd {
  SrdOperator            S;
  std::vector<SrdElem>   elems;
  std::vector<SayeNode>  qpool;
  std::vector<i32>       blk;        // SRD element -> block index
  LagrangeBasis          B;          // DG nodal basis on [0,1]
  std::vector<double>    u, su;      // flat state scratch
  i32 nMerged = 0;
};

void DgSolver::buildSrd(void) {
  if (!cutOn || nCutElem == 0) return;
  srd = new DgSrd();
  srd->B.init(dgOrder);

  // ---- collect the element set: cut blocks + two rings of fluid neighbours --
  std::vector<i32> mark(nBlocksMax, 0);
  for (i32 c = 0; c < nCutElem; c++) mark[cutBlk[c]] = 1;
  for (i32 ring = 0; ring < 2; ring++) {
    std::vector<i32> add;
    for (i32 b = 0; b < hashTable.nKeys; b++) {
      if (!mark[b]) continue;
      for (i32 o = 0; o < 27; o++) {
        i32 nn = nbrIdxList[27*b+o];
        if (nn == bEmpty || nn < 0 || mark[nn]) continue;
        if (ibClassList[nn] != IB_FLUID) continue;
        add.push_back(nn);
      }
    }
    for (i32 b : add) mark[b] = 1;
  }

  std::vector<i32> srdOf(nBlocksMax, -1);
  double w[NNODE], xi[NNODE];
  dgGetHostOps(w, xi, gauss);
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty || !mark[b]) continue;
    i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
    double h[3]; hostElemSizeLocal(*this, lvl, h);
    SrdElem E{};
    for (i32 f = 0; f < 6; f++) E.nbr[f] = -1;
    E.x0[0] = ib*h[0]; E.x0[1] = jb*h[1]; E.x0[2] = kb*h[2];
    E.h[0] = h[0]; E.h[1] = h[1]; E.h[2] = h[2];
    E.qOff = (i32)srd->qpool.size();
    i32 c = blkCut[b];
    if (c >= 0) {                          // cut: the fitted positive rule
      for (i32 q = cutVolOff[c]; q < cutVolOff[c+1]; q++) srd->qpool.push_back(cutVolP[q]);
    } else {                               // uncut: tensor GLL collocation rule
      for (i32 kk = 0; kk < NNODE; kk++)
      for (i32 jj = 0; jj < NNODE; jj++)
      for (i32 ii = 0; ii < NNODE; ii++) {
        SayeNode s{};
        s.x[0] = (real)(0.5*(xi[ii]+1.0)); s.x[1] = (real)(0.5*(xi[jj]+1.0));
        s.x[2] = (real)(0.5*(xi[kk]+1.0));
        s.w = (real)(0.125*w[ii]*w[jj]*w[kk]);
        srd->qpool.push_back(s);
      }
    }
    E.qN = (i32)srd->qpool.size() - E.qOff;
    E.vol = 0;
    for (i32 q = E.qOff; q < E.qOff+E.qN; q++) E.vol += (double)srd->qpool[q].w;
    E.vol *= E.hv();
    srdOf[b] = (i32)srd->elems.size();
    srd->elems.push_back(E);
    srd->blk.push_back(b);
  }
  // face neighbours within the set
  const i32 faceOff[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  for (size_t e = 0; e < srd->elems.size(); e++) {
    i32 b = srd->blk[e];
    for (i32 f = 0; f < 6; f++) {
      i32 o[3] = {1+faceOff[f][0], 1+faceOff[f][1], 1+faceOff[f][2]};
      i32 nn = nbrIdxList[27*b + o[0] + 3*o[1] + 9*o[2]];
      if (nn != bEmpty && nn >= 0 && srdOf[nn] >= 0) srd->elems[e].nbr[f] = srdOf[nn];
    }
  }

  srd->S.buildNeighborhoods(srd->elems);
  srd->S.buildReverse();
  // PROJECTION DEGREE 0, deliberately.  ||S|| = 1 holds in L2 -- but L2
  // contractivity is NOT a max principle: at degree N the neighbourhood
  // projection of shock-like data OVERSHOOTS pointwise (measured: max 8.2 in
  // -> 30.2 out on the Mach-3 startup, negative rho, sanitizer feedback,
  // blowup).  Berger & Giuliani's original FV form projects CELL AVERAGES:
  // degree 0 makes S a volume-weighted CONVEX combination of conservative
  // states, positivity-preserving by construction.  Raising this together
  // with the cut elements' own degree needs a trouble detector (cut-MOOD);
  // that is the accuracy roadmap, not the stability baseline.
  { const char *se = getenv("CUT_SRDDEG");
    srd->S.factor(srd->elems, srd->qpool.data(), se ? atoi(se) : 0); }
  for (i32 k = 0; k < srd->S.nElem; k++) if (!srd->S.trivial[k]) srd->nMerged++;
  const size_t nd = (size_t)srd->elems.size()*blockSizeTot*5;
  srd->u.resize(nd); srd->su.resize(nd);
  double vmin = 1e300;
  for (const SrdElem &E : srd->elems) if (E.vol < vmin) vmin = E.vol;
  printf("srd    : %zu elements (%d merged), smallest vol %.3e of full %.3e\n",
         srd->elems.size(), srd->nMerged, vmin,
         srd->elems.empty() ? 0.0 : srd->elems[0].hv());
}

void DgSolver::applySrd(void) {
  if (!srd || srd->nMerged == 0) return;
  if (getenv("CUT_NOSRD")) return;
  cudaDeviceSynchronize();
  const i32 nE = (i32)srd->elems.size();
  for (i32 e = 0; e < nE; e++) {
    i32 b = srd->blk[e];
    for (i32 nd = 0; nd < blockSizeTot; nd++)
      for (i32 q = 0; q < 5; q++)
        srd->u[((size_t)e*blockSizeTot+nd)*5+q] =
          (double)getField(D_RHO+q)[(size_t)b*blockSizeTot+nd];
  }
  srdApply(srd->S, srd->elems, srd->qpool.data(), srd->B,
           srd->u.data(), srd->su.data(), 5);
  if (getenv("SRD_DBG")) {
    static i32 nCall = 0;
    if (nCall < 12) {
      double mi=0, mo=0; i32 eo=-1;
      for (size_t t = 0; t < srd->u.size(); t++) {
        if (fabs(srd->u[t]) > mi) mi = fabs(srd->u[t]);
        if (fabs(srd->su[t]) > mo) { mo = fabs(srd->su[t]); eo = (i32)(t/(blockSizeTot*5)); }
      }
      i32 ei=-1; double mi2=0;
      for (size_t t = 0; t < srd->u.size(); t++)
        if (fabs(srd->u[t]) > mi2) { mi2 = fabs(srd->u[t]); ei = (i32)(t/(blockSizeTot*5)); }
      printf("[srddbg] call %d  max|in| %.3e at srdElem %d (blk %d, %s)   max|out| %.3e\n",
             nCall, mi, ei, ei>=0?srd->blk[ei]:-1,
             (ei>=0 && blkCut[srd->blk[ei]]>=0)?"CUT":"cart", mo);
      nCall++;
    }
  }
  for (i32 e = 0; e < nE; e++) {
    i32 b = srd->blk[e];
    for (i32 nd = 0; nd < blockSizeTot; nd++)
      for (i32 q = 0; q < 5; q++)
        getField(D_RHO+q)[(size_t)b*blockSizeTot+nd] =
          (real)srd->su[((size_t)e*blockSizeTot+nd)*5+q];
  }
}
