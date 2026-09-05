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
// CUT_PHIDIST=1 restores the DISTANCE form below; the default is the QUADRATIC
// one.  Every cut rule is built from fitPoly3's degree-p interpolant of this
// function at the element's own solution nodes, so the geometry the rules
// integrate is the FIT, not this function.  r - R is not a polynomial, so its
// cubic interpolant has O(h^4) error and a zero set that wobbles relative to the
// true circle, differently on a face than in the interior.  r^2 - R^2 has the
// SAME ZERO SET and is a quadratic, which a cubic fit reproduces EXACTLY, so the
// volume, face and wall rules all integrate one consistent algebraic surface.
// MEASURED on case 9, p3, --cutmodal: fp32 free-stream |RHS| 8.245 -> 1.866 and
// worst bndIncons 7.72e-04 -> 4.08e-04.  In fp64 the two forms are equivalent
// (1.7e-09 vs 3.0e-09) -- the fit error only dominates once float rounding does
// not, so this is an fp32 robustness fix, not an accuracy one.
static double dgHostIbPhi(const DgSolver &g, double x, double y) {
  double dx = x - (double)g.ibX, dy = y - (double)g.ibY;
  static const bool useDist = (getenv("CUT_PHIDIST") != nullptr);
  if (useDist) return sqrt(dx*dx + dy*dy) - (double)g.ibR;
  return dx*dx + dy*dy - (double)g.ibR*(double)g.ibR;
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

// ---------------------------------------------------------------------------
//  TANGENCY GUARD.  A body tangent to a grid line is a DEGENERATE input, and it
//  is not exotic -- case 9 ships with R = 0.5 and h = domx/nblocks = 0.25, i.e.
//  R/h = 2.0000 exactly, so the cylinder touches four cell edges by
//  construction.  Measured cost of that one coincidence:
//    * the node-sign classifier flags 20 blocks as cut when only 12 are (the
//      8 extras touch the body at a single point and come back wallArea == 0);
//    * Saye subdivides the tangent faces 15x, 1500 points on a unit face;
//    * the face rules lose polynomial exactness: 8.4e-03 against closed-form
//      integrals, versus 8.0e-10 as soon as the tangency is broken;
//    * the cut volume fraction collapses to 0.087 (wedges) from 0.45;
//    * the volume rule's moment error goes 1.8e-10 -> 4.1e-04.
//  Every one of those recovers the moment the geometry is non-degenerate, and
//  the perturbation needed is tiny: 2e-6 of the radius is not enough, 2e-5 is
//  (measured, both directions).  So: detect it and step off it.
//
//  The perturbation is GLOBAL and applied to the sampled level set, not per
//  cell -- neighbours must agree about where the interface is, or the shared
//  face rules stop matching.
// ---------------------------------------------------------------------------
void DgSolver::buildCutElems(void) {
  if (!cutOn) return;
  cudaDeviceSynchronize();
  if (cutEps == (real)0 && !getenv("CUT_NOGEOMGUARD")) {
    // --- DEGENERACY AUDIT + ESCALATING SHIFT ------------------------------
    // Probe the geometry at a candidate shift and score how degenerate it is.
    // Four signatures, each measured on this exact case and each with a
    // documented downstream cost:
    //   T  tangency      node-flagged block with ZERO wall area.  The
    //                    classifier calls it cut, the surface inside has no
    //                    measure.  Costs: 8 false cuts, all snapped away.
    //   S  sliver        fluid fraction within kSliver of 0 or 1.  A cell that
    //                    is essentially all one phase but carries cut
    //                    machinery, mass-matrix conditioning and a wall.
    //   N  needle face   a face with a nonzero but negligible fluid area.
    //                    100 quadrature points on 4e-05 of a face.
    // and one COST signal that is deliberately NOT part of the verdict:
    //   X  subdivision   Saye exploding the point count on a near-tangent face
    //                    (1500 points on a unit face).  This is expensive but
    //                    NOT wrong: at R/h = 1.65 eight faces still trip it and
    //                    those same rules are polynomially exact to 8.0e-10
    //                    (verified against closed-form integrals).  Counting it
    //                    as degeneracy made the guard cry wolf on a geometry
    //                    that is fine, so it is reported and not acted on.
    // Shifts are tried in increasing size and the FIRST clean one wins; the
    // measured threshold on case 9 is between 2e-6 and 2e-5 of the radius, so
    // the ladder starts below it and steps past.  Positive shrinks the body --
    // growing it instead manufactures sub-resolution solid slivers (measured:
    // bndIncons 1.7e-10 -> 2.1e-04).  The shift is GLOBAL: neighbours must
    // agree on where the interface is or the shared face rules stop matching.
    const double kSliver = 1e-3, kNeedle = 1e-3;
    double hMin = 1e30;
    for (i32 b = 0; b < hashTable.nKeys; b++) {
      if (bLocList[b] == kEmpty) continue;
      i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
      double hh[3]; hostElemSizeLocal(*this, lvl, hh);
      hMin = fmin(hMin, fmin(hh[0], hh[1]));
    }
    double w0[NNODE], xi0[NNODE];
    dgGetHostOps(w0, xi0, gauss);
    std::vector<SayeNode> arena0(1<<21), scratch0(1<<18);
    SayeArena ar0; ar0.buf=arena0.data(); ar0.cap=1<<21; ar0.top=0;
    SayeCfg cfg0 = SayeCfg::def(); cfg0.ng = 10;
    if (getenv("CUT_MAXDEPTH")) cfg0.maxDepth = atoi(getenv("CUT_MAXDEPTH"));

    auto audit = [&](double eps, i32 cnt[4]) {
      cnt[0]=cnt[1]=cnt[2]=cnt[3]=0;
      for (i32 b = 0; b < hashTable.nKeys; b++) {
        u64 loc = bLocList[b];
        if (loc == kEmpty) continue;
        i32 lvl, ib, jb, kb; decode(loc, lvl, ib, jb, kb);
        double hh[3]; hostElemSizeLocal(*this, lvl, hh);
        std::vector<real> v0((size_t)NNODE*NNODE*NNODE);
        bool anyF=false, anyS=false;
        for (i32 k = 0; k < NNODE; k++)
        for (i32 j = 0; j < NNODE; j++)
        for (i32 i = 0; i < NNODE; i++) {
          double X = (ib + 0.5*(xi0[i]+1.0))*hh[0];
          double Y = (jb + 0.5*(xi0[j]+1.0))*hh[1];
          double f = -dgHostIbPhi(*this, X, Y) - eps;
          v0[i + NNODE*(j + NNODE*k)] = (real)f;
          if (f < 0) anyF = true; else anyS = true;
        }
        if (!anyF || !anyS) continue;
        PolyND phi0 = fitPoly3(dgOrder, v0.data());
        SayeSet ws; ws.p=scratch0.data(); ws.n=0; ws.cap=1<<18; ws.ovf=false;
        sayeSurface(phi0, &ws, &ar0, cfg0);
        double wa=0; for (i32 q=0;q<ws.n;q++) wa += (double)ws.p[q].w;
        if (wa <= 1e-12) { cnt[0]++; continue; }            // T
        SayeSet vs; vs.p=scratch0.data(); vs.n=0; vs.cap=1<<18; vs.ovf=false;
        sayeVolume(phi0, &vs, &ar0, cfg0);
        double vol=0; for (i32 q=0;q<vs.n;q++) vol += (double)vs.p[q].w;
        if (vol < kSliver || vol > 1.0-kSliver) cnt[1]++;   // S
        for (i32 d=0; d<3; d++) for (i32 sd=0; sd<2; sd++) {
          SayeSet fs; fs.p=scratch0.data(); fs.n=0; fs.cap=1<<18; fs.ovf=false;
          sayeFace(phi0, d, sd, &fs, &ar0, cfg0);
          double fa=0; for (i32 q=0;q<fs.n;q++) fa += (double)fs.p[q].w;
          if (fa > 1e-14 && fa < kNeedle) cnt[2]++;         // N
          if (fs.n > 400) cnt[3]++;                         // X
        }
      }
    };

    i32 c0[4]; audit(0.0, c0);
    const i32 bad0 = c0[0]+c0[1]+c0[2];      // X is COST, not correctness
    if (bad0 > 0) {
      printf("cut    : DEGENERATE GEOMETRY -- %d tangency, %d sliver, "
             "%d needle-face  (+%d face(s) heavily subdivided: cost, not error)\n",
             c0[0], c0[1], c0[2], c0[3]);
      const double ladder[] = {1e-5, 1e-4, 1e-3, 1e-2};
      double bestEps = 0; i32 bestBad = bad0;
      for (double f : ladder) {
        i32 c[4]; const double eps = f*hMin;
        audit(eps, c);
        const i32 bad = c[0]+c[1]+c[2];
        printf("       : shift %.2e (%.0e of a cell) -> "
               "%d tangency, %d sliver, %d needle\n", eps, f, c[0], c[1], c[2]);
        if (bad < bestBad) { bestBad = bad; bestEps = eps; }
        if (bad == 0) break;
      }
      cutEps = (real)bestEps;
      if (bestBad == 0)
        printf("       : adopting shift %.3e -- geometry is now non-degenerate\n",
               (double)cutEps);
      else
        printf("       : WARNING no shift cleared it; best is %.3e with %d "
               "signature(s) left.\n"
               "       : that is under-resolution, not degeneracy -- REFINE the "
               "wall band.\n", (double)cutEps, bestBad);
      printf("       : CUT_NOGEOMGUARD=1 reproduces the unshifted build\n");
    }
  }

  const i32 N = dgOrder, nb = CutBasis::count(N);
  cutNb = nb;

  double w[NNODE], xi[NNODE];
  dgGetHostOps(w, xi, gauss);            // solution-point coordinates on [-1,1]

  // ---- pass 1: which blocks are cut, and build their operators ------------
  std::vector<SayeNode> arena(1<<21), scratch(1<<18);
  SayeArena ar; ar.buf = arena.data(); ar.cap = 1<<21; ar.top = 0;
  SayeCfg cfg = SayeCfg::def();
  cfg.ng = getenv("CUT_NG") ? atoi(getenv("CUT_NG")) : 10;   // 5 is NOT enough -- measured
  // subdivision cap: see the long note on SayeCfg::def().  The default (10) is
  // what makes the discrete divergence theorem close at P3; this override is
  // for reproducing the old geometry, not for tuning.
  if (getenv("CUT_MAXDEPTH")) cfg.maxDepth = atoi(getenv("CUT_MAXDEPTH"));

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
      double f = -dgHostIbPhi(*this, X, Y) - cutEps;   // Saye active = phi<0 = FLUID
                                              // cutEps: see the tangency guard
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
        if (!getenv("CUT_NOFACEOVR")) ovPtr[fc] = &ovStore[fc];
      }
    }
    CutElemOps E;
    // ffXi/ffW: the DG solution rule.  A fully-fluid face then carries exactly
    // the rule dgRhsCutKernel integrates it with (the neighbour's tensor face
    // nodes), so the GCL is fitted against the runtime rule -- see the note in
    // CutElem.h.
    // pseudo2D: build the rules as (2-D slice) x (Gauss in z) -- exactly
    // z-symmetric, and cheaper than the 3-D recursion (see SayeQuad.h)
    const i32 p2dNz = (pseudo2D && !getenv("CUT_NO2DRULE")) ? NNODE+1 : 0;
    if (!cutElemBuild(phi, N, E, ar, cfg, scratch, 1e-6, ovPtr, xi, w, NNODE, p2dNz)) continue;
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
  cudaMallocManaged(&cutDbg, 2*sizeof(i32));
  cutDbg[0] = cutDbg[1] = 0;
  cudaMallocManaged(&cutBlk,  nCutElem*sizeof(i32));
  cudaMallocManaged(&cutNbOf, nCutElem*sizeof(i32));
  cudaMallocManaged(&cutNbLo, nCutElem*sizeof(i32));
  cudaMallocManaged(&cutM11,  (size_t)nCutElem*CUT_NBMAX_H*CUT_NBMAX_H*sizeof(real));
  memset(cutM11, 0, (size_t)nCutElem*CUT_NBMAX_H*CUT_NBMAX_H*sizeof(real));
  cudaMallocManaged(&cutCen,  (size_t)nCutElem*4*sizeof(real));
  cudaMallocManaged(&cutQual, (size_t)nCutElem*sizeof(real));
  cudaMallocManaged(&cutWallN, (size_t)nCutElem*2*sizeof(real));
  cudaMallocManaged(&cutLc, (size_t)nCutElem*CUT_NBMAX_H*CUT_NBMAX_H*sizeof(real));
  memset(cutLc, 0, (size_t)nCutElem*CUT_NBMAX_H*CUT_NBMAX_H*sizeof(real));
  cudaMallocManaged(&blkCut,  nBlocksMax*sizeof(i32));
  for (i32 b = 0; b < nBlocksMax; b++) blkCut[b] = -1;

  size_t nVolT = 0, nWalT = 0, nFacT = 0;
  for (i32 c = 0; c < nCutElem; c++) {
    const CutElemOps &E = ops[c];
    cutBlk[c] = blkOf[c];  blkCut[blkOf[c]] = c;
    cutNbOf[c] = E.B.nb;
    cutNbLo[c] = E.nbLo;   // sensor threshold index; M11inv no longer needed
                           // (orthonormal energies are plain coefficient sums)
    cutQual[c] = (real)E.bndIncons;
    { double nx=0, ny=0;
      for (const SayeNode &sn : E.wall) { nx += (double)sn.w*(double)sn.n[0];
                                          ny += (double)sn.w*(double)sn.n[1]; }
      double nm = sqrt(nx*nx+ny*ny);
      cutWallN[2*c]   = (real)(nm>1e-14 ? nx/nm : 1.0);
      cutWallN[2*c+1] = (real)(nm>1e-14 ? ny/nm : 0.0); }
    cutCen[4*c+0]=(real)E.B.c[0]; cutCen[4*c+1]=(real)E.B.c[1];
    cutCen[4*c+2]=(real)E.B.c[2]; cutCen[4*c+3]=(real)E.B.s;
    for (const SayeNode &s : E.vol)  cutVolP[nVolT++] = s;
    for (const SayeNode &s : E.wall) cutWalP[nWalT++] = s;
    for (i32 f = 0; f < 6; f++) {
      double a = 0;
      for (const SayeNode &s : E.face[f]) { cutFacP[nFacT++] = s; a += (double)s.w; }
      cutFacA[6*c+f] = (real)a;
    }
    // Cholesky factor L (lower triangle of Mchol): the kernel orthonormalizes
    // per point via a forward solve, and the mass becomes exactly I.
    const i32 nbE = E.B.nb;               // may be < nb on a degenerate sliver
    for (i32 i = 0; i < nbE; i++)
      for (i32 j = 0; j <= i; j++)
        cutLc[(size_t)c*CUT_NBMAX_H*CUT_NBMAX_H + (size_t)i*CUT_NBMAX_H + j] =
          (real)E.Mchol[(size_t)i*nbE + j];
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
  size_t shm = (5*blockSizeTot + 10*CUT_NBMAX_H + 4)*sizeof(real);
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
  // srdApply reads element states through THIS basis; the quadrature pool and
  // the modal conversion use dgGetHostOps(..., gauss).  Under --gauss the
  // solution points are Gauss-Legendre, so a GLL basis here would read the
  // right numbers through the wrong polynomials -- silently.
  if (gauss) {
    printf("srd    : WARNING --gauss with SRD: the SRD nodal basis is GLL while "
           "the solution points are Gauss-Legendre; results are not trustworthy\n");
  }
  srd->B.init(dgOrder);
  // SMALL-CELL THRESHOLD.  SRD is a filter: it replaces a flagged element's
  // state by a neighbourhood projection, so applying it to a HEALTHY cell costs
  // accuracy and time for a problem that cell does not have.  The default 0.5
  // flags everything under half a background volume, which on case 9 at h=0.25
  // is every cut element in the band (min fluid fraction 0.45).  --srdvol sets
  // it; 0.1 restricts SRD to genuinely small cells.
  { const char *sv = getenv("CUT_SRDVOL");
    if (srdVolFrac > 0) srd->S.volFrac = srdVolFrac;
    else if (sv) srd->S.volFrac = atof(sv); }
  printf("srd    : small-cell threshold volFrac = %.3f of a background cell\n",
         srd->S.volFrac);

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
  {
    // SRD PROJECTION DEGREE.  Taylor, Wilcox & Chan project onto P_N over the
    // merge neighbourhood -- N being the SOLUTION degree -- and degree 0 appears
    // NOWHERE in the paper: their Def 2.2, Eq 26 and the contractivity lemma are
    // all stated for P_N, and their runs are N = 1,2,3,4.  "Preserves polynomial
    // order", which is the property that makes SRD free, is definitionally
    // impossible at degree 0.
    //
    // MEASURED CONSEQUENCE of doing it properly: at degree 0 this scheme is
    // stable, at 1 the solution corrupts (wall Cp min -20), at 2 and 3 it blows
    // up (t=0.56, t=0.28).  The reason is
    // that SRD is a CONTRACTIVE FILTER and the theorem is conditional -- it says
    // an ENERGY-STABLE scheme survives one.  Our cut RHS is a standard weak form
    // with Rusanov/HLLC faces and is not energy stable, so the theorem does not
    // cover it and the filter has nothing to protect.  Degree 0 is then acting
    // as a low-pass stabiliser rather than as state redistribution.
    //
    // THE COST IS NOT SUBTLE: a degree-0 projection replaces every merged cut
    // element by a CONSTANT, three times per step (once per RK stage), so the
    // wall trace is piecewise constant per element and the cut band is
    // first-order no matter what degree the elements carry.  Measured: wall Cp
    // is a staircase with one plateau per cut element.
    // DEFAULT IS THE SOLUTION DEGREE, per the paper.  A degree-0 projection
    // replaces every merged cut element by a CONSTANT three times per step and
    // makes the cut band first-order however high the element degree -- it is
    // not state redistribution, it is a low-pass filter wearing its name.
    const char *se = getenv("CUT_SRDDEG");
    i32 sdeg = se ? atoi(se) : dgOrder;
    if (sdeg > dgOrder) {
      // above the solution degree the nodal write-back truncates the projection
      // at the (p+1)^3 tensor nodes and CONSERVATION IS LOST
      printf("srd    : CUT_SRDDEG=%d exceeds the solution degree %d -- clamped "
             "(the write-back would truncate it and break conservation)\n",
             sdeg, dgOrder);
      sdeg = dgOrder;
    }
    if (sdeg < 0) sdeg = 0;
    srd->S.factor(srd->elems, srd->qpool.data(), sdeg);
    if (sdeg < dgOrder)
      printf("srd    : projection degree %d < solution degree %d -- the cut band "
             "is FIRST-ORDER (see the note in DgCutBuild.cu; CUT_SRDDEG=%d is "
             "the faithful setting and needs an energy-stable RHS)\n",
             sdeg, dgOrder, dgOrder);
  }
  for (i32 k = 0; k < srd->S.nElem; k++) if (!srd->S.trivial[k]) srd->nMerged++;
  const size_t nd = (size_t)srd->elems.size()*blockSizeTot*5;
  srd->u.resize(nd); srd->su.resize(nd);
  double vmin = 1e300;
  for (const SrdElem &E : srd->elems) if (E.vol < vmin) vmin = E.vol;
  double vnb = 1e300;
  for (i32 k = 0; k < srd->S.nElem; k++) {
    if (srd->S.trivial[k]) continue;
    double vv = 0; for (i32 m : srd->S.M[k]) vv += srd->elems[m].vol;
    vnb = fmin(vnb, vv);
  }
  printf("srd    : %zu elements (%d merged), smallest vol %.3e of full %.3e, "
         "smallest MERGED neighbourhood %.3e\n",
         srd->elems.size(), srd->nMerged, vmin,
         srd->elems.empty() ? 0.0 : srd->elems[0].hv(),
         (vnb < 1e299) ? vnb : 0.0);
  if (srd->S.nShort > 0)
    printf("srd    : WARNING %d neighbourhood(s) never reached the volume "
           "target and are used anyway -- SRD did not buy the CFL relief there\n",
           srd->S.nShort);
}

// ---------------------------------------------------------------------------
//  HOST-SIDE ACCESS TO A CUT ELEMENT'S ACTUAL SOLUTION.
//  Re-runs dgRhsCutKernel's own nodal->modal projection (DgSolverKernels.cu:5920)
//  and evaluates the resulting polynomial.  Shared by the field writer and by
//  the image path: both need the cut element's REAL representation, and the
//  coefficients live only in the kernel's shared memory.
// ---------------------------------------------------------------------------
struct CutHostEval {
  DgSolver *g = nullptr;
  i32 c = -1, nb = 0;
  std::vector<double> cmod;
  double xiN[NNODE], wN[NNODE];

  void begin(DgSolver &G, i32 cIdx) {
    g = &G; c = cIdx; nb = G.cutNbOf[cIdx];
    dgGetHostOps(wN, xiN, G.gauss);
    cmod.assign((size_t)nb*5, 0.0);
    const i32 b = G.cutBlk[cIdx];
    if (G.cutModal) {                 // the coefficients ARE the state
      for (i32 m = 0; m < nb; m++)
        for (i32 q = 0; q < 5; q++)
          cmod[(size_t)m*5+q] = (double)G.getField(D_RHO+q)[(size_t)b*blockSizeTot + m];
      return;
    }
    std::vector<double> psi(CUT_NBMAX_H), Lx(NNODE), Ly(NNODE), Lz(NNODE);
    for (i32 q = G.cutVolOff[cIdx]; q < G.cutVolOff[cIdx+1]; q++) {
      const SayeNode &s = G.cutVolP[q];
      const double xr[3] = {(double)s.x[0], (double)s.x[1], (double)s.x[2]};
      basis(xr, psi.data());
      lag(2.0*xr[0]-1.0, Lx.data()); lag(2.0*xr[1]-1.0, Ly.data()); lag(2.0*xr[2]-1.0, Lz.data());
      for (i32 fq = 0; fq < 5; fq++) {
        double uq = 0.0;
        for (i32 k = 0; k < NNODE; k++) for (i32 j = 0; j < NNODE; j++) for (i32 i = 0; i < NNODE; i++)
          uq += Lx[i]*Ly[j]*Lz[k]
              * (double)G.getField(D_RHO+fq)[(size_t)b*blockSizeTot + i + NNODE*(j + NNODE*k)];
        for (i32 m = 0; m < nb; m++) cmod[(size_t)m*5+fq] += (double)s.w*psi[m]*uq;
      }
    }
  }
  void lag(double x, double *L) const {
    for (i32 a = 0; a < NNODE; a++) { double v = 1.0;
      for (i32 m = 0; m < NNODE; m++) if (m != a) v *= (x - xiN[m])/(xiN[a] - xiN[m]);
      L[a] = v; }
  }
  // orthonormal cut basis, host twin of dgCutPsiO (DgSolverKernels.cu:5841)
  void basis(const double xr[3], double *psi) const {
    const real *cen = g->cutCen + 4*c;
    double u[3];
    for (i32 d = 0; d < 3; d++) u[d] = (xr[d] - (double)cen[d])/(double)cen[3];
    i32 m = 0;
    for (i32 deg = 0; deg <= dgOrder && m < nb; deg++)
    for (i32 i = deg; i >= 0 && m < nb; i--)
    for (i32 j = deg-i; j >= 0 && m < nb; j--) {
      const i32 e[3] = { i, j, deg-i-j };
      double v = 1.0;
      for (i32 d = 0; d < 3; d++) for (i32 a = 0; a < e[d]; a++) v *= u[d];
      psi[m++] = v;
    }
    const real *Lc = g->cutLc + (size_t)c*CUT_NBMAX_H*CUT_NBMAX_H;
    for (i32 i = 0; i < nb; i++) { double t = psi[i];
      for (i32 j = 0; j < i; j++) t -= (double)Lc[(size_t)i*CUT_NBMAX_H+j]*psi[j];
      psi[i] = t/(double)Lc[(size_t)i*CUT_NBMAX_H+i]; }
  }
  // conserved variables of the element's own polynomial at a reference point
  void consAt(const double xr[3], double U[5]) const {
    std::vector<double> psi(nb);
    basis(xr, psi.data());
    for (i32 q = 0; q < 5; q++) { double t = 0;
      for (i32 m = 0; m < nb; m++) t += cmod[(size_t)m*5+q]*psi[m];
      U[q] = t; }
  }
};

// ---------------------------------------------------------------------------
//  Flatten the SRD operator into device arrays.  Called once, after buildSrd.
//  Everything here is static for the life of the run (the wall band is frozen
//  and the block sort is pinned), so this is pure setup.
// ---------------------------------------------------------------------------
void DgSolver::buildSrdDevice(void) {
  if (!srd || srd->nMerged == 0) return;
  if (getenv("CUT_SRDHOST")) { srdOnDev = 0; return; }
  const i32 nE = (i32)srd->elems.size();
  const i32 nb = srd->S.nb;
  srdNE = nE; srdNb = nb; srdDeg = srd->S.N;

  auto allocI = [&](i32 **p2, size_t n) { cudaMallocManaged(p2, n*sizeof(i32)); };
  auto allocD = [&](double **p2, size_t n) { cudaMallocManaged(p2, n*sizeof(double)); };
  allocI(&srdBlk, nE);  allocI(&srdQOff, nE+1); allocI(&srdCcnt, nE);
  allocI(&srdMOff, nE+1); allocI(&srdCOff, nE+1);
  allocD(&srdX0, 3*nE); allocD(&srdH, 3*nE); allocD(&srdBas, 4*nE);
  allocD(&srdChol, (size_t)nE*nb*nb);
  allocD(&srdCoef, (size_t)nE*nb*5);
  allocD(&srdU,   (size_t)nE*blockSizeTot*5);
  cudaMallocManaged(&srdTriv, nE*sizeof(char));
  cudaMallocManaged(&srdQ, (srd->qpool.size()?srd->qpool.size():1)*sizeof(SayeNode));
  memcpy(srdQ, srd->qpool.data(), srd->qpool.size()*sizeof(SayeNode));

  size_t mTot = 0, cTot = 0;
  for (i32 k = 0; k < nE; k++) { mTot += srd->S.M[k].size(); cTot += srd->S.C[k].size(); }
  allocI(&srdM, mTot?mTot:1); allocI(&srdC, cTot?cTot:1);

  srdMOff[0] = srdCOff[0] = srdQOff[0] = 0;
  i32 mAt = 0, cAt = 0;
  for (i32 k = 0; k < nE; k++) {
    const SrdElem &E = srd->elems[k];
    srdBlk[k] = srd->blk[k];
    for (i32 d = 0; d < 3; d++) { srdX0[3*k+d] = E.x0[d]; srdH[3*k+d] = E.h[d]; }
    srdQOff[k+1] = srdQOff[k] + E.qN;
    srdCcnt[k] = srd->S.Ccnt[k];
    srdTriv[k] = srd->S.trivial[k];
    const SrdBasis &B = srd->S.basis[k];
    srdBas[4*k+0]=B.c[0]; srdBas[4*k+1]=B.c[1]; srdBas[4*k+2]=B.c[2]; srdBas[4*k+3]=B.s;
    if (!srd->S.trivial[k] && (i32)srd->S.chol[k].size() >= nb*nb)
      memcpy(srdChol + (size_t)k*nb*nb, srd->S.chol[k].data(), (size_t)nb*nb*sizeof(double));
    for (i32 j : srd->S.M[k]) srdM[mAt++] = j;
    for (i32 j : srd->S.C[k]) srdC[cAt++] = j;
    srdMOff[k+1] = mAt; srdCOff[k+1] = cAt;
  }
  // qOff in the flattened pool must match the elements' own slices
  for (i32 k = 0; k < nE; k++)
    if (srdQOff[k] != srd->elems[k].qOff) {
      printf("srd    : device qOff mismatch at %d (%d vs %d) -- staying on the host\n",
             k, srdQOff[k], srd->elems[k].qOff);
      srdOnDev = 0; return;
    }
  srdOnDev = 1;
  printf("srd    : DEVICE apply ON -- %d elements, %d modes, pools M %zu / C %zu, %.2f MB\n",
         nE, nb, mTot, cTot,
         ((double)nE*nb*nb + (double)nE*nb*5 + (double)nE*blockSizeTot*5)*sizeof(double)/1048576.0);
}

// three kernels, no host round trip (see the note in DgSolver.cuh)
void DgSolver::applySrdDevice(void) {
  const size_t shProj = (size_t)srdNb*5*sizeof(double);
  const size_t shScat = ((size_t)blockSizeTot*5 + (size_t)CUT_NBMAX_H*5)*sizeof(double);
  dgSrdGatherKernel <<<srdNE, 64>>>(*this);
  dgSrdProjectKernel<<<srdNE, 128, shProj>>>(*this);
  dgSrdScatterKernel<<<srdNE, 64,  shScat>>>(*this);
}

void DgSolver::applySrd(void) {
  if (!srd || srd->nMerged == 0) return;
  if (getenv("CUT_NOSRD")) return;
  if (srdOnDev) { applySrdDevice(); return; }
  cudaDeviceSynchronize();
  const i32 nE = (i32)srd->elems.size();
  // ---- TEMP conservation probe (SRD_CONS=1 / FRD_CONS=1) -------------------
  // Totals over the SRD ELEMENT SET measured with the SOLVER'S OWN functional:
  // cut elements over their Saye volume rule (same as dgCutConserved), uncut
  // over the tensor GLL rule (same as dgTotalConserved).  base = D_RHO gives
  // the state total, base = D_RHS gives the total RATE.
  auto srdSetTot = [&](i32 base, double T[3]) {
    T[0]=T[1]=T[2]=0;
    double wq[NNODE], xq[NNODE]; dgGetHostOps(wq, xq, gauss);
    std::vector<double> psi(CUT_NBMAX_H);
    for (i32 e2 = 0; e2 < nE; e2++) {
      const i32 b2 = srd->blk[e2];
      i32 lvl,ib2,jb2,kb2; decode(bLocList[b2], lvl, ib2, jb2, kb2);
      double hh[3]; hostElemSizeLocal(*this, lvl, hh);
      const i32 c2 = (cutModal && blkCut) ? blkCut[b2] : -1;
      if (c2 >= 0) {
        const double jac = hh[0]*hh[1]*hh[2];
        CutHostEval ev2; ev2.begin(*this, c2);          // basis()/nb only
        const i32 nb2 = cutNbOf[c2];
        for (i32 g2 = cutVolOff[c2]; g2 < cutVolOff[c2+1]; g2++) {
          const SayeNode &sn = cutVolP[g2];
          const double xr[3] = {(double)sn.x[0],(double)sn.x[1],(double)sn.x[2]};
          ev2.basis(xr, psi.data());
          const double wv = (double)sn.w*jac;
          for (i32 f2 = 0; f2 < 3; f2++) {
            const i32 fq = (f2==0)?0:((f2==1)?1:4);
            double v = 0;
            for (i32 m = 0; m < nb2; m++)
              v += (double)getField(base+fq)[(size_t)b2*blockSizeTot+m]*psi[m];
            T[f2] += wv*v;
          }
        }
      } else {
        for (i32 nd = 0; nd < blockSizeTot; nd++) {
          i32 i2=nd%NNODE, j2=(nd/NNODE)%NNODE, k2=nd/(NNODE*NNODE);
          double wv = (0.5*hh[0]*wq[i2])*(0.5*hh[1]*wq[j2])*(0.5*hh[2]*wq[k2]);
          T[0]+=wv*(double)getField(base+0)[(size_t)b2*blockSizeTot+nd];
          T[1]+=wv*(double)getField(base+1)[(size_t)b2*blockSizeTot+nd];
          T[2]+=wv*(double)getField(base+4)[(size_t)b2*blockSizeTot+nd];
        }
      }
    }
  };
  const bool consProbe = getenv("SRD_CONS") != nullptr;
  double T0[3] = {0,0,0};
  if (consProbe) srdSetTot(D_RHO, T0);
  // MODAL cut blocks hold coefficients, and srdApply consumes NODAL values on
  // the tensor grid (it evaluates them with the tensor Lagrange basis at each
  // element's quadrature points).  Convert at the BOUNDARY rather than teaching
  // the operator about cut bases -- and the conversion is EXACT, not an
  // approximation: a total-degree-N polynomial with N <= p lies in the tensor
  // Q^p space, so interpolation at the (p+1)^3 Lobatto nodes reproduces it
  // identically.  The round trip therefore costs round-off, not accuracy.
  double wN[NNODE], xiN[NNODE];
  dgGetHostOps(wN, xiN, gauss);
  for (i32 e = 0; e < nE; e++) {
    i32 b = srd->blk[e];
    const i32 c = (cutModal && blkCut) ? blkCut[b] : -1;
    if (c >= 0) {
      CutHostEval ev; ev.begin(*this, c);
      for (i32 nd = 0; nd < blockSizeTot; nd++) {
        const i32 i2 = nd%NNODE, j2 = (nd/NNODE)%NNODE, k2 = nd/(NNODE*NNODE);
        const double xr[3] = { 0.5*(xiN[i2]+1.0), 0.5*(xiN[j2]+1.0), 0.5*(xiN[k2]+1.0) };
        double U[5]; ev.consAt(xr, U);
        for (i32 q = 0; q < 5; q++) srd->u[((size_t)e*blockSizeTot+nd)*5+q] = U[q];
      }
      continue;
    }
    for (i32 nd = 0; nd < blockSizeTot; nd++)
      for (i32 q = 0; q < 5; q++)
        srd->u[((size_t)e*blockSizeTot+nd)*5+q] =
          (double)getField(D_RHO+q)[(size_t)b*blockSizeTot+nd];
  }
  // SRD_RTCHECK=1: the boundary conversion's IDENTITY test.  Gathering a cut
  // element to the tensor grid and projecting straight back over the fluid rule
  // is algebraically the identity (a total-degree-N polynomial with N <= p lies
  // in Q^p, so tensor interpolation reproduces it).  In floating point it is
  // not: the tensor nodes sit INSIDE THE SOLID, where the element's own basis
  // is an extrapolation -- CutElem.h records max|psi~| = 54.3 over the fluid
  // rule against 2.44e+04 at the tensor nodes on the case-9 wedge.  The round
  // trip is therefore a ~450x cancellation, run three times per step.  This
  // measures what it actually costs, with NO redistribution in between.
  if (getenv("SRD_RTCHECK")) {
    static i32 nRt = 0;
    double worst = 0, worstIn = 0; i32 worstE = -1;
    for (i32 e = 0; e < nE; e++) {
      const i32 b = srd->blk[e];
      const i32 c = (cutModal && blkCut) ? blkCut[b] : -1;
      if (c < 0) continue;
      const i32 nb = cutNbOf[c];
      CutHostEval ev; ev.begin(*this, c);
      std::vector<double> cm((size_t)nb*5, 0.0), psi(nb), Lx(NNODE), Ly(NNODE), Lz(NNODE);
      for (i32 g = cutVolOff[c]; g < cutVolOff[c+1]; g++) {
        const SayeNode &sn = cutVolP[g];
        const double xr[3] = {(double)sn.x[0], (double)sn.x[1], (double)sn.x[2]};
        ev.basis(xr, psi.data());
        ev.lag(2.0*xr[0]-1.0, Lx.data());
        ev.lag(2.0*xr[1]-1.0, Ly.data());
        ev.lag(2.0*xr[2]-1.0, Lz.data());
        for (i32 q = 0; q < 5; q++) {
          double uq = 0;
          for (i32 k2 = 0; k2 < NNODE; k2++) for (i32 j2 = 0; j2 < NNODE; j2++)
          for (i32 i2 = 0; i2 < NNODE; i2++)
            uq += Lx[i2]*Ly[j2]*Lz[k2]
                * srd->u[((size_t)e*blockSizeTot + i2 + NNODE*(j2 + NNODE*k2))*5+q];
          for (i32 m = 0; m < nb; m++) cm[(size_t)m*5+q] += (double)sn.w*psi[m]*uq;
        }
      }
      double d = 0, r = 0;
      for (i32 m = 0; m < nb; m++) for (i32 q = 0; q < 5; q++) {
        const double o = ev.cmod[(size_t)m*5+q];
        d = fmax(d, fabs(cm[(size_t)m*5+q] - o));
        r = fmax(r, fabs(o));
      }
      if (d/fmax(r,1e-300) > worst) { worst = d/fmax(r,1e-300); worstIn = r; worstE = c; }
    }
    if (nRt < 8 || (nRt % 200) == 0)
      printf("[srdrt] call %d  worst round-trip rel drift %.3e on cut elem %d (|c|max %.3e)\n",
             nRt, worst, worstE, worstIn);
    nRt++;
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
    const i32 c = (cutModal && blkCut) ? blkCut[b] : -1;
    if (c >= 0) {
      // project the redistributed state back onto the element's own basis --
      // the same weighted sum dgRhsCutKernel used to do, over the fluid rule
      const i32 nb = cutNbOf[c];
      CutHostEval ev; ev.begin(*this, c);          // for lag()/basis() only
      std::vector<double> cm((size_t)nb*5, 0.0), psi(nb), Lx(NNODE), Ly(NNODE), Lz(NNODE);
      for (i32 g = cutVolOff[c]; g < cutVolOff[c+1]; g++) {
        const SayeNode &sn = cutVolP[g];
        const double xr[3] = {(double)sn.x[0], (double)sn.x[1], (double)sn.x[2]};
        ev.basis(xr, psi.data());
        ev.lag(2.0*xr[0]-1.0, Lx.data());
        ev.lag(2.0*xr[1]-1.0, Ly.data());
        ev.lag(2.0*xr[2]-1.0, Lz.data());
        for (i32 q = 0; q < 5; q++) {
          double uq = 0;
          for (i32 k2 = 0; k2 < NNODE; k2++) for (i32 j2 = 0; j2 < NNODE; j2++)
          for (i32 i2 = 0; i2 < NNODE; i2++)
            uq += Lx[i2]*Ly[j2]*Lz[k2]
                * srd->su[((size_t)e*blockSizeTot + i2 + NNODE*(j2 + NNODE*k2))*5+q];
          for (i32 m = 0; m < nb; m++) cm[(size_t)m*5+q] += (double)sn.w*psi[m]*uq;
        }
      }
      for (i32 nd = 0; nd < blockSizeTot; nd++)
        for (i32 q = 0; q < 5; q++)
          getField(D_RHO+q)[(size_t)b*blockSizeTot+nd] =
              (nd < nb) ? (real)cm[(size_t)nd*5+q] : (real)0;
      continue;
    }
    for (i32 nd = 0; nd < blockSizeTot; nd++)
      for (i32 q = 0; q < 5; q++)
        getField(D_RHO+q)[(size_t)b*blockSizeTot+nd] =
          (real)srd->su[((size_t)e*blockSizeTot+nd)*5+q];
  }
  if (consProbe) {
    double T1[3]; srdSetTot(D_RHO, T1);
    static i32 nC = 0; static double acc[3] = {0,0,0};
    for (i32 i2 = 0; i2 < 3; i2++) acc[i2] += T1[i2]-T0[i2];
    if (nC < 40 || (nC%30)==0)
      printf("[srdcons] call %4d  setMass %.15e  dMass %+.4e (cum %+.4e)  "
             "dMomx %+.4e  dE %+.4e\n", nC, T0[0], T1[0]-T0[0], acc[0],
             T1[1]-T0[1], T1[2]-T0[2]);
    nC++;
  }
}

// ---------------------------------------------------------------------------
//  FLUX REDISTRIBUTION (Chern & Colella).  The P0 wedge's pathology is not its
//  STATE but its UPDATE RATE: a full-area face over an 0.087 volume is an 11x
//  amplification, and the resulting stage-frequency dent flicker is what pumps
//  a P^N neighbour at supersonic speeds (measured: byte-identical blowup
//  whether that neighbour is cut or Cartesian -- the wedge drives it).
//
//  Fix at the source: the small member takes the MERGED rate,
//      r_merged = sum_j vol_j r_j / vol_k   over its neighbourhood,
//  and the excess flux  delta = vol_small (r_own - r_merged)  is deposited
//  into the partners, volume-weighted.  Total change:
//      vol_s (r_merged - r_own) + delta = 0   -- exactly conservative.
//  First-order at the wedge; it is P0 by the thin-cell rule anyway.
// ---------------------------------------------------------------------------
void DgSolver::redistributeFlux(void) {
  if (!srd || srd->nMerged == 0) return;
  if (getenv("CUT_NOFRD")) return;
  cudaDeviceSynchronize();
  // ---- TEMP probe (FRD_CONS=1): total mass RATE over the SRD element set ---
  const i32 nEf = (i32)srd->elems.size();
  auto frdSetRate = [&](double T[3]) {
    T[0]=T[1]=T[2]=0;
    double wq[NNODE], xq[NNODE]; dgGetHostOps(wq, xq, gauss);
    std::vector<double> psi(CUT_NBMAX_H);
    for (i32 e2 = 0; e2 < nEf; e2++) {
      const i32 b2 = srd->blk[e2];
      i32 lvl,ib2,jb2,kb2; decode(bLocList[b2], lvl, ib2, jb2, kb2);
      double hh[3]; hostElemSizeLocal(*this, lvl, hh);
      const i32 c2 = (cutModal && blkCut) ? blkCut[b2] : -1;
      if (c2 >= 0) {
        const double jac = hh[0]*hh[1]*hh[2];
        CutHostEval ev2; ev2.begin(*this, c2);
        const i32 nb2 = cutNbOf[c2];
        for (i32 g2 = cutVolOff[c2]; g2 < cutVolOff[c2+1]; g2++) {
          const SayeNode &sn = cutVolP[g2];
          const double xr[3] = {(double)sn.x[0],(double)sn.x[1],(double)sn.x[2]};
          ev2.basis(xr, psi.data());
          const double wv = (double)sn.w*jac;
          for (i32 f2 = 0; f2 < 3; f2++) {
            const i32 fq = (f2==0)?0:((f2==1)?1:4);
            double v = 0;
            for (i32 m = 0; m < nb2; m++)
              v += (double)getField(D_RHS+fq)[(size_t)b2*blockSizeTot+m]*psi[m];
            T[f2] += wv*v;
          }
        }
      } else {
        for (i32 nd = 0; nd < blockSizeTot; nd++) {
          i32 i2=nd%NNODE, j2=(nd/NNODE)%NNODE, k2=nd/(NNODE*NNODE);
          double wv = (0.5*hh[0]*wq[i2])*(0.5*hh[1]*wq[j2])*(0.5*hh[2]*wq[k2]);
          T[0]+=wv*(double)getField(D_RHS+0)[(size_t)b2*blockSizeTot+nd];
          T[1]+=wv*(double)getField(D_RHS+1)[(size_t)b2*blockSizeTot+nd];
          T[2]+=wv*(double)getField(D_RHS+4)[(size_t)b2*blockSizeTot+nd];
        }
      }
    }
  };
  const bool frdProbe = getenv("FRD_CONS") != nullptr;
  double R0[3] = {0,0,0};
  if (frdProbe) frdSetRate(R0);
  const double vFull = srd->elems.empty() ? 1.0 : srd->elems[0].hv();
  for (i32 k = 0; k < srd->S.nElem; k++) {
    if (srd->S.trivial[k]) continue;
    // the flagged (small) member is k itself by construction
    const i32 bS = srd->blk[k];
    if (blkCut[bS] < 0) continue;              // only small CUT members
    if (srd->elems[k].vol >= srd->S.volFrac*vFull) continue;
    // MODAL-AWARE MEAN AND SHIFT.  Under --cutmodal a cut block's D_RHS slots
    // hold the RHS's MODAL COEFFICIENTS in the element's own orthonormal basis,
    // not nodal values.  The previous code read slot 0 as "the P0 nodal
    // constant" -- it is c~_0, which is the mean scaled by 1/psi~_0 = L00 -- and
    // then wrote the merged rate into ALL blockSizeTot slots, setting every mode
    // c~_m to the same number and filling the m >= nb slots the RHS kernel had
    // zeroed.  That does not damp the small element's rate; it REPLACES its
    // whole residual polynomial with a spurious one, three times per step, on
    // exactly the 0.088-volume wedges.  MEASURED on case 9, single level, M=0.3,
    // solid wall, SRD degree 3: with this path disabled (CUT_NOFRD=1) the run
    // completes to t=1.0 and the interior mass source at t=0.2 falls 1.15e-02
    // -> 2.43e-04; with it enabled the run dies at t=0.247.
    //
    // psi~_0 is the constant 1/L00, so
    //     mean(R) = c~_0 * psi~_0 = c~_0 / L00,      c~_0 = mean(R) * L00
    // and shifting the mean by `add` while PRESERVING the higher modes is
    //     c~_0 += add * L00
    // -- which is the order-preserving statement of "the small member takes the
    // merged rate".  Flattening to a constant would be a P0 cut cell.
    auto cutL00 = [&](i32 c) {
      return (double)cutLc[(size_t)c*CUT_NBMAX_H*CUT_NBMAX_H];
    };
    const double L00S = cutL00(blkCut[bS]);
    double rS[5], rM[5] = {0,0,0,0,0};
    for (i32 q = 0; q < 5; q++)
      rS[q] = (double)getField(D_RHS+q)[(size_t)bS*blockSizeTot]/L00S;
    double volK = 0, volP = 0;
    double w[NNODE], xi[NNODE]; dgGetHostOps(w, xi, gauss);
    for (i32 j : srd->S.M[k]) {
      const double vj = srd->elems[j].vol; volK += vj;
      if (j != k) volP += vj;
      const i32 bj = srd->blk[j];
      if (j == k) { for (i32 q = 0; q < 5; q++) rM[q] += vj*rS[q]; continue; }
      const i32 cj = (cutModal && blkCut) ? blkCut[bj] : -1;
      if (cj >= 0) {                      // a CUT partner is modal too
        const double L00j = cutL00(cj);
        for (i32 q = 0; q < 5; q++)
          rM[q] += vj*(double)getField(D_RHS+q)[(size_t)bj*blockSizeTot]/L00j;
        continue;
      }
      for (i32 q = 0; q < 5; q++) {
        double m = 0, ws = 0;
        for (i32 nd = 0; nd < blockSizeTot; nd++) {
          i32 i2 = nd % NNODE, j2 = (nd/NNODE) % NNODE, k2 = nd/(NNODE*NNODE);
          double wn = w[i2]*w[j2]*w[k2];
          m += wn*(double)getField(D_RHS+q)[(size_t)bj*blockSizeTot+nd]; ws += wn;
        }
        rM[q] += vj*(m/ws);
      }
    }
    if (volK <= 0 || volP <= 0) continue;
    for (i32 q = 0; q < 5; q++) rM[q] /= volK;
    // small member -> merged rate; excess -> partners, volume-weighted.
    // Conservation: the small member's mass rate moves by vol_k*(rM - rS) and
    // the partners take sum_j vol_j*(dlt/volP) = dlt = vol_k*(rS - rM).
    double dlt[5];
    for (i32 q = 0; q < 5; q++) dlt[q] = srd->elems[k].vol*(rS[q]-rM[q]);
    for (i32 q = 0; q < 5; q++)                       // shift the MEAN only
      getField(D_RHS+q)[(size_t)bS*blockSizeTot] = (real)(rM[q]*L00S);
    for (i32 j : srd->S.M[k]) {
      if (j == k) continue;
      const i32 bj = srd->blk[j];
      const i32 cj = (cutModal && blkCut) ? blkCut[bj] : -1;
      for (i32 q = 0; q < 5; q++) {
        const double add = dlt[q]/volP;
        if (cj >= 0) {                    // modal: shift the mean coefficient
          getField(D_RHS+q)[(size_t)bj*blockSizeTot] += (real)(add*cutL00(cj));
        } else {
          for (i32 nd = 0; nd < blockSizeTot; nd++)
            getField(D_RHS+q)[(size_t)bj*blockSizeTot+nd] += (real)add;
        }
      }
    }
  }
  if (frdProbe) {
    double R1[3]; frdSetRate(R1);
    static i32 nF = 0;
    if (nF < 40 || (nF%30)==0)
      printf("[frdcons] call %4d  rate0 %+.6e -> %+.6e   dRate %+.6e\n",
             nF, R0[0], R1[0], R1[0]-R0[0]);
    nF++;
  }
}

// ===========================================================================
//  CHARACTERISTIC BARTH-JESPERSEN LIMITER for cut elements
//  (Giuliani, SIAM J. Sci. Comput. 44 (2022): the working shock recipe for
//  SRD-stabilized cut-cell DG, demonstrated through a Mach-10 DMR with volume
//  fractions of 4e-6.)
//
//  Two of his findings drive the design:
//    * limiting in CONSERVED variables "produced unsatisfactory and
//      oscillatory results" -- the transform to CHARACTERISTIC variables of
//      the flux Jacobian is load-bearing;
//    * for cut cells the eigendirection is taken PARALLEL TO THE BOUNDARY
//      (the wall tangent), not axis-aligned.
//
//  Form: nodal deviation scaling.  With cell mean Ubar and the p-degree nodal
//  polynomial U(x), evaluate U at the four lateral NEIGHBOUR CENTROIDS
//  (reference points outside [0,1] -- extrapolation is exactly what BJ tests),
//  transform deviations to characteristic variables, and compute per-field
//  phi in [0,1] so the extrapolated values lie within the range spanned by the
//  neighbours' mean deviations.  Then U := Ubar + R diag(phi) L (U - Ubar).
//  P0 elements are untouched (no deviation); the state's conservative mean is
//  exactly preserved.
// ===========================================================================

// right/left eigenvector matrices of dF.n/dU for the gamma-law Euler system,
// normal n (2-D flow in x-y; z velocity rides along as a passive component).
static void dgEulerEigH(const double W[5], const double n[2],
                        double R[5][5], double L[5][5]) {
  const double g = (double)dgGam;
  double rho = fmax(W[0], 1e-12), u = W[1], v = W[2], w = W[3], p = fmax(W[4], 1e-12);
  double c = sqrt(g*p/rho), c2 = c*c;
  double q2 = u*u+v*v+w*w, H = c2/(g-1.0) + 0.5*q2;
  double un = u*n[0]+v*n[1], ut = -u*n[1]+v*n[0];
  // R columns: [un-c, entropy, shear-t, shear-z, un+c]
  double Rc[5][5] = {
    {1,            1,        0,     0, 1           },
    {u-c*n[0],     u,       -n[1],  0, u+c*n[0]    },
    {v-c*n[1],     v,        n[0],  0, v+c*n[1]    },
    {w,            w,        0,     1, w           },
    {H-c*un,       0.5*q2,   ut,    w, H+c*un      }};
  for (int i=0;i<5;i++) for (int j=0;j<5;j++) R[i][j]=Rc[i][j];
  double gm = g-1.0;
  double L0[5][5] = {
    { 0.5*(gm*0.5*q2/c2 + un/c), -0.5*(gm*u/c2 + n[0]/c), -0.5*(gm*v/c2 + n[1]/c), -0.5*gm*w/c2, 0.5*gm/c2 },
    { 1.0-gm*0.5*q2/c2,           gm*u/c2,                 gm*v/c2,                 gm*w/c2,     -gm/c2     },
    { -ut,                        -n[1],                    n[0],                    0,            0         },
    { -w,                          0,                       0,                       1,            0         },
    { 0.5*(gm*0.5*q2/c2 - un/c), -0.5*(gm*u/c2 - n[0]/c), -0.5*(gm*v/c2 - n[1]/c), -0.5*gm*w/c2, 0.5*gm/c2 }};
  for (int i=0;i<5;i++) for (int j=0;j<5;j++) L[i][j]=L0[i][j];
}

void DgSolver::applyCutLimiter(void) {
  // same reason as applySrd: this reads the cell mean off the tensor nodes
  if (cutModal) return;
  if (!srd || !cutOn || nCutElem == 0) return;
  if (getenv("CUT_NOBJ")) return;
  cudaDeviceSynchronize();
  double w[NNODE], xi[NNODE]; dgGetHostOps(w, xi, gauss);
  // 1-D Lagrange values at an arbitrary reference point (extrapolation allowed)
  auto lag1 = [&](double x, double *Lv){
    for (i32 a = 0; a < NNODE; a++) {
      double t = 1;
      for (i32 b2 = 0; b2 < NNODE; b2++) if (b2 != a) t *= (x - xi[b2])/(xi[a] - xi[b2]);
      Lv[a] = t;
    } };
  const double ctr[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};   // lateral neighbour centres
  const i32 faceSlot2[4] = {0+3*1+9*1, 2+3*1+9*1, 1+3*0+9*1, 1+3*2+9*1};

  for (i32 c = 0; c < nCutElem; c++) {
    const i32 b = cutBlk[c];
    if (cutNbOf[c] <= 1) continue;                       // P0: nothing to limit
    // cell mean (GLL over the tensor cell -- the state's natural mean)
    double Ub[5] = {0,0,0,0,0}, wsum = 0, U[5][blockSizeTot];
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      i32 i2=nd%NNODE, j2=(nd/NNODE)%NNODE, k2=nd/(NNODE*NNODE);
      double wn = w[i2]*w[j2]*w[k2]; wsum += wn;
      for (i32 q = 0; q < 5; q++) {
        U[q][nd] = (double)getField(D_RHO+q)[(size_t)b*blockSizeTot+nd];
        Ub[q] += wn*U[q][nd];
      } }
    for (i32 q = 0; q < 5; q++) Ub[q] /= wsum;
    // characteristic frame at the wall tangent (Giuliani's direction choice)
    double Wb[5] = { Ub[0], Ub[1]/fmax(Ub[0],1e-12), Ub[2]/fmax(Ub[0],1e-12),
                     Ub[3]/fmax(Ub[0],1e-12), 0 };
    Wb[4] = (dgGam-1.0)*(Ub[4]-0.5*(Ub[1]*Ub[1]+Ub[2]*Ub[2]+Ub[3]*Ub[3])/fmax(Ub[0],1e-12));
    if (Wb[0] <= 1e-12 || Wb[4] <= 1e-12) continue;
    double tan2[2] = { -(double)cutWallN[2*c+1], (double)cutWallN[2*c] };
    double R[5][5], L[5][5];
    dgEulerEigH(Wb, tan2, R, L);
    // neighbour mean deviations in characteristic variables
    double wlo[5], whi[5]; bool anyN = false;
    for (i32 q = 0; q < 5; q++) { wlo[q] = 0; whi[q] = 0; }
    for (i32 fdir = 0; fdir < 4; fdir++) {
      i32 nn = nbrIdxList[27*b + faceSlot2[fdir]];
      if (nn == bEmpty || nn < 0) continue;
      if (ibClassList[nn] == IB_DEAD) continue;
      double Un[5] = {0,0,0,0,0};
      for (i32 nd = 0; nd < blockSizeTot; nd++) {
        i32 i2=nd%NNODE, j2=(nd/NNODE)%NNODE, k2=nd/(NNODE*NNODE);
        double wn = w[i2]*w[j2]*w[k2];
        for (i32 q = 0; q < 5; q++)
          Un[q] += wn*(double)getField(D_RHO+q)[(size_t)nn*blockSizeTot+nd];
      }
      for (i32 q = 0; q < 5; q++) Un[q] /= wsum;
      anyN = true;
      double dw[5];
      for (i32 k2 = 0; k2 < 5; k2++) {
        double s2 = 0;
        for (i32 q = 0; q < 5; q++) s2 += L[k2][q]*(Un[q]-Ub[q]);
        dw[k2] = s2;
      }
      for (i32 k2 = 0; k2 < 5; k2++) { wlo[k2] = fmin(wlo[k2], dw[k2]); whi[k2] = fmax(whi[k2], dw[k2]); }
    }
    if (!anyN) continue;
    // extrapolate MY polynomial to the neighbour centroids; per-field BJ phi
    double phi[5] = {1,1,1,1,1};
    for (i32 fdir = 0; fdir < 4; fdir++) {
      double xr[3] = { 0.5+ctr[fdir][0], 0.5+ctr[fdir][1], 0.5 };
      double Lx[NNODE], Ly[NNODE], Lz[NNODE];
      lag1(2.0*xr[0]-1.0, Lx); lag1(2.0*xr[1]-1.0, Ly); lag1(2.0*xr[2]-1.0, Lz);
      double Ue[5] = {0,0,0,0,0};
      for (i32 nd = 0; nd < blockSizeTot; nd++) {
        i32 i2=nd%NNODE, j2=(nd/NNODE)%NNODE, k2=nd/(NNODE*NNODE);
        double ph = Lx[i2]*Ly[j2]*Lz[k2];
        for (i32 q = 0; q < 5; q++) Ue[q] += ph*U[q][nd];
      }
      for (i32 k2 = 0; k2 < 5; k2++) {
        double d = 0;
        for (i32 q = 0; q < 5; q++) d += L[k2][q]*(Ue[q]-Ub[q]);
        if (d > 1e-14)       phi[k2] = fmin(phi[k2], fmax(0.0, whi[k2])/d);
        else if (d < -1e-14) phi[k2] = fmin(phi[k2], fmax(0.0, -wlo[k2])/(-d));
      }
    }
    bool active = false;
    for (i32 k2 = 0; k2 < 5; k2++) { phi[k2] = fmin(phi[k2], 1.0); if (phi[k2] < 1.0) active = true; }
    if (!active) continue;
    // U := Ubar + R diag(phi) L (U - Ubar), nodal
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      double dw[5], du[5];
      for (i32 k2 = 0; k2 < 5; k2++) {
        double s2 = 0;
        for (i32 q = 0; q < 5; q++) s2 += L[k2][q]*(U[q][nd]-Ub[q]);
        dw[k2] = phi[k2]*s2;
      }
      for (i32 q = 0; q < 5; q++) {
        double s2 = 0;
        for (i32 k2 = 0; k2 < 5; k2++) s2 += R[q][k2]*dw[k2];
        du[q] = s2;
      }
      for (i32 q = 0; q < 5; q++)
        getField(D_RHO+q)[(size_t)b*blockSizeTot+nd] = (real)(Ub[q] + du[q]);
    }
  }
}

// ---------------------------------------------------------------------------
//  IMAGE PATCH.  dgComputeImageDataKernel voids the solid side of a cut element,
//  but its FLUID pixels were still hat-interpolated from the block's tensor
//  Lobatto nodes -- and those nodes include the solid-side ones, so the
//  interpolation drags the polynomial's unconstrained extension back into the
//  fluid pixels it was supposed to keep out.  Repaint them from the element's
//  OWN polynomial.  Conserved fields only (f = 0..4): a SCRATCH-backed paint
//  (pressure, troubled) is not a modal coefficient of anything.
//
//  MEASURED, and worth recording because it refutes the motivation above: the
//  difference is 2.3e-15.  A cut element's nodal state STARTS as the IC sampled
//  at the nodes and is only ever incremented by the modal RHS sampled at the
//  nodes, so it stays inside the element's own polynomial space and the tensor
//  interpolant reproduces the modal polynomial exactly.  This patch is
//  therefore INSURANCE, not a fix -- it earns its keep only where something
//  writes nodal values from outside that space (a non-representable IC, the
//  positivity clip, a MOOD redo).  The real defect was painting the solid side
//  at all, which the kernel mask fixes.  CUT_NOIMGPATCH=1 disables it.
// ---------------------------------------------------------------------------
void DgSolver::patchCutImage(i32 f) {
  if (!cutOn || nCutElem == 0 || f < 0 || f > 4) return;
  if (getenv("CUT_NOIMGPATCH")) return;     // A/B: paint from the tensor nodes instead
  cudaDeviceSynchronize();
  CutHostEval ev;
  i64 nPatch = 0; double dMax = 0;
  for (i32 c = 0; c < nCutElem; c++) {
    const i32 b = cutBlk[c];
    i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
    if (!isInteriorBlock(lvl, ib, jb, kb)) continue;
    double h[3]; hostElemSizeLocal(*this, lvl, h);
    ev.begin(*this, c);
    const i32 span = blockSize*powi(2, nLvls-1-lvl);
    for (i32 py = 0; py < span; py++) {
      const i32 jP = jb*span + py;
      if (jP < 0 || jP >= imageSizeX[1]) continue;
      for (i32 px = 0; px < span; px++) {
        const i32 iP = ib*span + px;
        if (iP < 0 || iP >= imageSizeX[0]) continue;
        const double xr[3] = { (px + 0.5)/span, (py + 0.5)/span, 0.5 };
        const double X = (ib + xr[0])*h[0], Y = (jb + xr[1])*h[1];
        const double dx = X - (double)ibX, dy = Y - (double)ibY;
        // same shifted surface as the build (see the degeneracy guard)
        if (sqrt(dx*dx + dy*dy) - (double)ibR < -(double)cutEps) continue;   // solid
        double U[5]; ev.consAt(xr, U);
        if (getenv("CUT_IMGDBG")) {
          const double was = (double)imageDataX[(u64)jP*imageSizeX[0] + iP];
          nPatch++; dMax = fmax(dMax, fabs(was - U[f]));
        }
        imageDataX[(u64)jP*imageSizeX[0] + iP] = (real)U[f];
      }
    }
  }
  if (getenv("CUT_IMGDBG"))
    printf("[imgpatch] f=%d  repainted %lld px  max |nodal - modal| = %.6e\n",
           f, (long long)nPatch, dMax);
}

// ---------------------------------------------------------------------------
//  Max relative deviation of the cut elements from a uniform state, measured
//  where the polynomial actually lives (its own volume rule) rather than at the
//  tensor nodes -- half of which are inside the solid, where the value is an
//  extension and deviating from the free stream means nothing.
// ---------------------------------------------------------------------------
double DgSolver::cutMaxDeviation(const double U0[5]) {
  if (!cutOn || nCutElem == 0) return 0.0;
  cudaDeviceSynchronize();
  double dev = 0;
  CutHostEval ev;
  for (i32 c = 0; c < nCutElem; c++) {
    ev.begin(*this, c);
    for (i32 q = cutVolOff[c]; q < cutVolOff[c+1]; q++) {
      const SayeNode &s = cutVolP[q];
      if ((double)s.w <= 0.0) continue;
      const double xr[3] = {(double)s.x[0], (double)s.x[1], (double)s.x[2]};
      double U[5]; ev.consAt(xr, U);
      for (i32 f = 0; f < 5; f++)
        dev = fmax(dev, fabs(U[f] - U0[f])/fmax(fabs(U0[f]), 1.0));
    }
  }
  return dev;
}

// ---------------------------------------------------------------------------
//  CUT-CELL OUTPUT.  The first writer in this solver that samples a cut
//  element's ACTUAL solution.
//
//  A cut element's state is a modal polynomial in the orthonormal basis
//  psi~ = L^-1 psi, supported on the FLUID region {phi>0} of its cell.  Those
//  coefficients exist only in shared memory inside dgRhsCutKernel and are never
//  persisted, so every existing artifact -- the PNGs, --cutdump -- has instead
//  been reading the block's tensor Lobatto node slots, which on a cut element
//  include points buried inside the solid where the polynomial is an
//  unconstrained extension.  Same trap as reporting nodal values in a B-spline
//  basis: the numbers are real, they are just not samples of the field.
//
//  So: re-run the kernel's own nodal->modal projection on the host (identical
//  arithmetic, DgSolverKernels.cu:5920-5932), then evaluate at points that are
//  inside the region the polynomial represents BY CONSTRUCTION -- the Saye
//  volume rule for the interior, the Saye surface rule for the wall.  The wall
//  samples are also the first wall data a --cutcell run has ever produced:
//  computeIbGates/writeIbSurface are gated on ibOn, which buildCutElems sets to
//  0, so cut runs emitted no Cp, no wall pressure and no drag at all.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  Domain totals over the CUT BAND, integrated over the FLUID region.
//
//  dgTotalConserved sums the tensor GLL grid with full-cell weights, which is
//  wrong for a cut element twice over: under --cutmodal the field slots hold
//  modal coefficients rather than nodal values, and the full-cell weights
//  integrate the solid side as well.  Both errors vanish on a uniform state --
//  a constant polynomial has c~_m = 0 for m > 0 and the offset is fixed -- so
//  the free-stream gate never saw them, while dM/M0 on any DEVELOPING flow was
//  measuring the coefficients drifting, not mass moving.
//
//  Here the element's own fitted volume rule is used, which is the same rule
//  the RHS integrates over, so what this reports is exactly the quantity the
//  scheme conserves.  Reference measure x h0 h1 h2 = physical volume.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  STEADY-STATE MONITOR:  ||dU/dt||_L2(fluid) / ||U||_L2(fluid).
//
//  Integrated with the SAME functional the conservation monitor uses -- cut
//  elements over their own Saye volume rule with their modal coefficients,
//  uncut elements over the tensor GLL rule -- so a cut block contributes what
//  it actually holds rather than coefficients read as nodal values.  Solid
//  (IB_DEAD) blocks are excluded.
//
//  D_RHS holds the last RK stage's residual after dgStep returns, which is the
//  conventional steady-state residual.  A converged run drives this to the
//  level at which the spatial discretisation's own inconsistency lives; it will
//  NOT reach round-off on a cut mesh.
// ---------------------------------------------------------------------------
double DgSolver::dgResidualNorm(void) {
  cudaDeviceSynchronize();
  double num = 0, den = 0;
  double wq[NNODE], xq[NNODE]; dgGetHostOps(wq, xq, gauss);
  std::vector<double> psi(CUT_NBMAX_H);
  for (i32 b = 0; b < hashTable.nKeys; b++) {
    if (bLocList[b] == kEmpty) continue;
    if (ibClassList && ibClassList[b] == IB_DEAD) continue;
    if (ibOn && ibClassList && ibClassList[b] != IB_FLUID) continue;
    i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
    double h[3]; hostElemSizeLocal(*this, lvl, h);
    const i32 c = (cutOn && blkCut) ? blkCut[b] : -1;
    if (c >= 0) {
      const double jac = h[0]*h[1]*h[2];
      const i32 nb = cutNbOf[c];
      CutHostEval ev; ev.begin(*this, c);              // basis() only
      for (i32 g = cutVolOff[c]; g < cutVolOff[c+1]; g++) {
        const SayeNode &sn = cutVolP[g];
        const double xr[3] = {(double)sn.x[0], (double)sn.x[1], (double)sn.x[2]};
        ev.basis(xr, psi.data());
        const double wv = (double)sn.w*jac;
        for (i32 q = 0; q < 5; q++) {
          double r = 0, u = 0;
          for (i32 m = 0; m < nb; m++) {
            r += (double)getField(D_RHS+q)[(size_t)b*blockSizeTot+m]*psi[m];
            u += (double)getField(D_RHO+q)[(size_t)b*blockSizeTot+m]*psi[m];
          }
          num += wv*r*r; den += wv*u*u;
        }
      }
      continue;
    }
    for (i32 nd = 0; nd < blockSizeTot; nd++) {
      const i32 i2 = nd%NNODE, j2 = (nd/NNODE)%NNODE, k2 = nd/(NNODE*NNODE);
      const double wv = (0.5*h[0]*wq[i2])*(0.5*h[1]*wq[j2])*(0.5*h[2]*wq[k2]);
      for (i32 q = 0; q < 5; q++) {
        const double r = (double)getField(D_RHS+q)[(size_t)b*blockSizeTot+nd];
        const double u = (double)getField(D_RHO+q)[(size_t)b*blockSizeTot+nd];
        num += wv*r*r; den += wv*u*u;
      }
    }
  }
  return (den > 0) ? sqrt(num/den) : 0.0;
}

void DgSolver::dgCutConserved(double &mass, double &momx, double &energy) {
  cudaDeviceSynchronize();
  mass = momx = energy = 0;
  if (!cutOn || nCutElem <= 0) return;
  CutHostEval ev;
  std::vector<double> psi(CUT_NBMAX_H);
  for (i32 c = 0; c < nCutElem; c++) {
    const i32 b = cutBlk[c];
    if (b < 0 || bLocList[b] == kEmpty) continue;
    i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
    double h[3]; hostElemSizeLocal(*this, lvl, h);
    const double jac = h[0]*h[1]*h[2];
    ev.begin(*this, c);
    for (i32 q = cutVolOff[c]; q < cutVolOff[c+1]; q++) {
      const SayeNode &s = cutVolP[q];
      const double xr[3] = {(double)s.x[0], (double)s.x[1], (double)s.x[2]};
      ev.basis(xr, psi.data());
      double U[5] = {0,0,0,0,0};
      for (i32 m = 0; m < ev.nb; m++)
        for (i32 fq = 0; fq < 5; fq++) U[fq] += ev.cmod[(size_t)m*5+fq]*psi[m];
      const double wv = (double)s.w*jac;
      mass   += wv*U[0];
      momx   += wv*U[1];
      energy += wv*U[4];
    }
  }
}

void DgSolver::writeCutFields(const char *stem) {
  if (!cutOn || nCutElem == 0) return;
  cudaDeviceSynchronize();

  double wq[NNODE], xi[NNODE];
  dgGetHostOps(wq, xi, gauss);

  char fn[256];
  snprintf(fn, sizeof fn, "%s_geom.csv", stem);
  FILE *fg = fopen(fn, "w");
  snprintf(fn, sizeof fn, "%s_wall.csv", stem);
  FILE *fw = fopen(fn, "w");
  snprintf(fn, sizeof fn, "%s_vol.csv", stem);
  FILE *fv = fopen(fn, "w");
  if (!fg || !fw || !fv) { if(fg)fclose(fg); if(fw)fclose(fw); if(fv)fclose(fv); return; }
  fprintf(fg, "elem,block,ib,jb,lvl,x0,y0,hx,hy,volfrac,wallarea,nmodes,bndincons\n");
  fprintf(fw, "elem,x,y,w,nx,ny,rho,u,v,p,cp,mach\n");
  fprintf(fv, "elem,x,y,w,rho,u,v,p\n");

  const double pInf = 1.0/(double)dgGam, qInf = 0.5*(double)machInf*(double)machInf;

  for (i32 c = 0; c < nCutElem; c++) {
    const i32 b = cutBlk[c], nb = cutNbOf[c];
    i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
    double h[3]; hostElemSizeLocal(*this, lvl, h);

    // ONE evaluator for every sample (CutHostEval): it knows whether the block
    // holds nodal values or coefficients.  This function used to carry its own
    // copy of the projection, written before that existed, and under
    // --cutmodal it re-projected coefficients as if they were node values --
    // which is where a wall Cp of -16 and NEGATIVE wall pressures came from,
    // while cut_fine.csv (which already used the shared evaluator) reported the
    // same solution as entirely sensible.  Duplicated state logic, one copy
    // updated.
    CutHostEval ev; ev.begin(*this, c);
    double volFrac = 0.0;
    for (i32 g = cutVolOff[c]; g < cutVolOff[c+1]; g++) volFrac += (double)cutVolP[g].w;
    auto sampleAt = [&](const double xr[3], double W[5]) {
      double U[5]; ev.consAt(xr, U);
      W[0] = U[0];
      W[1] = U[1]/U[0]; W[2] = U[2]/U[0]; W[3] = U[3]/U[0];
      W[4] = ((double)dgGam-1.0)*(U[4] - 0.5*(U[1]*U[1]+U[2]*U[2]+U[3]*U[3])/U[0]);
    };

    double wallArea = 0;
    for (i32 g = cutWalOff[c]; g < cutWalOff[c+1]; g++) wallArea += (double)cutWalP[g].w;
    fprintf(fg, "%d,%d,%d,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8e,%.8e,%d,%.6e\n",
            c, b, ib, jb, lvl, ib*h[0], jb*h[1], h[0], h[1],
            volFrac, wallArea, nb, cutQual ? (double)cutQual[c] : -1.0);

    for (i32 g = cutWalOff[c]; g < cutWalOff[c+1]; g++) {
      const SayeNode &s = cutWalP[g];
      const double xr[3] = {(double)s.x[0], (double)s.x[1], (double)s.x[2]};
      double W[5]; sampleAt(xr, W);
      const double X = (ib + xr[0])*h[0], Y = (jb + xr[1])*h[1];
      const double aSnd = sqrt((double)dgGam*fmax(W[4],1e-30)/fmax(W[0],1e-30));
      fprintf(fw, "%d,%.8f,%.8f,%.8e,%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.6f\n",
              c, X, Y, (double)s.w, (double)s.n[0], (double)s.n[1],
              W[0], W[1], W[2], W[4], (W[4]-pInf)/fmax(qInf,1e-30),
              sqrt(W[1]*W[1]+W[2]*W[2])/aSnd);
    }
    for (i32 g = cutVolOff[c]; g < cutVolOff[c+1]; g++) {
      const SayeNode &s = cutVolP[g];
      if ((double)s.w <= 0.0) continue;             // padded zero-weight point
      const double xr[3] = {(double)s.x[0], (double)s.x[1], (double)s.x[2]};
      double W[5]; sampleAt(xr, W);
      fprintf(fv, "%d,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e\n",
              c, (ib + xr[0])*h[0], (jb + xr[1])*h[1], (double)s.w,
              W[0], W[1], W[2], W[4]);
    }
  }
  // ---- FINE SAMPLE: the raster gives a cut element only blockSize pixels per
  // axis, which is why the cut band reads as blocks whatever the degree.  Here
  // the element's own polynomial is evaluated on a dense reference grid,
  // FLUID SIDE ONLY, so the band can be drawn at whatever resolution the
  // picture wants without inventing data in the solid.
  const i32 nfine = getenv("CUT_NFINE") ? atoi(getenv("CUT_NFINE")) : 16;
  snprintf(fn, sizeof fn, "%s_fine.csv", stem);
  if (FILE *ff = fopen(fn, "w")) {
    fprintf(ff, "elem,x,y,rho,u,v,p\n");
    for (i32 c = 0; c < nCutElem; c++) {
      const i32 b = cutBlk[c];
      i32 lvl, ib, jb, kb; decode(bLocList[b], lvl, ib, jb, kb);
      double h[3]; hostElemSizeLocal(*this, lvl, h);
      CutHostEval ev; ev.begin(*this, c);
      for (i32 jy = 0; jy < nfine; jy++)
      for (i32 ix = 0; ix < nfine; ix++) {
        const double xr[3] = {(ix+0.5)/nfine, (jy+0.5)/nfine, 0.5};
        const double X = (ib + xr[0])*h[0], Y = (jb + xr[1])*h[1];
        const double dx = X - (double)ibX, dy = Y - (double)ibY;
        if (sqrt(dx*dx + dy*dy) - (double)ibR < -(double)cutEps) continue;   // solid
        double U[5]; ev.consAt(xr, U);
        const double rho = U[0];
        const double pr = ((double)dgGam-1.0)*(U[4] - 0.5*(U[1]*U[1]+U[2]*U[2]+U[3]*U[3])/rho);
        fprintf(ff, "%d,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e\n",
                c, X, Y, rho, U[1]/rho, U[2]/rho, pr);
      }
    }
    fclose(ff);
  }
  fclose(fg); fclose(fw); fclose(fv);
  printf("[cutfields] wrote %s_{geom,wall,vol,fine}.csv  (%d cut elements)\n", stem, nCutElem);
}
