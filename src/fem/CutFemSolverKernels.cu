#include "CutFemSolverKernels.cuh"
#include "MultiLevelSparseGridKernels.cuh"

//
// CutFEM kernels.  The mesh kernels touch the octree; everything after that
// works on the flat element / node / face arrays built by
// CutFemSolver::setupDofs.
//

// Reference (h = 1) Q1 elasticity element matrix, mu/lambda already folded in.
// A physical element of size h has K = h*Kref (gradients ~ 1/h, volume ~ h^3).
__constant__ real c_Kref[576];

void femSetKref(const real *K576) {
  cudaMemcpyToSymbol(c_Kref, K576, 576*sizeof(real));
}

// 4x4 bilinear mass matrix on the unit square (M1 tensor M1), index m = b + 2c
__constant__ real c_Mface[16];
static const real h_Mface[16] = {
  (real)(4.0/36), (real)(2.0/36), (real)(2.0/36), (real)(1.0/36),
  (real)(2.0/36), (real)(4.0/36), (real)(1.0/36), (real)(2.0/36),
  (real)(2.0/36), (real)(1.0/36), (real)(4.0/36), (real)(2.0/36),
  (real)(1.0/36), (real)(2.0/36), (real)(2.0/36), (real)(4.0/36),
};
void femInitFaceMass(void) { cudaMemcpyToSymbol(c_Mface, h_Mface, 16*sizeof(real)); }

// ---------------------------------------------------------------------------
//  small device helpers
// ---------------------------------------------------------------------------

// gradient of one Q1 shape function (physical, ih = 1/h)
__device__ __forceinline__ void q1GradOne(real xi, real et, real ze, real ih,
                                          i32 n, real G[3]) {
  i32 a = n&1, b = (n>>1)&1, c = (n>>2)&1;
  real sx = a ? xi : 1-xi, sy = b ? et : 1-et, sz = c ? ze : 1-ze;
  real dx = a ? (real)1 : (real)-1, dy = b ? (real)1 : (real)-1, dz = c ? (real)1 : (real)-1;
  G[0] = dx*sy*sz*ih;
  G[1] = sx*dy*sz*ih;
  G[2] = sx*sy*dz*ih;
}

__device__ __forceinline__ real q1ShapeOne(real xi, real et, real ze, i32 n) {
  i32 a = n&1, b = (n>>1)&1, c = (n>>2)&1;
  return (a ? xi : 1-xi) * (b ? et : 1-et) * (c ? ze : 1-ze);
}

// 3-point Gauss on [0,1]
__device__ __forceinline__ void gauss3(i32 q, real &x, real &w) {
  const real g[3]  = {(real)0.1127016653792583, (real)0.5, (real)0.8872983346207417};
  const real ww[3] = {(real)(5.0/18), (real)(8.0/18), (real)(5.0/18)};
  x = g[q]; w = ww[q];
}

// uniform block reduction of a double into acc[slot].  Every thread of the
// CUDA block must reach this (it syncs), so callers put it OUTSIDE any
// divergent element loop.
__device__ void femBlockAdd(double v, double *acc, i32 slot) {
  __shared__ double s[32];
  i32 lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (i32 o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  if (lane == 0) s[warp] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    double t = 0;
    i32 nw = (blockDim.x + 31)/32;
    for (i32 i = 0; i < nw; i++) t += s[i];
    if (t != 0) atomicAdd(&acc[slot], t);
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
//  mesh: reduce the dense base grid to the ACTIVE mesh K_h (2.13)
// ---------------------------------------------------------------------------
//
// One CUDA block per grid block; each thread evaluates the level set at one of
// the (blockSize+1)^3 block-corner nodes.  A block survives if any of its cells
// has a corner with phi < 0, i.e. if its trilinear level set goes negative --
// exactly the condition K \cap Omega != 0 for at least one cell.
//
__global__ void femBlockActiveKernel(CutFemSolver &S) {
  const i32 nn1 = blockSize + 1;
  const i32 nnb = nn1*nn1*nn1;
  __shared__ real sp[(blockSize+1)*(blockSize+1)*(blockSize+1)];
  __shared__ i32  keep;

  for (i32 b = blockIdx.x; b < S.hashTable.nKeys; b += gridDim.x) {
    u64 loc = S.bLocList[b];
    if (loc == kEmpty) { if (threadIdx.x == 0) S.bFlagsList[b] = DELETE; continue; }
    i32 lvl, ib, jb, kb; S.decode(loc, lvl, ib, jb, kb);
    real h = S.getDx(lvl);

    if (threadIdx.x == 0) keep = 0;
    __syncthreads();
    for (i32 t = threadIdx.x; t < nnb; t += blockDim.x) {
      i32 i = t % nn1, j = (t/nn1) % nn1, k = t/(nn1*nn1);
      sp[t] = S.ls.phi((ib*blockSize + i)*h, (jb*blockSize + j)*h, (kb*blockSize + k)*h);
    }
    __syncthreads();
    for (i32 t = threadIdx.x; t < blockSizeTot; t += blockDim.x) {
      i32 i = t % blockSize, j = (t/blockSize) % blockSize, k = t/(blockSize*blockSize);
      for (i32 n = 0; n < 8; n++) {
        i32 ii = i + (n&1), jj = j + ((n>>1)&1), kk = k + ((n>>2)&1);
        if (sp[ii + nn1*(jj + nn1*kk)] < 0) { keep = 1; break; }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) S.bFlagsList[b] = keep ? KEEP : DELETE;
    __syncthreads();
  }
}

// retire the flagged blocks (the lean grid has no cFlagsList / fieldData, so
// this replaces MultiLevelSparseGridKernels' deleteDataKernel)
__global__ void femPruneKernel(CutFemSolver &S) {
  for (i32 b = blockIdx.x*blockDim.x + threadIdx.x; b < S.hashTable.nKeys;
       b += gridDim.x*blockDim.x) {
    if (S.bFlagsList[b] == DELETE && S.bLocList[b] != kEmpty) {
      S.bLocList[b] = kEmpty;
      S.bIdxList[b] = bEmpty;
      atomicAdd(&S.nBlocks, -1);
    }
  }
}

// Nodal 1-jet of the surviving blocks, in post-sort block order: phi in
// phiBlk, and the unit normal n = grad phi / |grad phi| (computational frame) in
// nrmBlk.  The oracle computes the gradient anyway on its way to the sign
// (phiGrad threads it through the whole CSG tree), so the normal is essentially
// free.  It is used to detect slivers and creases (see femSliverKernel) and is
// available to any cut computation that wants a better interface than linear.
__global__ void femPhiKernel(CutFemSolver &S) {
  const i32 nn1 = blockSize + 1;
  const i32 nnb = nn1*nn1*nn1;
  for (i32 b = blockIdx.x; b < S.hashTable.nKeys; b += gridDim.x) {
    u64 loc = S.bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; S.decode(loc, lvl, ib, jb, kb);
    real h = S.getDx(lvl);
    for (i32 t = threadIdx.x; t < nnb; t += blockDim.x) {
      i32 i = t % nn1, j = (t/nn1) % nn1, k = t/(nn1*nn1);
      real g[3];
      real p = S.ls.phiGrad((ib*blockSize + i)*h, (jb*blockSize + j)*h,
                            (kb*blockSize + k)*h, g);
      S.phiBlk[(size_t)b*nnb + t] = p;
      real gn = sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
      real inv = (gn > (real)1e-20) ? 1/gn : (real)0;
      S.nrmBlk[3*((size_t)b*nnb + t)]     = g[0]*inv;
      S.nrmBlk[3*((size_t)b*nnb + t) + 1] = g[1]*inv;
      S.nrmBlk[3*((size_t)b*nnb + t) + 2] = g[2]*inv;
    }
  }
}

//
// Per-cut-cell geometric quality, from the 1-jet ALONE (no marching tets).
//
// PREDICTED cut fraction: linearize phi about the cell centre as the half-space
// { phi_c + n_c . (x - x_c) < 0 } and intersect with the unit cube -- a
// closed-form slab-cube volume.  Cheap, and it agrees with the true fraction
// wherever the surface is planar across the cell (which is where a fraction is
// meaningful in the first place).
//
// CREASE / under-resolution flag: the spread of the 8 corner normals.  If they
// disagree by more than `creaseCos`, the cell straddles a kink in phi (two CSG
// surfaces meet inside it) or a feature thinner than h -- and the piecewise-
// planar reconstruction the cut quadrature relies on is INVALID there.  That is
// the actionable signal: refine, or the geometry in that cell is wrong.
//
__device__ __forceinline__ real cubeSlabVol(real phic, const real nc[3]) {
  // |Omega cap cube| for the half-space phic + nc.(x - center) < 0 on [0,1]^3.
  // Shift so the plane is  a.x = d  with x in [0,1]^3; use the standard
  // simplex-sum formula.  a = nc, d = -phic + 0.5*(|nx|+|ny|+|nz|-sum nc) ...
  real a[3] = {nc[0], nc[1], nc[2]};
  // signed distance from cube centre (0.5,0.5,0.5): phic already is phi there
  real s = fabs(a[0]) + fabs(a[1]) + fabs(a[2]);
  if (s < (real)1e-20) return (phic < 0) ? (real)1 : (real)0;
  // fold to the positive orthant: reflect so all a_i >= 0, volume is invariant
  real A[3] = {fabs(a[0]), fabs(a[1]), fabs(a[2])};
  // plane through centre: a.(x-0.5) = -phic  ->  A.x = c with
  // c = -phic + 0.5*(A0+A1+A2)  (reflection maps a_i<0 var to 1-var)
  real c = -phic + (real)0.5*s;
  // volume of { A.x <= c, x in [0,1]^3 } by inclusion-exclusion of clamped
  // corner tetrahedra (Eberly).  A_i > 0, so:
  auto clamp01 = [](real v){ return v < 0 ? (real)0 : (v > 1 ? (real)1 : v); };
  real vol = 0;
  // f(t) = t^3 for t>=0 else 0, times sign; standard formula
  auto pos3 = [](real t){ return t > 0 ? t*t*t : (real)0; };
  real denom = 6*A[0]*A[1]*A[2];
  vol = ( pos3(c)
        - pos3(c - A[0]) - pos3(c - A[1]) - pos3(c - A[2])
        + pos3(c - A[0] - A[1]) + pos3(c - A[0] - A[2]) + pos3(c - A[1] - A[2])
        - pos3(c - A[0] - A[1] - A[2]) ) / denom;
  (void)clamp01;
  return vol < 0 ? (real)0 : (vol > 1 ? (real)1 : vol);
}

__global__ void femSliverKernel(CutFemSolver &S) {
  for (i32 ic = blockIdx.x*blockDim.x + threadIdx.x; ic < S.nCut;
       ic += gridDim.x*blockDim.x) {
    real phic = 0, nc[3] = {0,0,0};
    for (i32 n = 0; n < 8; n++) {
      phic += S.cutPhi[8*(size_t)ic + n];
      for (i32 d = 0; d < 3; d++) nc[d] += S.cutNrm[3*(8*(size_t)ic + n) + d];
    }
    phic *= (real)0.125;                       // phi at the cell centre (trilinear)
    real nn = sqrt(nc[0]*nc[0] + nc[1]*nc[1] + nc[2]*nc[2]);
    real nu[3] = {nc[0]/8, nc[1]/8, nc[2]/8};  // mean normal (unnormalized magnitude
                                               // = 1 for coherent, < 1 for a crease)
    real inv = (nn > (real)1e-20) ? 1/nn : 0;
    real ncu[3] = {nc[0]*inv, nc[1]*inv, nc[2]*inv};

    // predicted fraction (in units of the cube; the physical measure cancels in
    // the ratio the census reports)
    real fpred = cubeSlabVol(phic, ncu);
    S.cutFpred[ic] = fpred;

    // coherence = |mean of the 8 unit normals|.  1 = all agree (smooth), and it
    // falls toward 0 as they spread; 1 - coherence is the crease indicator.
    real coh = nn/8;
    S.cutCoh[ic] = coh;
    (void)nu;
  }
}

// ---------------------------------------------------------------------------
//  cut-element quadrature: one CUDA block per cut element
// ---------------------------------------------------------------------------
//
// Phase A -- each thread marches one Kuhn tetrahedron of one sub-cube into a
//   private buffer, then reserves a slot range in the shared point list.
// Phase B -- threads split the 576 matrix entries and the 24 load entries and
//   sweep the shared point list.
//
// The 24x24 holds a(.,.) restricted to K \cap Omega plus the Nitsche terms
// (2.21) on the Dirichlet part of dOmega \cap K; the 24-vector holds (f,v),
// the Neumann traction and the Nitsche data terms (2.27).
//
__global__ void femCutElemKernel(CutFemSolver &S) {
  __shared__ real sqv[4*FEM_MAXQV];
  __shared__ real sqs[7*FEM_MAXQS];
  __shared__ real sK[576];
  __shared__ real sF[24];
  __shared__ real sPhi[8];
  __shared__ real sC[8][3];
  __shared__ real sVol, sArea;
  __shared__ i32  snv, sns, sAllIn;

  const real mu = S.prob.mu, lam = S.prob.lam;
  const i32  sub = S.cutSub;
  const i32  nTet = 6*sub*sub*sub;

  for (i32 ic = blockIdx.x; ic < S.nCut; ic += gridDim.x) {
    i32 e = S.cutElem[ic];
    real h = S.eH[e];

    if (threadIdx.x == 0) { snv = 0; sns = 0; sAllIn = 1; sVol = 0; sArea = 0; }
    if (threadIdx.x < 8) {
      real p = S.cutPhi[8*(size_t)ic + threadIdx.x];
      sPhi[threadIdx.x] = p;
      if (p >= 0) sAllIn = 0;
      // physical corner: the map is the identity in Cartesian mode
      i32 n = threadIdx.x;
      real q0 = S.eX0[3*(size_t)e]   + h*(n&1);
      real q1 = S.eX0[3*(size_t)e+1] + h*((n>>1)&1);
      real q2 = S.eX0[3*(size_t)e+2] + h*((n>>2)&1);
      S.ls.toPhys(q0, q1, q2, sC[n][0], sC[n][1], sC[n][2]);
    }
    for (i32 t = threadIdx.x; t < 576; t += blockDim.x) sK[t] = 0;
    for (i32 t = threadIdx.x; t < 24;  t += blockDim.x) sF[t] = 0;
    __syncthreads();

    // ---- phase A: quadrature points, in REFERENCE coordinates -------------
    if (sAllIn) {
      // fully interior: 3x3x3 Gauss is exact for the Q1 stiffness integrand,
      // where a marching-tet rule (degree 2 per sub-tet) would not be
      if (threadIdx.x == 0) {
        snv = 27;
        for (i32 qk = 0; qk < 3; qk++)
        for (i32 qj = 0; qj < 3; qj++)
        for (i32 qi = 0; qi < 3; qi++) {
          real xi, et, ze, wx, wy, wz;
          gauss3(qi, xi, wx); gauss3(qj, et, wy); gauss3(qk, ze, wz);
          real *o = sqv + 4*(qi + 3*qj + 9*qk);
          o[0] = xi; o[1] = et; o[2] = ze; o[3] = wx*wy*wz;
        }
      }
    } else {
      for (i32 t = threadIdx.x; t < nTet; t += blockDim.x) {
        real lqv[4*12], lqs[7*6];
        CutQuadBuf B;
        B.qv = lqv; B.qs = lqs; B.maxv = 12; B.maxs = 6; B.nv = 0; B.ns = 0;
        cqOneTet(B, sPhi, sub, h, t);
        if (B.nv) {
          i32 o = atomicAdd(&snv, B.nv);
          if (o + B.nv <= FEM_MAXQV)
            for (i32 q = 0; q < 4*B.nv; q++) sqv[4*o + q] = lqv[q];
        }
        if (B.ns) {
          i32 o = atomicAdd(&sns, B.ns);
          if (o + B.ns <= FEM_MAXQS)
            for (i32 q = 0; q < 7*B.ns; q++) sqs[7*o + q] = lqs[q];
        }
      }
    }
    __syncthreads();
    i32 nv = min(snv, FEM_MAXQV), ns = min(sns, FEM_MAXQS);

    // ---- the element's own cut measures ------------------------------------
    //
    // Volume and interface area of K \cap Omega.  These drive the sliver census
    // and the geometry diagnostics.
    //
    // Note the Nitsche penalty stays the paper's FIXED beta/h.  Scaling it by
    // the element's area-to-volume ratio (the usual inverse-estimate constant)
    // was tried and measurably HURT: on a thin cut element that ratio is huge,
    // the penalty over-constrains, and the cylindrical MMS error went from
    // 1.4e-2 to 2.9e-2.  Keeping beta fixed and letting the ghost penalty
    // restore the inverse estimate is exactly the paper's design.
    {
      real vloc = 0, aloc = 0;
      for (i32 q = threadIdx.x; q < nv; q += blockDim.x) {
        const real *P = sqv + 4*q;
        real G[8][3], detJ, Ji[3][3];
        if (q1GradIso(sC, P[0], P[1], P[2], G, detJ, Ji)) vloc += P[3]*fabs(detJ);
      }
      for (i32 q = threadIdx.x; q < ns; q += blockDim.x) {
        const real *P = sqs + 7*q;
        real G[8][3], detJ, Ji[3][3];
        if (!q1GradIso(sC, P[0], P[1], P[2], G, detJ, Ji)) continue;
        real mv[3];
        for (i32 i = 0; i < 3; i++) {
          real g = 0;
          for (i32 j = 0; j < 3; j++) g += Ji[j][i]*P[4+j];
          mv[i] = g;
        }
        aloc += P[3]*fabs(detJ)*sqrt(mv[0]*mv[0] + mv[1]*mv[1] + mv[2]*mv[2]);
      }
      atomicAdd(&sVol, vloc);
      atomicAdd(&sArea, aloc);
    }
    __syncthreads();
    real betaEff = S.gammaD/h;

    // ---- phase B1: stiffness ---------------------------------------------
    for (i32 r = threadIdx.x; r < 576; r += blockDim.x) {
      i32 row = r/24, col = r%24;
      i32 n = row/3, ii = row%3, m = col/3, jj = col%3;
      real dij = (ii == jj) ? (real)1 : (real)0;
      real s = 0;
      for (i32 q = 0; q < nv; q++) {
        const real *P = sqv + 4*q;
        real G[8][3], detJ, Ji[3][3];
        if (!q1GradIso(sC, P[0], P[1], P[2], G, detJ, Ji)) continue;
        real w = P[3]*fabs(detJ);
        const real *Gn = G[n], *Gm = G[m];
        real gg = dij ? (Gn[0]*Gm[0] + Gn[1]*Gm[1] + Gn[2]*Gm[2]) : (real)0;
        s += w*(mu*gg + mu*Gn[jj]*Gm[ii] + lam*Gn[ii]*Gm[jj]);
      }
      // Nitsche (2.21) on the Dirichlet part of the interface
      for (i32 q = 0; q < ns; q++) {
        const real *P = sqs + 7*q;
        real G[8][3], detJ, Ji[3][3];
        if (!q1GradIso(sC, P[0], P[1], P[2], G, detJ, Ji)) continue;
        // Nanson: n_phys ~ J^-T n_ref, dA_phys = |J^-T n_ref| detJ dA_ref
        real mvec[3];
        for (i32 i = 0; i < 3; i++) {
          real g = 0;
          for (i32 j = 0; j < 3; j++) g += Ji[j][i]*P[4+j];
          mvec[i] = g;
        }
        real mn = sqrt(mvec[0]*mvec[0] + mvec[1]*mvec[1] + mvec[2]*mvec[2]);
        if (!(mn > 0)) continue;
        real nrm[3] = {mvec[0]/mn, mvec[1]/mn, mvec[2]/mn};
        real wA = P[3]*fabs(detJ)*mn;
        real X[3]; q1PosIso(sC, P[0], P[1], P[2], X);
        if (!S.prob.isDirichlet(X[0], X[1], X[2])) continue;
        real Nn = q1ShapeOne(P[0], P[1], P[2], n);
        real Nm = q1ShapeOne(P[0], P[1], P[2], m);
        const real *Gn = G[n], *Gm = G[m];
        real gnn = Gn[0]*nrm[0] + Gn[1]*nrm[1] + Gn[2]*nrm[2];
        real gmn = Gm[0]*nrm[0] + Gm[1]*nrm[1] + Gm[2]*nrm[2];
        real t1 = -Nm*(mu*(dij*gnn + nrm[ii]*Gn[jj]) + lam*Gn[ii]*nrm[jj]);
        real t2 = -Nn*(mu*(dij*gmn + nrm[jj]*Gm[ii]) + lam*Gm[jj]*nrm[ii]);
        real t3 = betaEff * Nn*Nm*(2*mu*dij + lam*nrm[ii]*nrm[jj]);
        s += wA*(t1 + t2 + t3);
      }
      sK[r] = s;
    }

    // ---- phase B2: load ---------------------------------------------------
    for (i32 row = threadIdx.x; row < 24; row += blockDim.x) {
      i32 m = row/3, jj = row%3;
      real s = 0;
      for (i32 q = 0; q < nv; q++) {
        const real *P = sqv + 4*q;
        real G[8][3], detJ, Ji[3][3];
        if (!q1GradIso(sC, P[0], P[1], P[2], G, detJ, Ji)) continue;
        real X[3]; q1PosIso(sC, P[0], P[1], P[2], X);
        real f[3]; S.prob.bodyForce(X[0], X[1], X[2], f);
        s += P[3]*fabs(detJ)*f[jj]*q1ShapeOne(P[0], P[1], P[2], m);
      }
      for (i32 q = 0; q < ns; q++) {
        const real *P = sqs + 7*q;
        real G[8][3], detJ, Ji[3][3];
        if (!q1GradIso(sC, P[0], P[1], P[2], G, detJ, Ji)) continue;
        real mvec[3];
        for (i32 i = 0; i < 3; i++) {
          real g = 0;
          for (i32 j = 0; j < 3; j++) g += Ji[j][i]*P[4+j];
          mvec[i] = g;
        }
        real mn = sqrt(mvec[0]*mvec[0] + mvec[1]*mvec[1] + mvec[2]*mvec[2]);
        if (!(mn > 0)) continue;
        real nrm[3] = {mvec[0]/mn, mvec[1]/mn, mvec[2]/mn};
        real wA = P[3]*fabs(detJ)*mn;
        real X[3]; q1PosIso(sC, P[0], P[1], P[2], X);
        real Nm = q1ShapeOne(P[0], P[1], P[2], m);
        if (S.prob.isDirichlet(X[0], X[1], X[2])) {
          real g[3]; S.prob.dirichletData(X[0], X[1], X[2], g);
          const real *Gm = G[m];
          real gmn = Gm[0]*nrm[0] + Gm[1]*nrm[1] + Gm[2]*nrm[2];
          real gdotn = g[0]*nrm[0] + g[1]*nrm[1] + g[2]*nrm[2];
          real t1 = 0;
          for (i32 i = 0; i < 3; i++) {
            real dij = (i == jj) ? (real)1 : (real)0;
            t1 -= g[i]*(mu*(dij*gmn + nrm[jj]*Gm[i]) + lam*Gm[jj]*nrm[i]);
          }
          real t2 = betaEff * Nm*(2*mu*g[jj] + lam*gdotn*nrm[jj]);
          s += wA*(t1 + t2);
        } else {
          real g[3]; S.prob.neumannData(X[0], X[1], X[2], nrm, g);
          s += wA*g[jj]*Nm;
        }
      }
      sF[row] = s;
    }
    __syncthreads();

    for (i32 r = threadIdx.x; r < 576; r += blockDim.x) S.cutK[576*(size_t)ic + r] = sK[r];
    for (i32 r = threadIdx.x; r < 24;  r += blockDim.x) S.cutF[24*(size_t)ic + r]  = sF[r];

    // geometry diagnostics (the measures were already accumulated above)
    double vloc = (threadIdx.x == 0) ? (double)sVol : 0.0;
    double aloc = (threadIdx.x == 0) ? (double)sArea : 0.0;
    double dloc = 0;
    for (i32 q = threadIdx.x; q < ns; q += blockDim.x) {
      const real *P = sqs + 7*q;
      real G[8][3], detJ, Ji[3][3];
      if (!q1GradIso(sC, P[0], P[1], P[2], G, detJ, Ji)) continue;
      real mv[3];
      for (i32 i = 0; i < 3; i++) {
        real g = 0;
        for (i32 j = 0; j < 3; j++) g += Ji[j][i]*P[4+j];
        mv[i] = g;
      }
      real dA = P[3]*fabs(detJ)*sqrt(mv[0]*mv[0] + mv[1]*mv[1] + mv[2]*mv[2]);
      real X[3]; q1PosIso(sC, P[0], P[1], P[2], X);
      if (S.prob.isDirichlet(X[0], X[1], X[2])) dloc += dA;
    }
    femBlockAdd(vloc, S.acc, 0);
    femBlockAdd(aloc, S.acc, 1);
    femBlockAdd(dloc, S.acc, 6);
    // Sliver census: the smallest cut-volume fraction anywhere, and the
    // smallest among elements touching a theta face of the sector.  A periodic
    // boundary that lands on cell faces must not create slivers there -- if it
    // did, the tie would be sitting on top of exactly the degenerate cells the
    // ghost penalty exists to control.
    if (threadIdx.x == 0 && !sAllIn) {
      real hh = h*h*h;
      real frac = sVol/fmax(hh, (real)1e-30);
      // decade histogram of the cut-volume fraction: the honest way to see
      // whether refinement is thinning the small cuts out or just making more
      // of them.  Bin b counts fractions in [1e-(b+1), 1e-b).
      if (frac > 0) {
        i32 b = (i32)floor(-log10((double)frac));
        if (b < 0) b = 0;
        if (b > 15) b = 15;
        atomicAdd(&S.fracHist[b], 1);
      }
      if (frac > 0) {
        atomicMin((unsigned long long*)&S.slivMin,
                  (unsigned long long)__double_as_longlong((double)frac));
        i32 j = (i32)lrint((S.eX0[3*(size_t)e+1])/h);
        if (j == 0 || j == S.nThetaCells-1)
          atomicMin((unsigned long long*)&S.slivMinTheta,
                    (unsigned long long)__double_as_longlong((double)frac));
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
//  loads
// ---------------------------------------------------------------------------
__global__ void femFullLoadKernel(CutFemSolver &S) {
  double volAll = 0;
  for (i32 e = blockIdx.x*blockDim.x + threadIdx.x; e < S.nElem;
       e += gridDim.x*blockDim.x) {
    if (S.eCut[e] >= 0) continue;
    real h = S.eH[e], h3 = h*h*h;
    // PHYSICAL corners.  Every kernel evaluates the problem data in physical
    // (world) coordinates -- the cut kernel does it through the isoparametric
    // map, so this one must too, or interior and cut elements would be loaded
    // from f() sampled in two different frames.
    real C[8][3];
    for (i32 n = 0; n < 8; n++)
      S.ls.toPhys(S.eX0[3*(size_t)e]   + h*(n&1),
                  S.eX0[3*(size_t)e+1] + h*((n>>1)&1),
                  S.eX0[3*(size_t)e+2] + h*((n>>2)&1), C[n][0], C[n][1], C[n][2]);
    real Fl[24];
    for (i32 t = 0; t < 24; t++) Fl[t] = 0;
    for (i32 qk = 0; qk < 3; qk++)
    for (i32 qj = 0; qj < 3; qj++)
    for (i32 qi = 0; qi < 3; qi++) {
      real xi, et, ze, wx, wy, wz;
      gauss3(qi, xi, wx); gauss3(qj, et, wy); gauss3(qk, ze, wz);
      real w = wx*wy*wz*h3;
      real X[3]; q1PosIso(C, xi, et, ze, X);
      real f[3]; S.prob.bodyForce(X[0], X[1], X[2], f);
      real N[8]; q1Shape(xi, et, ze, N);
      for (i32 m = 0; m < 8; m++)
        for (i32 j = 0; j < 3; j++) Fl[3*m+j] += w*f[j]*N[m];
      volAll += w;
    }
    for (i32 m = 0; m < 8; m++) {
      i32 nd = S.eNode[8*e+m];
      for (i32 j = 0; j < 3; j++) atomicAdd(&S.yn[3*nd+j], Fl[3*m+j]);
    }
  }
  femBlockAdd(volAll, S.acc, 0);   // uniform: outside the divergent loop
}

__global__ void femCutLoadKernel(CutFemSolver &S) {
  for (i32 ic = blockIdx.x*blockDim.x + threadIdx.x; ic < S.nCut;
       ic += gridDim.x*blockDim.x) {
    i32 e = S.cutElem[ic];
    for (i32 m = 0; m < 8; m++) {
      i32 nd = S.eNode[8*e+m];
      for (i32 j = 0; j < 3; j++)
        atomicAdd(&S.yn[3*nd+j], S.cutF[24*(size_t)ic + 3*m + j]);
    }
  }
}

// ---------------------------------------------------------------------------
//  cyclic constraint operator
// ---------------------------------------------------------------------------
//
// P maps the real dofs to every node; the only non-identity block is the
// rotation by one pitch on the sector's far theta face.  R acts on the
// Cartesian displacement components, about the machine axis (+Z).
//
__device__ __forceinline__ void femRot(real c, real s, const real in[3], real out[3]) {
  out[0] = c*in[0] - s*in[1];
  out[1] = s*in[0] + c*in[1];
  out[2] = in[2];
}

__global__ void femProlongKernel(CutFemSolver &S, const real *x) {
  real c = cos(S.pitchAngle), s = sin(S.pitchAngle);
  for (i32 n = blockIdx.x*blockDim.x + threadIdx.x; n < S.nNode;
       n += gridDim.x*blockDim.x) {
    i32 m = S.nMap[n];
    real v[3] = {x[3*m], x[3*m+1], x[3*m+2]};
    if (S.nRot[n]) { real w[3]; femRot(c, s, v, w); v[0]=w[0]; v[1]=w[1]; v[2]=w[2]; }
    S.xn[3*n] = v[0]; S.xn[3*n+1] = v[1]; S.xn[3*n+2] = v[2];
    S.yn[3*n] = 0;    S.yn[3*n+1] = 0;    S.yn[3*n+2] = 0;
  }
}

__global__ void femRestrictKernel(CutFemSolver &S, real *y) {
  real c = cos(S.pitchAngle), s = sin(S.pitchAngle);
  for (i32 n = blockIdx.x*blockDim.x + threadIdx.x; n < S.nNode;
       n += gridDim.x*blockDim.x) {
    i32 m = S.nMap[n];
    real v[3] = {S.yn[3*n], S.yn[3*n+1], S.yn[3*n+2]};
    if (S.nRot[n]) { real w[3]; femRot(c, -s, v, w); v[0]=w[0]; v[1]=w[1]; v[2]=w[2]; }
    atomicAdd(&y[3*m],   v[0]);
    atomicAdd(&y[3*m+1], v[1]);
    atomicAdd(&y[3*m+2], v[2]);
  }
}

// diag(P^T D P) for a per-node diagonal D: (R^T D R)_ii = sum_k R_ki^2 D_kk
__global__ void femDiagRestrictKernel(CutFemSolver &S, real *d) {
  real c = cos(S.pitchAngle), s = sin(S.pitchAngle);
  for (i32 n = blockIdx.x*blockDim.x + threadIdx.x; n < S.nNode;
       n += gridDim.x*blockDim.x) {
    i32 m = S.nMap[n];
    real D[3] = {S.yn[3*n], S.yn[3*n+1], S.yn[3*n+2]};
    if (S.nRot[n]) {
      real t0 = c*c*D[0] + s*s*D[1];
      real t1 = s*s*D[0] + c*c*D[1];
      D[0] = t0; D[1] = t1;
    }
    atomicAdd(&d[3*m],   D[0]);
    atomicAdd(&d[3*m+1], D[1]);
    atomicAdd(&d[3*m+2], D[2]);
  }
}

// ---------------------------------------------------------------------------
//  operator:  y += A x
// ---------------------------------------------------------------------------
__global__ void femElemApplyKernel(CutFemSolver &S, const real *x, real *y) {
  for (i32 e = blockIdx.x*blockDim.x + threadIdx.x; e < S.nElem;
       e += gridDim.x*blockDim.x) {
    i32 nd[8];
    real xl[24];
    for (i32 m = 0; m < 8; m++) {
      nd[m] = S.eNode[8*e+m];
      xl[3*m]   = x[3*nd[m]];
      xl[3*m+1] = x[3*nd[m]+1];
      xl[3*m+2] = x[3*nd[m]+2];
    }
    i32 ic = S.eCut[e];
    const real *K; real sc;
    if (ic >= 0) { K = S.cutK + 576*(size_t)ic; sc = 1; }
    else         { K = c_Kref;                  sc = S.eH[e]; }
    for (i32 r = 0; r < 24; r++) {
      const real *Kr = K + 24*r;
      real s = 0;
      for (i32 c = 0; c < 24; c++) s += Kr[c]*xl[c];
      atomicAdd(&y[3*nd[r/3] + (r%3)], sc*s);
    }
  }
}

//
// Ghost penalty.  fNode stores [0..3] far-left, [4..7] on-face, [8..11]
// far-right in matching (b,c) order, so the jump of the normal derivative has
// nodal values q_m = (x_farR - 2 x_face + x_farL)/h and the face form is
// coef * q^T M q -- see the class header for where the h powers went.
//
__global__ void femFaceApplyKernel(CutFemSolver &S, const real *x, real *y) {
  for (i32 f = blockIdx.x*blockDim.x + threadIdx.x; f < S.nFace;
       f += gridDim.x*blockDim.x) {
    const i32 *nd = S.fNode + 12*f;
    real coef = S.fCoef[f];
    for (i32 i = 0; i < 3; i++) {
      real q[4], Mq[4];
      for (i32 m = 0; m < 4; m++)
        q[m] = x[3*nd[8+m]+i] - 2*x[3*nd[4+m]+i] + x[3*nd[m]+i];
      for (i32 m = 0; m < 4; m++) {
        real s = 0;
        for (i32 m2 = 0; m2 < 4; m2++) s += c_Mface[4*m+m2]*q[m2];
        Mq[m] = coef*s;
      }
      for (i32 m = 0; m < 4; m++) {
        atomicAdd(&y[3*nd[m]+i],       Mq[m]);
        atomicAdd(&y[3*nd[4+m]+i], -2*Mq[m]);
        atomicAdd(&y[3*nd[8+m]+i],     Mq[m]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
//  Jacobi diagonal
// ---------------------------------------------------------------------------
__global__ void femDiagElemKernel(CutFemSolver &S) {
  for (i32 e = blockIdx.x*blockDim.x + threadIdx.x; e < S.nElem;
       e += gridDim.x*blockDim.x) {
    i32 ic = S.eCut[e];
    const real *K; real sc;
    if (ic >= 0) { K = S.cutK + 576*(size_t)ic; sc = 1; }
    else         { K = c_Kref;                  sc = S.eH[e]; }
    for (i32 r = 0; r < 24; r++)
      atomicAdd(&S.yn[3*S.eNode[8*e + r/3] + (r%3)], sc*K[24*r + r]);
  }
}

__global__ void femDiagFaceKernel(CutFemSolver &S) {
  for (i32 f = blockIdx.x*blockDim.x + threadIdx.x; f < S.nFace;
       f += gridDim.x*blockDim.x) {
    const i32 *nd = S.fNode + 12*f;
    real coef = S.fCoef[f];
    for (i32 m = 0; m < 4; m++) {
      real d = coef*c_Mface[4*m+m];       // dq/dx = 1 on the far nodes, -2 on the face
      for (i32 i = 0; i < 3; i++) {
        atomicAdd(&S.yn[3*nd[m]+i],   d);
        atomicAdd(&S.yn[3*nd[4+m]+i], 4*d);
        atomicAdd(&S.yn[3*nd[8+m]+i], d);
      }
    }
  }
}

// ---------------------------------------------------------------------------
//  error norms over Omega (manufactured solution)
//    acc[2] = ||u-u_h||^2_L2   acc[3] = ||u||^2
//    acc[4] = energy error^2   acc[5] = energy norm^2
// ---------------------------------------------------------------------------
__global__ void femErrorKernel(CutFemSolver &S) {
  __shared__ real sqv[4*FEM_MAXQV];
  __shared__ real sPhi[8];
  __shared__ real ul[24];
  __shared__ real sC[8][3];
  __shared__ i32  snv;

  const real mu = S.prob.mu, lam = S.prob.lam;
  const i32  sub = S.cutSub;
  const i32  nTet = 6*sub*sub*sub;

  for (i32 e = blockIdx.x; e < S.nElem; e += gridDim.x) {
    real h = S.eH[e];
    i32 ic = S.eCut[e];

    if (threadIdx.x < 24)
      ul[threadIdx.x] = S.xn[3*S.eNode[8*e + threadIdx.x/3] + (threadIdx.x%3)];
    if (threadIdx.x < 8) {
      i32 n = threadIdx.x;
      S.ls.toPhys(S.eX0[3*(size_t)e]   + h*(n&1),
                  S.eX0[3*(size_t)e+1] + h*((n>>1)&1),
                  S.eX0[3*(size_t)e+2] + h*((n>>2)&1),
                  sC[n][0], sC[n][1], sC[n][2]);
    }
    if (threadIdx.x == 0) snv = 0;
    if (ic >= 0 && threadIdx.x < 8) sPhi[threadIdx.x] = S.cutPhi[8*(size_t)ic + threadIdx.x];
    __syncthreads();

    if (ic >= 0) {
      for (i32 t = threadIdx.x; t < nTet; t += blockDim.x) {
        real lqv[4*12], lqs[7*6];
        CutQuadBuf B;
        B.qv = lqv; B.qs = lqs; B.maxv = 12; B.maxs = 6; B.nv = 0; B.ns = 0;
        cqOneTet(B, sPhi, sub, h, t);
        if (B.nv) {
          i32 o = atomicAdd(&snv, B.nv);
          if (o + B.nv <= FEM_MAXQV)
            for (i32 q = 0; q < 4*B.nv; q++) sqv[4*o+q] = lqv[q];
        }
      }
    } else if (threadIdx.x == 0) {
      snv = 27;                                  // 3x3x3 Gauss on a full element
      for (i32 qk = 0; qk < 3; qk++)
      for (i32 qj = 0; qj < 3; qj++)
      for (i32 qi = 0; qi < 3; qi++) {
        real xi, et, ze, wx, wy, wz;
        gauss3(qi, xi, wx); gauss3(qj, et, wy); gauss3(qk, ze, wz);
        real *o = sqv + 4*(qi + 3*qj + 9*qk);
        o[0] = xi; o[1] = et; o[2] = ze; o[3] = wx*wy*wz;   // REFERENCE weight
      }
    }
    __syncthreads();
    i32 nv = min(snv, FEM_MAXQV);

    double eL2 = 0, nL2 = 0, eEn = 0, nEn = 0;
    for (i32 q = threadIdx.x; q < nv; q += blockDim.x) {
      const real *P = sqv + 4*q;
      real N[8], G[8][3], detJ, Ji[3][3];
      q1Shape(P[0], P[1], P[2], N);
      if (!q1GradIso(sC, P[0], P[1], P[2], G, detJ, Ji)) continue;
      real uu[3] = {0,0,0}, gh[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
      for (i32 m = 0; m < 8; m++)
        for (i32 i = 0; i < 3; i++) {
          uu[i] += N[m]*ul[3*m+i];
          for (i32 j = 0; j < 3; j++) gh[i][j] += G[m][j]*ul[3*m+i];
        }
      real X[3]; q1PosIso(sC, P[0], P[1], P[2], X);
      real ue[3], ge[3][3];
      S.prob.exactU(X[0], X[1], X[2], ue);
      S.prob.exactGradU(X[0], X[1], X[2], ge);
      real w = P[3]*fabs(detJ);
      for (i32 i = 0; i < 3; i++) {
        double d = uu[i] - ue[i];
        eL2 += w*d*d;
        nL2 += w*(double)ue[i]*ue[i];
      }
      double de = 0, ne = 0, dd = 0, nd = 0;
      for (i32 i = 0; i < 3; i++) {
        dd += gh[i][i] - ge[i][i];
        nd += ge[i][i];
        for (i32 j = 0; j < 3; j++) {
          double e1 = 0.5*((gh[i][j]-ge[i][j]) + (gh[j][i]-ge[j][i]));
          double e2 = 0.5*((double)ge[i][j] + ge[j][i]);
          de += e1*e1; ne += e2*e2;
        }
      }
      eEn += w*(2*mu*de + lam*dd*dd);
      nEn += w*(2*mu*ne + lam*nd*nd);
    }
    femBlockAdd(eL2, S.acc, 2);
    femBlockAdd(nL2, S.acc, 3);
    femBlockAdd(eEn, S.acc, 4);
    femBlockAdd(nEn, S.acc, 5);
    __syncthreads();
  }
}

// Sample the level set on a structured grid, on device -- the host BVH is far
// too slow for a few million probes (measured: minutes).
__global__ void femIsoSampleKernel(BladeSdf ls, real d0, real d1, real d2,
                                   i32 N0, i32 N1, i32 N2, float *xyz, float *phi) {
  size_t nTot = (size_t)(N0+1)*(N1+1)*(N2+1);
  for (size_t t = (size_t)blockIdx.x*blockDim.x + threadIdx.x; t < nTot;
       t += (size_t)gridDim.x*blockDim.x) {
    i32 i = (i32)(t % (size_t)(N0+1));
    i32 j = (i32)((t/(size_t)(N0+1)) % (size_t)(N1+1));
    i32 k = (i32)(t/((size_t)(N0+1)*(N1+1)));
    real q0 = i*d0, q1 = j*d1, q2 = k*d2;
    real X, Y, Z; ls.toPhys(q0, q1, q2, X, Y, Z);
    xyz[3*t] = (float)X; xyz[3*t+1] = (float)Y; xyz[3*t+2] = (float)Z;
    phi[t] = (float)ls.phi(q0, q1, q2);
  }
}

// ---------------------------------------------------------------------------
//  dense vector helpers
// ---------------------------------------------------------------------------
__global__ void femSetKernel(real *x, real v, i32 n) {
  for (i32 i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
    x[i] = v;
}

__global__ void femDotKernel(const real *a, const real *b, i32 n, double *out) {
  double s = 0;
  for (i32 i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
    s += (double)a[i]*(double)b[i];
  femBlockAdd(s, out, 0);
}

__global__ void femAxpyKernel(real *y, const real *x, real a, i32 n) {
  for (i32 i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
    y[i] += a*x[i];
}

__global__ void femXpayKernel(real *y, const real *x, real a, i32 n) {
  for (i32 i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
    y[i] = x[i] + a*y[i];
}

__global__ void femJacobiKernel(real *z, const real *r, const real *d, i32 n) {
  for (i32 i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
    z[i] = (d[i] > 0) ? r[i]/d[i] : r[i];
}
