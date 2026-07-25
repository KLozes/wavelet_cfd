#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <functional>
#include <unordered_map>
#include <vector>

#include "CutFemSolver.cuh"
#include "CutFemSolverKernels.cuh"
#include "MultiLevelSparseGridKernels.cuh"

//
// Host side of the CutFEM solver: the sparse active mesh, dof numbering, the
// stabilized-face list, the CG driver and the VTK output.
//
// The numerics live on the GPU (CutFemSolverKernels.cu); everything structural
// is built here.  That split is deliberate: the structure is built once and is
// far easier to get right (and to assert on) in serial host code, while the
// element quadrature and the CG iteration -- which dominate -- stay on device.
//

static long nowUs(void) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// host mirror of MultiLevelSparseGrid::decode (that one is __device__)
static inline void hDecode(u64 loc, i32 &lvl, i32 &i, i32 &j, i32 &k) {
  lvl = (i32)(loc >> 60);
  k = (i32)((loc >> 40) & ((1u<<20)-1)) - 1;
  j = (i32)((loc >> 20) & ((1u<<20)-1)) - 1;
  i = (i32)( loc        & ((1u<<20)-1)) - 1;
}

// node key from integer grid node coordinates (single level, so this is unique)
static inline u64 nodeKeyOf(i32 I, i32 J, i32 K) {
  return (u64)I | ((u64)J << 21) | ((u64)K << 42);
}
static inline void nodeKeyDec(u64 key, i32 &I, i32 &J, i32 &K) {
  I = (i32)( key        & ((1u<<21)-1));
  J = (i32)((key >> 21) & ((1u<<21)-1));
  K = (i32)((key >> 42) & ((1u<<21)-1));
}
// cell key, same packing (cells never collide with nodes -- separate tables)
static inline u64 cellKeyOf(i32 i, i32 j, i32 k) { return nodeKeyOf(i, j, k); }

// run f(begin,end) over [0,n) on all cores
template <class F>
static void parFor(i32 n, F f) {
  i32 nt = (i32)std::thread::hardware_concurrency();
  if (nt < 1) nt = 1;
  if (n < 4*nt) { f(0, n); return; }
  std::vector<std::thread> th;
  i32 chunk = (n + nt - 1)/nt;
  for (i32 t = 0; t < nt; t++) {
    i32 a = t*chunk, b = std::min(n, a + chunk);
    if (a >= b) break;
    th.emplace_back([&f,a,b]{ f(a,b); });
  }
  for (auto &t : th) t.join();
}

// ---------------------------------------------------------------------------
//  initialize
// ---------------------------------------------------------------------------
void CutFemSolver::initialize(void) {
  real dx = domainSize[0]/baseGridSize[0];
  real dy = domainSize[1]/baseGridSize[1];
  real dz = domainSize[2]/baseGridSize[2];
  if (fabs(dx-dy) > 1e-5*dx || fabs(dx-dz) > 1e-5*dx) {
    printf("ERROR: CutFEM assumes cubic cells (dx=%g dy=%g dz=%g)\n",
           (double)dx, (double)dy, (double)dz);
    exit(1);
  }
  if (gammaA < 0) gammaA = (2*prob.mu + prob.lam)*(real)1e-4;   // paper Sec 5

  // reference (h = 1) Q1 elasticity element matrix.  For u = N_n e_i and
  // v = N_m e_j the integrand of a(u,v) is
  //   mu dij grad N_n . grad N_m + mu d_j N_n d_i N_m + lam d_i N_n d_j N_m,
  // which is degree 2 per direction, so 2x2x2 Gauss is exact.
  real Kref[576];
  for (i32 t = 0; t < 576; t++) Kref[t] = 0;
  for (i32 q = 0; q < 8; q++) {
    real x[3], w;
    gauss2Point(q, x, w);
    real G[8][3]; q1Grad(x[0], x[1], x[2], (real)1, G);
    for (i32 n = 0; n < 8; n++)
    for (i32 ii = 0; ii < 3; ii++)
    for (i32 m = 0; m < 8; m++)
    for (i32 jj = 0; jj < 3; jj++) {
      real gg = (ii == jj) ? (G[n][0]*G[m][0] + G[n][1]*G[m][1] + G[n][2]*G[m][2]) : (real)0;
      Kref[24*(3*n+ii) + (3*m+jj)] +=
          w*(prob.mu*gg + prob.mu*G[n][jj]*G[m][ii] + prob.lam*G[n][ii]*G[m][jj]);
    }
  }
  femSetKref(Kref);
  femInitFaceMass();
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
//  mesh: the active set K_h = { K : K \cap Omega != 0 }
// ---------------------------------------------------------------------------
//
// The dense base grid is materialized, tested against the level set, and
// pruned in one pass.  Peak block usage is therefore the FULL bounding-box
// grid even though only the body survives -- see the NCELLS_MAX note in the
// Makefile for the resolution this caps.
//
void CutFemSolver::buildMesh(void) {
  long t0 = nowUs();
  initializeBaseGrid();
  i32 nDense = hashTable.nKeys;

  femBlockActiveKernel<<<std::min(nDense, 65535), 128>>>(*this);
  cudaDeviceSynchronize();
  nBlocks = hashTable.nKeys;
  femPruneKernel<<<1024, 256>>>(*this);
  cudaDeviceSynchronize();
  sortBlocks();
  cudaDeviceSynchronize();

  const i32 nnb = (blockSize+1)*(blockSize+1)*(blockSize+1);
  if (phiBlk) cudaFree(phiBlk);
  if (nrmBlk) cudaFree(nrmBlk);
  cudaMallocManaged(&phiBlk, (size_t)hashTable.nKeys*nnb*sizeof(real));
  cudaMallocManaged(&nrmBlk, 3*(size_t)hashTable.nKeys*nnb*sizeof(real));
  femPhiKernel<<<std::min(hashTable.nKeys, 65535), 128>>>(*this);
  cudaDeviceSynchronize();

  meshMs = (nowUs() - t0)/1000.0;
  printf("mesh   : %d of %d background blocks are active (%.1f%%)\n",
         hashTable.nKeys, nDense, 100.0*hashTable.nKeys/std::max(1, nDense));
}

// ---------------------------------------------------------------------------
//  dofs: elements, nodes, neighbours, stabilized faces
// ---------------------------------------------------------------------------
void CutFemSolver::setupDofs(void) {
  long t0 = nowUs();
  freeMesh();

  const i32 nB  = hashTable.nKeys;
  const i32 nn1 = blockSize + 1;
  const i32 nnb = nn1*nn1*nn1;
  const real h  = cellSize();

  // ---- active elements ----------------------------------------------------
  struct Elem { i32 ci, cj, ck; real phi[8]; real nrm[24]; bool cut; };
  std::vector<Elem> elems;
  elems.reserve((size_t)nB*blockSizeTot/2);
  for (i32 b = 0; b < nB; b++) {
    u64 loc = bLocList[b];
    if (loc == kEmpty) continue;
    i32 lvl, ib, jb, kb; hDecode(loc, lvl, ib, jb, kb);
    const real *pb = &phiBlk[(size_t)b*nnb];
    for (i32 k = 0; k < blockSize; k++)
    for (i32 j = 0; j < blockSize; j++)
    for (i32 i = 0; i < blockSize; i++) {
      Elem E;
      real mn = 1e30, mx = -1e30;
      for (i32 n = 0; n < 8; n++) {
        i32 nn = (i + (n&1)) + nn1*((j + ((n>>1)&1)) + nn1*(k + ((n>>2)&1)));
        real p = pb[nn];
        E.phi[n] = p;
        for (i32 d = 0; d < 3; d++)
          E.nrm[3*n+d] = nrmBlk[3*((size_t)b*nnb + nn) + d];
        mn = fmin(mn, (double)p); mx = fmax(mx, (double)p);
      }
      if (mn >= 0) continue;                    // element misses Omega
      E.ci = ib*blockSize + i;
      E.cj = jb*blockSize + j;
      E.ck = kb*blockSize + k;
      E.cut = (mx > 0);
      elems.push_back(E);
    }
  }
  nElem = (i32)elems.size();
  if (nElem == 0) {
    printf("ERROR: no active elements -- Omega does not meet the background grid\n");
    exit(1);
  }

  // ---- drop free-floating components (rigid-body null modes) ---------------
  //
  // The one-pitch sector can contain a tiny detached piece -- a blade LE/TE tip
  // poking through a theta face, a stray platform corner -- that touches neither
  // the clamp nor, through the cyclic seam, a clamped component.  Such a piece
  // is a free rigid body: 6 null modes, and the centrifugal load has a nonzero
  // resultant on it, so b leaves range(A) and CG diverges (proven by the range
  // test: b = A*rand converges on the identical operator).  Union-find over face
  // adjacency -- a face joins two cells only where the trilinear phi goes
  // negative on it, and the theta seam wraps -- finds the components; any that
  // reaches no Dirichlet surface is removed here, before it can pollute the
  // dofs.  The pieces are negligible material (measured: <=16 of ~8000 cells).
  if (nCleanComp) {
    std::unordered_map<u64,i32> c2e;
    c2e.reserve((size_t)nElem*2);
    for (i32 e = 0; e < nElem; e++)
      c2e[cellKeyOf(elems[e].ci, elems[e].cj, elems[e].ck)] = e;
    std::vector<i32> par(nElem);
    for (i32 e = 0; e < nElem; e++) par[e] = e;
    std::function<i32(i32)> find = [&](i32 x){ while(par[x]!=x){par[x]=par[par[x]]; x=par[x];} return x; };
    auto uni = [&](i32 a,i32 b){ a=find(a); b=find(b); if(a!=b) par[a]=b; };
    static const i32 DIR6[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
    static const i32 FCORN[6][4] = {{0,2,4,6},{1,3,5,7},{0,1,4,5},{2,3,6,7},{0,1,2,3},{4,5,6,7}};
    for (i32 e = 0; e < nElem; e++)
      for (i32 f = 0; f < 6; f++) {
        i32 nj = elems[e].cj + DIR6[f][1];
        if (periodic) { if (nj < 0) nj = nThetaCells-1; else if (nj >= nThetaCells) nj = 0; }
        auto it = c2e.find(cellKeyOf(elems[e].ci+DIR6[f][0], nj, elems[e].ck+DIR6[f][2]));
        if (it == c2e.end()) continue;
        real mn = 1e30;
        for (i32 c = 0; c < 4; c++) mn = fmin(mn, elems[e].phi[FCORN[f][c]]);
        if (mn < 0) uni(e, it->second);
      }
    std::unordered_map<i32,char> cdir;
    for (i32 e = 0; e < nElem; e++)
      if (elems[e].cut) {
        real cx, cy, cz;
        ls.toPhys((elems[e].ci+(real)0.5)*h, (elems[e].cj+(real)0.5)*h,
                  (elems[e].ck+(real)0.5)*h, cx, cy, cz);
        if (prob.isDirichlet(cx, cy, cz)) cdir[find(e)] = 1;
      }
    std::vector<Elem> keep;
    keep.reserve(nElem);
    i32 nDrop = 0;
    for (i32 e = 0; e < nElem; e++) {
      if (cdir.count(find(e))) keep.push_back(elems[e]);
      else nDrop++;
    }
    if (nDrop) {
      printf("prune  : dropped %d free-floating element%s (no Dirichlet anchor -> "
             "rigid-body null modes)\n", nDrop, nDrop==1?"":"s");
      elems.swap(keep);
      nElem = (i32)elems.size();
    }
  }

  // ---- node numbering -----------------------------------------------------
  // sort + unique + binary search: deterministic, and faster than hashing at
  // these sizes.  One level everywhere, so every node is an unknown.
  std::vector<u64> keys((size_t)nElem*8);
  parFor(nElem, [&](i32 a, i32 b) {
    for (i32 e = a; e < b; e++) {
      const Elem &E = elems[e];
      for (i32 n = 0; n < 8; n++)
        keys[(size_t)8*e + n] = nodeKeyOf(E.ci + (n&1), E.cj + ((n>>1)&1), E.ck + ((n>>2)&1));
    }
  });
  std::vector<u64> nkey(keys);
  std::sort(nkey.begin(), nkey.end());
  nkey.erase(std::unique(nkey.begin(), nkey.end()), nkey.end());
  nNode = (i32)nkey.size();

  // ---- cyclic node map ----------------------------------------------------
  // The sector spans exactly nThetaCells cells, so the j = nThetaCells node
  // column is the geometric image of the j = 0 column under a rotation by one
  // pitch.  Those nodes are eliminated: each maps to its partner's real dof and
  // carries a flag saying the rotation applies.  With periodic off this is the
  // identity, so the same code path serves both.
  cudaMallocManaged(&nMap, (size_t)nNode*sizeof(i32));
  cudaMallocManaged(&nRot, (size_t)nNode*sizeof(i32));
  {
    std::vector<i32> realIdx(nNode, -1);
    nReal = 0;
    for (i32 n = 0; n < nNode; n++) {
      i32 I, J, K; nodeKeyDec(nkey[n], I, J, K);
      if (periodic && J == nThetaCells) continue;         // slave, assigned below
      realIdx[n] = nReal++;
      nMap[n] = realIdx[n];
      nRot[n] = 0;
    }
    i32 nSlave = 0, nOrphan = 0;
    for (i32 n = 0; n < nNode; n++) {
      if (realIdx[n] >= 0) continue;
      i32 I, J, K; nodeKeyDec(nkey[n], I, J, K);
      u64 pk = nodeKeyOf(I, 0, K);
      auto it = std::lower_bound(nkey.begin(), nkey.end(), pk);
      if (it == nkey.end() || *it != pk) {
        // no partner: the j = 0 column has no active element at this (I,K).
        // Keep it as its own dof rather than dropping the node.
        realIdx[n] = nReal++;
        nMap[n] = realIdx[n];
        nRot[n] = 0;
        nOrphan++;
        continue;
      }
      nMap[n] = realIdx[(i32)(it - nkey.begin())];
      nRot[n] = 1;
      nSlave++;
    }
    if (periodic)
      printf("cyclic : %d nodes tied across the pitch (%d unmatched kept free), "
             "%d -> %d dofs\n", nSlave, nOrphan, nNode, nReal);
  }

  cudaMallocManaged(&eNode,   (size_t)8*nElem*sizeof(i32));
  cudaMallocManaged(&eX0,     (size_t)3*nElem*sizeof(real));
  cudaMallocManaged(&eH,      (size_t)nElem*sizeof(real));
  cudaMallocManaged(&eCut,    (size_t)nElem*sizeof(i32));
  cudaMallocManaged(&eNbr,    (size_t)6*nElem*sizeof(i32));
  cudaMallocManaged(&nodeX,   (size_t)3*nNode*sizeof(real));
  cudaMallocManaged(&phiNode, (size_t)nNode*sizeof(real));

  // nodeX holds PHYSICAL positions (the VTK output needs them, and in
  // cylindrical mode the grid coordinates are not positions at all)
  parFor(nNode, [&](i32 a, i32 b) {
    for (i32 n = a; n < b; n++) {
      i32 I, J, K; nodeKeyDec(nkey[n], I, J, K);
      ls.toPhys(I*h, J*h, K*h, nodeX[3*n], nodeX[3*n+1], nodeX[3*n+2]);
    }
  });

  parFor(nElem, [&](i32 a, i32 b) {
    for (i32 e = a; e < b; e++) {
      const Elem &E = elems[e];
      eH[e] = h;
      eX0[3*e] = E.ci*h; eX0[3*e+1] = E.cj*h; eX0[3*e+2] = E.ck*h;
      eCut[e] = -1;
      for (i32 n = 0; n < 8; n++) {
        i32 id = (i32)(std::lower_bound(nkey.begin(), nkey.end(), keys[(size_t)8*e+n])
                       - nkey.begin());
        eNode[8*e+n] = id;
        phiNode[id]  = E.phi[n];
      }
    }
  });

  // ---- element-matrix bank ------------------------------------------------
  //
  // Cartesian: only CUT elements need a stored 24x24 -- every interior element
  // is the same cube, so they all share c_Kref scaled by h.
  //
  // Cylindrical: elements are curved (r, theta, z) bricks, so there is no shared
  // reference element and EVERY active element gets its own matrix.  The cut
  // kernel already handles that: its `sAllIn` branch integrates an uncut
  // element with 3x3x3 Gauss under the same isoparametric metric.
  const bool storeAll = (ls.coordMode != 0);
  std::vector<i32> cutList;
  for (i32 e = 0; e < nElem; e++) if (storeAll || elems[e].cut) cutList.push_back(e);
  nCut = (i32)cutList.size();
  nCutTrue = 0;
  for (i32 e = 0; e < nElem; e++) if (elems[e].cut) nCutTrue++;
  if (nCut) {
    cudaMallocManaged(&cutElem, (size_t)nCut*sizeof(i32));
    cudaMallocManaged(&cutPhi,  (size_t)8*nCut*sizeof(real));
    cudaMallocManaged(&cutNrm,  (size_t)24*nCut*sizeof(real));
    cudaMallocManaged(&cutFpred,(size_t)nCut*sizeof(real));
    cudaMallocManaged(&cutCoh,  (size_t)nCut*sizeof(real));
    cudaMallocManaged(&cutK,    (size_t)576*nCut*sizeof(real));
    cudaMallocManaged(&cutF,    (size_t)24*nCut*sizeof(real));
    for (i32 c = 0; c < nCut; c++) {
      i32 e = cutList[c];
      cutElem[c] = e;
      eCut[e] = c;
      for (i32 n = 0; n < 8; n++) cutPhi[8*c+n] = elems[e].phi[n];
      for (i32 t = 0; t < 24; t++) cutNrm[24*c+t] = elems[e].nrm[t];
    }
  }

  // ---- face neighbours ----------------------------------------------------
  std::unordered_map<u64,i32> cellToElem;
  cellToElem.reserve((size_t)nElem*2);
  for (i32 e = 0; e < nElem; e++)
    cellToElem[cellKeyOf(elems[e].ci, elems[e].cj, elems[e].ck)] = e;
  static const i32 DIR[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  parFor(nElem, [&](i32 a, i32 b) {
    for (i32 e = a; e < b; e++) {
      const Elem &E = elems[e];
      for (i32 f = 0; f < 6; f++) {
        i32 nj = E.cj + DIR[f][1];
        // periodic wrap in theta: the seam between cj = 0 and cj = nThetaCells-1
        // is a true interior face of the annulus, so the -theta neighbour of the
        // first column is the last column and vice versa.  Without this the seam
        // gets no neighbour -> no ghost penalty there, and the connectivity
        // check below would wrongly see the two halves as detached.
        if (periodic) {
          if (nj < 0) nj = nThetaCells - 1;
          else if (nj >= nThetaCells) nj = 0;
        }
        auto it = cellToElem.find(cellKeyOf(E.ci + DIR[f][0], nj, E.ck + DIR[f][2]));
        eNbr[6*e+f] = (it == cellToElem.end()) ? -1 : it->second;
      }
    }
  });

  // ---- connectivity + Dirichlet reachability ------------------------------
  //
  // The CG range test proved the load-case divergence is a rigid-body null
  // mode, not a sliver or an indefinite operator: b = A*rand converges, the
  // physical load does not.  A null mode appears when a connected component of
  // the active mesh carries NO Dirichlet boundary -- e.g. a blade slice that
  // enters the sector above the platform and touches neither the clamp nor,
  // through the cyclic seam, a clamped component.  Such a piece has 6 rigid-body
  // modes; the centrifugal load has a nonzero net force on it, so b leaves
  // range(A) and CG diverges.  Union-find over face adjacency (the periodic seam
  // included above) finds them; a face joins two elements only where it meets
  // Omega, so a hairline contact does not glue two bodies.
  {
    std::vector<i32> par(nElem);
    for (i32 e = 0; e < nElem; e++) par[e] = e;
    std::function<i32(i32)> find = [&](i32 a2) {
      while (par[a2] != a2) { par[a2] = par[par[a2]]; a2 = par[a2]; }
      return a2;
    };
    auto uni = [&](i32 a2, i32 b2) { a2 = find(a2); b2 = find(b2); if (a2 != b2) par[a2] = b2; };
    // face f of element e shares 4 corners; the trilinear phi restricted to the
    // face is negative somewhere iff min over those 4 nodal phi < 0
    static const i32 FCORN[6][4] = {{0,2,4,6},{1,3,5,7},{0,1,4,5},{2,3,6,7},{0,1,2,3},{4,5,6,7}};
    for (i32 e = 0; e < nElem; e++)
      for (i32 f = 0; f < 6; f++) {
        i32 nb = eNbr[6*e+f];
        if (nb < 0) continue;
        real mn = 1e30;
        for (i32 c = 0; c < 4; c++) mn = fmin(mn, elems[e].phi[FCORN[f][c]]);
        if (mn < 0) uni(e, nb);            // face carries material -> bodies joined
      }
    // per-component: size, and whether any element carries Dirichlet boundary
    std::unordered_map<i32,i32> csz;
    std::unordered_map<i32,char> cdir;
    for (i32 e = 0; e < nElem; e++) {
      i32 r = find(e);
      csz[r]++;
      // Dirichlet interface present in this element?  cheap host test at the
      // element centre against the same predicate the kernel uses
      if (elems[e].cut) {
        real cx, cy, cz;
        ls.toPhys(eX0[3*e] + h/2, eX0[3*e+1] + h/2, eX0[3*e+2] + h/2, cx, cy, cz);
        if (prob.isDirichlet(cx, cy, cz)) cdir[r] = 1;
      }
    }
    nComp = (i32)csz.size();
    nCompFree = 0; compMax = 0; freeMax = 0;
    for (auto &kv : csz) {
      compMax = std::max(compMax, kv.second);
      if (!cdir.count(kv.first)) { nCompFree++; freeMax = std::max(freeMax, kv.second); }
    }
  }

  // ---- stabilized faces  F_h(dOmega)  (2.16) ------------------------------
  //
  // Interior faces belonging to an element that meets the boundary.  Each face
  // is visited once, from the lower element of the pair.  fCoef folds the h
  // powers: j_h = h^3 q^T M q, and a_h adds gamma_a h^-2 j_h (2.20), so the
  // coefficient is gamma_a h.  stabMode 1 keeps that only where the cut
  // element carries Dirichlet data and uses the weaker gamma_a j_h (coefficient
  // gamma_a h^3) elsewhere (Remark 2.1).
  //
  std::vector<i32>  fn;
  std::vector<real> fc;
  if (gammaA > 0) {
    for (i32 e = 0; e < nElem; e++) {
      for (i32 d = 0; d < 3; d++) {
        i32 nb = eNbr[6*e + 2*d + 1];                 // +d neighbour
        if (nb < 0) continue;
        if (eCut[e] < 0 && eCut[nb] < 0) continue;    // neither side is cut
        bool strong = true;
        if (stabMode == 1) {
          strong = false;
          for (i32 t = 0; t < 2; t++) {
            i32 ee = t ? nb : e;
            if (eCut[ee] < 0) continue;
            if (prob.isDirichlet(eX0[3*ee] + h/2, eX0[3*ee+1] + h/2, eX0[3*ee+2] + h/2))
              strong = true;
          }
        }
        fc.push_back(strong ? gammaA*h : gammaA*h*h*h);
        i32 d1 = (d+1)%3, d2 = (d+2)%3;
        for (i32 m = 0; m < 4; m++) {                 // m = b + 2c
          i32 bit[3];
          bit[d1] = m & 1; bit[d2] = (m >> 1) & 1;
          bit[d] = 0;  i32 nFar  = bit[0] | (bit[1]<<1) | (bit[2]<<2);
          bit[d] = 1;  i32 nOnF  = bit[0] | (bit[1]<<1) | (bit[2]<<2);
          fn.push_back(eNode[8*e  + nFar]);           // far side of the left element
          fn.push_back(eNode[8*e  + nOnF]);           // on the face
          fn.push_back(eNode[8*nb + nOnF]);           // far side of the right element
        }
      }
    }
  }
  nFace = (i32)fc.size();
  if (nFace) {
    cudaMallocManaged(&fNode, (size_t)12*nFace*sizeof(i32));
    cudaMallocManaged(&fCoef, (size_t)nFace*sizeof(real));
    // fn was pushed interleaved (far-L, face, far-R) per m; de-interleave into
    // the [far-L x4 | face x4 | far-R x4] layout the kernel expects
    for (i32 f = 0; f < nFace; f++)
      for (i32 m = 0; m < 4; m++) {
        fNode[12*f + m]     = fn[(size_t)12*f + 3*m];
        fNode[12*f + 4 + m] = fn[(size_t)12*f + 3*m + 1];
        fNode[12*f + 8 + m] = fn[(size_t)12*f + 3*m + 2];
      }
    memcpy(fCoef, fc.data(), (size_t)nFace*sizeof(real));
  }

  allocVectors();
  cudaDeviceSynchronize();
  setupMs = (nowUs() - t0)/1000.0;
}

void CutFemSolver::allocVectors(void) {
  size_t n = (size_t)3*nReal*sizeof(real);
  size_t m = (size_t)3*nNode*sizeof(real);
  cudaMallocManaged(&uh,   n);
  cudaMallocManaged(&rhs,  n);
  cudaMallocManaged(&diag, n);
  cudaMallocManaged(&cgR,  n);
  cudaMallocManaged(&cgZ,  n);
  cudaMallocManaged(&cgP,  n);
  cudaMallocManaged(&cgQ,  n);
  cudaMallocManaged(&xn,   m);
  cudaMallocManaged(&yn,   m);
  cudaMemset(uh, 0, n);
}

void CutFemSolver::freeMesh(void) {
  cudaDeviceSynchronize();
  void *ptrs[] = {eNode, eX0, eH, eCut, eNbr, cutElem, cutPhi, cutNrm, cutFpred,
                  cutCoh, cutK, cutF, fNode, fCoef, nodeX, phiNode, nMap, nRot,
                  uh, rhs, diag, cgR, cgZ, cgP, cgQ, xn, yn};
  for (void *p : ptrs) if (p) cudaFree(p);
  eNode = eCut = eNbr = cutElem = fNode = nMap = nRot = nullptr;
  eX0 = eH = cutPhi = cutNrm = cutFpred = cutCoh = cutK = cutF = fCoef = nodeX = phiNode = nullptr;
  uh = rhs = diag = cgR = cgZ = cgP = cgQ = xn = yn = nullptr;
  nElem = nCut = nFace = nNode = nReal = 0;
}

// ---------------------------------------------------------------------------
//  assembly
// ---------------------------------------------------------------------------
void CutFemSolver::assemble(void) {
  long t0 = nowUs();
  const i32 nN = 3*nNode, nR = 3*nReal, BS = 256, GS = 1024;

  cudaMemset(acc, 0, 8*sizeof(double));
  slivMin = slivMinTheta = 1e30;
  cudaMemset(fracHist, 0, 16*sizeof(i32));
  cudaDeviceSynchronize();
  if (nCut) femSliverKernel<<<GS,BS>>>(*this);    // 1-jet quality, before the heavy kernel
  cudaDeviceSynchronize();
  if (nCut) femCutElemKernel<<<std::min(nCut, 65535), FEM_EPB>>>(*this);
  cudaDeviceSynchronize();

  // 1-jet geometric-quality summary.  The paper (Wichrowski, CAMWA 2026) makes
  // the point sharply: ghost penalty gives stability "independent of the
  // location of the interface", i.e. slivers are STABILIZED, not resolved away
  // -- and the census confirms the smallest cut fraction gets WORSE under
  // refinement.  So the useful resolution question is not "are there slivers"
  // but "is each FEATURE resolved": a creased cut cell (its 8 corner normals
  // disagree) straddles a kink in phi -- two CSG surfaces meeting, or a feature
  // thinner than h -- where the piecewise-planar cut reconstruction is invalid.
  nCrease = 0;
  double cohMin = 1;
  for (i32 c = 0; c < nCut; c++) {
    if (cutCoh[c] < creaseCos) nCrease++;
    cohMin = std::min(cohMin, (double)cutCoh[c]);
  }
  cohWorst = cohMin;

  // load: assemble in node space, then restrict through the cyclic constraint
  femSetKernel<<<GS,BS>>>(yn, 0, nN);
  cudaDeviceSynchronize();
  femFullLoadKernel<<<GS,BS>>>(*this);
  if (nCut) femCutLoadKernel<<<GS,BS>>>(*this);
  femSetKernel<<<GS,BS>>>(rhs, 0, nR);
  cudaDeviceSynchronize();
  femRestrictKernel<<<GS,BS>>>(*this, rhs);

  // Jacobi diagonal, likewise
  femSetKernel<<<GS,BS>>>(yn, 0, nN);
  cudaDeviceSynchronize();
  femDiagElemKernel<<<GS,BS>>>(*this);
  if (nFace) femDiagFaceKernel<<<GS,BS>>>(*this);
  femSetKernel<<<GS,BS>>>(diag, 0, nR);
  cudaDeviceSynchronize();
  femDiagRestrictKernel<<<GS,BS>>>(*this, diag);
  cudaDeviceSynchronize();

  // The Jacobi preconditioner needs a positive diagonal.  A non-positive entry
  // means some dof has no (or negative) self-stiffness -- a sliver the ghost
  // penalty failed to reach, or a Nitsche term overwhelming a tiny cut volume --
  // and CG will diverge rather than converge slowly, so it is worth naming.
  {
    double dmin = 1e300, dmax = -1e300;
    i32 nBad = 0;
    for (i32 i = 0; i < nR; i++) {
      double d = diag[i];
      if (d <= 0) nBad++;
      dmin = std::min(dmin, d); dmax = std::max(dmax, d);
    }
    diagMin = dmin; diagMax = dmax; nDiagBad = nBad;
  }

  volOmega  = acc[0];
  areaGamma = acc[1];
  areaDirich = acc[6];
  assembleMs = (nowUs() - t0)/1000.0;
}

// ---------------------------------------------------------------------------
//  y = A x
// ---------------------------------------------------------------------------
void CutFemSolver::applyA(const real *x, real *y) {
  const i32 BS = 256, GS = 1024;
  femProlongKernel<<<GS,BS>>>(*this, x);          // xn = P x, yn = 0
  cudaDeviceSynchronize();
  femElemApplyKernel<<<GS,BS>>>(*this, xn, yn);
  if (nFace) femFaceApplyKernel<<<GS,BS>>>(*this, xn, yn);
  femSetKernel<<<GS,BS>>>(y, 0, 3*nReal);
  cudaDeviceSynchronize();
  femRestrictKernel<<<GS,BS>>>(*this, y);         // y = P^T yn
  cudaDeviceSynchronize();
}

double CutFemSolver::dot(const real *a, const real *b, i32 n) {
  cudaMemset(acc + 7, 0, sizeof(double));
  femDotKernel<<<1024,256>>>(a, b, n, acc + 7);
  cudaDeviceSynchronize();
  return acc[7];
}

// Probe A for definiteness.  CG with an SPD operator and a positive-definite
// preconditioner cannot diverge, so if it does the operator itself is the thing
// to interrogate -- and a Rayleigh quotient on random vectors says so directly.
void CutFemSolver::spdProbe(i32 nTrial) {
  const i32 n = 3*nReal;
  std::vector<real> x(n);
  double worst = 1e300;
  unsigned s0 = 12345;
  for (i32 t = 0; t < nTrial; t++) {
    for (i32 i = 0; i < n; i++) {
      s0 = s0*1664525u + 1013904223u;
      x[i] = (real)((double)(s0 >> 8)/8388608.0 - 1.0);
    }
    cudaMemcpy(cgP, x.data(), (size_t)n*sizeof(real), cudaMemcpyDefault);
    applyA(cgP, cgQ);
    double xx = dot(cgP, cgP, n), xax = dot(cgP, cgQ, n);
    worst = std::min(worst, xax/xx);
  }
  printf("spd    : min Rayleigh quotient over %d random vectors = %.6e%s\n",
         nTrial, worst, worst < 0 ? "   <-- A IS INDEFINITE" : "");
}

// ---------------------------------------------------------------------------
//  preconditioned CG (diagonal scaling, the paper's (5.4))
// ---------------------------------------------------------------------------
void CutFemSolver::solveCg(void) {
  long t0 = nowUs();
  const i32 n = 3*nReal, BS = 256, GS = 1024;

  // Range test (--rangetest): replace b by A*x_rand, which is in range(A) by
  // construction.  If CG then converges but the physical load diverges, the
  // OPERATOR is fine and the load's b is out of range -- a rigid-body null mode
  // from a component the boundary conditions do not anchor (the workflow's
  // diagnosis).  If it still diverges, the operator itself is indefinite.
  if (rangeTest) {
    std::vector<real> xr(n);
    unsigned s0 = 987654321u;
    for (i32 i = 0; i < n; i++) { s0 = s0*1664525u + 1013904223u; xr[i] = (real)((double)(s0>>8)/8388608.0 - 1.0); }
    cudaMemcpy(cgP, xr.data(), (size_t)n*sizeof(real), cudaMemcpyDefault);
    applyA(cgP, rhs);
    printf("rangetest: b <- A*rand (in range(A) by construction)\n");
  }

  femSetKernel<<<GS,BS>>>(uh, 0, n);
  cudaDeviceSynchronize();
  cudaMemcpy(cgR, rhs, (size_t)n*sizeof(real), cudaMemcpyDefault);   // r = b - A*0
  femJacobiKernel<<<GS,BS>>>(cgZ, cgR, diag, n);
  cudaDeviceSynchronize();
  cudaMemcpy(cgP, cgZ, (size_t)n*sizeof(real), cudaMemcpyDefault);

  {   // a non-finite load makes CG "diverge" in a way no amount of
      // preconditioning explains, so name it before iterating
    i32 nBad = 0;
    double bmax = 0;
    for (i32 i = 0; i < n; i++) {
      double v = rhs[i];
      if (!std::isfinite(v)) nBad++;
      else bmax = std::max(bmax, std::fabs(v));
    }
    if (nBad) printf("WARNING: %d of %d load entries are not finite\n", nBad, n);
    printf("rhs    : max |b| = %.6e%s\n", bmax, nBad ? "  (finite entries only)" : "");
  }
  double rz = dot(cgR, cgZ, n);
  double b0 = sqrt(dot(cgR, cgR, n));
  if (b0 == 0) { cgIters = 0; cgRes = 0; solveMs = 0; return; }

  i32 it = 0;
  double rn = b0;
  for (; it < cgMaxIt; it++) {
    applyA(cgP, cgQ);
    double pq = dot(cgP, cgQ, n);
    if (!(pq > 0)) { printf("WARNING: CG breakdown, p'Ap = %.3e\n", pq); break; }
    real alpha = (real)(rz/pq);
    femAxpyKernel<<<GS,BS>>>(uh,  cgP, alpha, n);
    femAxpyKernel<<<GS,BS>>>(cgR, cgQ, -alpha, n);
    cudaDeviceSynchronize();
    rn = sqrt(dot(cgR, cgR, n));
    if (rn <= cgTol*b0) { it++; break; }
    femJacobiKernel<<<GS,BS>>>(cgZ, cgR, diag, n);
    cudaDeviceSynchronize();
    double rz1 = dot(cgR, cgZ, n);
    real beta = (real)(rz1/rz);
    rz = rz1;
    femXpayKernel<<<GS,BS>>>(cgP, cgZ, beta, n);        // p = z + beta p
    cudaDeviceSynchronize();
  }
  cgIters = it;
  cgRes = rn/b0;
  solveMs = (nowUs() - t0)/1000.0;
}

// ---------------------------------------------------------------------------
//  errors
// ---------------------------------------------------------------------------
void CutFemSolver::computeErrors(void) {
  if (prob.caseId != CASE_MMS && prob.caseId != CASE_MMS_CYL) return;
  cudaMemset(acc, 0, 8*sizeof(double));
  femProlongKernel<<<1024,256>>>(*this, uh);
  cudaDeviceSynchronize();
  femErrorKernel<<<std::min(nElem, 65535), FEM_EPB>>>(*this);
  cudaDeviceSynchronize();
  errL2      = sqrt(acc[2]);
  normL2     = sqrt(acc[3]);
  errEnergy  = sqrt(acc[4]);
  normEnergy = sqrt(acc[5]);
}

// ---------------------------------------------------------------------------
//  driver / report
// ---------------------------------------------------------------------------
void CutFemSolver::run(void) {
  if (femMethod == 1) { runSbm(); return; }  // shifted boundary (CutFemSbm.cu)
  if (femOrder > 1) { runQp(); return; }   // higher-order host path (CutFemQp.cu)
  initialize();
  buildMesh();
  setupDofs();
  assemble();
  if (spdCheck) spdProbe();
  solveCg();
  computeErrors();
  report();
}

void CutFemSolver::report(void) {
  printf("active : %d elements (%d cut, %.1f%%), %d stabilized faces, "
         "%d stored element matrices\n",
         nElem, nCutTrue, 100.0*nCutTrue/std::max(1,nElem), nFace, nCut);
  printf("dofs   : %d nodes -> %d unknowns   h = %.6g\n",
         nNode, 3*nReal, (double)cellSize());
  printf("connect: %d component%s (largest %d of %d elems)%s\n",
         nComp, nComp==1?"":"s", compMax, nElem,
         nCompFree ? "" : "  all reach a Dirichlet boundary");
  if (nCompFree)
    printf("         WARNING: %d component%s carry NO Dirichlet boundary (largest %d"
           " elems) -> rigid-body null modes; the centrifugal load is out of range(A)"
           " and CG will diverge\n", nCompFree, nCompFree==1?"":"s", freeMax);
  printf("geom   : |Omega_h| = %.8g", volOmega);
  if (volExact > 0)  printf("   exact %.8g   err %.3e (%.3f%%)",
                            volExact, volOmega-volExact,
                            100.0*fabs(volOmega-volExact)/volExact);
  printf("\n         |Gamma_h| = %.8g", areaGamma);
  if (areaExact > 0) printf("   exact %.8g   err %.3e (%.3f%%)",
                            areaExact, areaGamma-areaExact,
                            100.0*fabs(areaGamma-areaExact)/areaExact);
  printf("   Dirichlet %.6g (%.2f%% of the surface)%s\n", areaDirich,
         100.0*areaDirich/std::max(1e-30, areaGamma),
         areaDirich <= 0 ? "   <-- NOTHING IS CLAMPED: 6 rigid-body null modes" : "");
  if (nDiagBad)
    printf("WARNING: %d of %d diagonal entries are non-positive (min %.3e) --\n"
           "         the Jacobi preconditioner is invalid there\n",
           nDiagBad, 3*nReal, diagMin);
  printf("diag   : %.3e .. %.3e  (ratio %.2e)\n", diagMin, diagMax,
         diagMax/std::max(1e-300, diagMin));
  {
    printf("sliver : cut-volume-fraction histogram (decade : count)\n        ");
    i32 tot = 0;
    for (i32 b = 0; b < 16; b++) tot += fracHist[b];
    for (i32 b = 0; b < 13; b++)
      if (fracHist[b]) printf(" 1e-%d:%d", b+1, fracHist[b]);
    printf("   (%d cut cells)\n", tot);
  }
  if (slivMinTheta > 1e29)
    printf("sliver : smallest cut-volume fraction %.3e; none on the theta faces\n",
           slivMin);
  else
    printf("sliver : smallest cut-volume fraction %.3e; on the theta faces %.3e\n",
           slivMin, slivMinTheta);
  printf("resolve: %d of %d cut cells CREASED (corner normals disagree > %.0f deg;"
         " worst coherence %.3f) -- %.1f%% under-resolved\n",
         nCrease, nCut, acos((double)creaseCos)*180/PI, cohWorst,
         100.0*nCrease/std::max(1,nCut));
  printf("solve  : CG %d iters, rel res %.2e   (mesh %.0f ms, dofs %.0f ms, "
         "assemble %.0f ms, solve %.0f ms)\n",
         cgIters, cgRes, meshMs, setupMs, assembleMs, solveMs);
  if ((prob.caseId == CASE_MMS || prob.caseId == CASE_MMS_CYL) && normL2 > 0)
    printf("error  : L2 %.6e (rel %.4e)   energy %.6e (rel %.4e)\n",
           errL2, errL2/normL2, errEnergy, errEnergy/normEnergy);
}

// ---------------------------------------------------------------------------
//  output
// ---------------------------------------------------------------------------
//
// The active elements as a VTK unstructured grid.  Displacement is a point
// vector (Warp By Vector shows the deformed body); phi lets ParaView clip to
// Omega, and the cell arrays flag the cut elements.
//
void CutFemSolver::writeVtu(const char *fileName) {
  femProlongKernel<<<1024,256>>>(*this, uh);      // node-space displacement
  cudaDeviceSynchronize();
  // VTK_HEXAHEDRON ordering vs our n = a + 2b + 4c numbering
  static const i32 VTKHEX[8] = {0,1,3,2,4,5,7,6};

  std::ofstream os(fileName, std::ios::binary);
  os << "<?xml version=\"1.0\"?>\n"
     << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
     << "  <UnstructuredGrid>\n    <Piece NumberOfPoints=\"" << nNode
     << "\" NumberOfCells=\"" << nElem << "\">\n";

  os << "      <Points>\n        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (i32 n = 0; n < nNode; n++)
    os << (float)nodeX[3*n] << " " << (float)nodeX[3*n+1] << " "
       << (float)nodeX[3*n+2] << "\n";
  os << "        </DataArray>\n      </Points>\n";

  os << "      <PointData Vectors=\"u\">\n"
     << "        <DataArray type=\"Float32\" Name=\"u\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (i32 n = 0; n < nNode; n++)
    os << (float)xn[3*n] << " " << (float)xn[3*n+1] << " " << (float)xn[3*n+2] << "\n";
  os << "        </DataArray>\n"
     << "        <DataArray type=\"Float32\" Name=\"phi\" format=\"ascii\">\n";
  for (i32 n = 0; n < nNode; n++) os << (float)phiNode[n] << "\n";
  os << "        </DataArray>\n      </PointData>\n";

  os << "      <CellData Scalars=\"vonMises\">\n"
     << "        <DataArray type=\"Float32\" Name=\"vonMises\" format=\"ascii\">\n";
  for (i32 e = 0; e < nElem; e++) {
    real G[8][3]; q1Grad((real)0.5, (real)0.5, (real)0.5, eH[e], G);
    real gu[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for (i32 m = 0; m < 8; m++) {
      i32 nd = eNode[8*e+m];
      // xn is the NODE-space displacement (length 3*nNode, prolongated above);
      // uh is the REAL-dof vector (length 3*nReal < 3*nNode under the cyclic
      // tie), so indexing it by a node id overreads and segfaults at scale.
      for (i32 i = 0; i < 3; i++)
        for (i32 j = 0; j < 3; j++) gu[i][j] += G[m][j]*xn[3*nd+i];
    }
    os << (float)vonMises(gu, prob.mu, prob.lam) << "\n";
  }
  os << "        </DataArray>\n"
     << "        <DataArray type=\"Int32\" Name=\"cut\" format=\"ascii\">\n";
  for (i32 e = 0; e < nElem; e++) os << (eCut[e] >= 0 ? 1 : 0) << "\n";
  os << "        </DataArray>\n      </CellData>\n";

  os << "      <Cells>\n        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
  for (i32 e = 0; e < nElem; e++) {
    for (i32 m = 0; m < 8; m++) os << eNode[8*e + VTKHEX[m]] << " ";
    os << "\n";
  }
  os << "        </DataArray>\n        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
  for (i32 e = 0; e < nElem; e++) os << 8*(e+1) << "\n";
  os << "        </DataArray>\n        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (i32 e = 0; e < nElem; e++) os << "12\n";
  os << "        </DataArray>\n      </Cells>\n    </Piece>\n"
     << "  </UnstructuredGrid>\n</VTKFile>\n";
}

//
// The reconstructed interface itself -- the CutFEM answer ON the body surface,
// not on the voxel staircase.  Marching tets is re-run on the host so the raw
// triangles (not just quadrature points) are available.
//
void CutFemSolver::writeSurfaceVtu(const char *fileName) {
  femProlongKernel<<<1024,256>>>(*this, uh);
  cudaDeviceSynchronize();
  std::vector<float> P, U, VM;
  i32 nTri = 0;
  const i32 maxt = 6*cutSub*cutSub*cutSub*2;
  std::vector<real> qv(4*12*maxt), qs(7*6*maxt), tri(9*maxt);

  for (i32 c = 0; c < nCut; c++) {
    i32 e = cutElem[c];
    real h = eH[e];
    CutQuadBuf B;
    B.qv = qv.data(); B.qs = qs.data(); B.maxv = (i32)qv.size()/4; B.maxs = (i32)qs.size()/7;
    B.tri = tri.data(); B.maxt = maxt;
    real phiN[8];
    for (i32 n = 0; n < 8; n++) phiN[n] = cutPhi[8*c+n];
    cutQuadrature(B, phiN, cutSub, h);

    real ul[24];
    for (i32 m = 0; m < 8; m++)
      for (i32 i = 0; i < 3; i++) ul[3*m+i] = xn[3*eNode[8*e+m]+i];

    for (i32 t = 0; t < B.nt; t++) {
      for (i32 v = 0; v < 3; v++) {
        const real *r = B.tri + 9*t + 3*v;
        real N[8], G[8][3];
        q1Shape(r[0], r[1], r[2], N);
        q1Grad(r[0], r[1], r[2], h, G);
        real u[3] = {0,0,0}, gu[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        for (i32 m = 0; m < 8; m++)
          for (i32 i = 0; i < 3; i++) {
            u[i] += N[m]*ul[3*m+i];
            for (i32 j = 0; j < 3; j++) gu[i][j] += G[m][j]*ul[3*m+i];
          }
        real Xp[3];
        ls.toPhys(eX0[3*e]   + h*r[0],
                  eX0[3*e+1] + h*r[1],
                  eX0[3*e+2] + h*r[2], Xp[0], Xp[1], Xp[2]);
        for (i32 d = 0; d < 3; d++) P.push_back((float)Xp[d]);
        for (i32 d = 0; d < 3; d++) U.push_back((float)u[d]);
        VM.push_back((float)vonMises(gu, prob.mu, prob.lam));
      }
      nTri++;
    }
  }

  std::ofstream os(fileName, std::ios::binary);
  os << "<?xml version=\"1.0\"?>\n"
     << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
     << "  <UnstructuredGrid>\n    <Piece NumberOfPoints=\"" << 3*nTri
     << "\" NumberOfCells=\"" << nTri << "\">\n";
  os << "      <Points>\n        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (size_t i = 0; i < P.size(); i += 3) os << P[i] << " " << P[i+1] << " " << P[i+2] << "\n";
  os << "        </DataArray>\n      </Points>\n";
  os << "      <PointData Vectors=\"u\" Scalars=\"vonMises\">\n"
     << "        <DataArray type=\"Float32\" Name=\"u\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (size_t i = 0; i < U.size(); i += 3) os << U[i] << " " << U[i+1] << " " << U[i+2] << "\n";
  os << "        </DataArray>\n"
     << "        <DataArray type=\"Float32\" Name=\"vonMises\" format=\"ascii\">\n";
  for (float v : VM) os << v << "\n";
  os << "        </DataArray>\n      </PointData>\n";
  os << "      <Cells>\n        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
  for (i32 t = 0; t < nTri; t++) os << 3*t << " " << 3*t+1 << " " << 3*t+2 << "\n";
  os << "        </DataArray>\n        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
  for (i32 t = 0; t < nTri; t++) os << 3*(t+1) << "\n";
  os << "        </DataArray>\n        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (i32 t = 0; t < nTri; t++) os << "5\n";
  os << "        </DataArray>\n      </Cells>\n    </Piece>\n"
     << "  </UnstructuredGrid>\n</VTKFile>\n";
  printf("  surface: %d interface triangles\n", nTri);
}
