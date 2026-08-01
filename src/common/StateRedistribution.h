#ifndef COMMON_STATEREDISTRIBUTION_H
#define COMMON_STATEREDISTRIBUTION_H

// ---------------------------------------------------------------------------
//  STATE REDISTRIBUTION (SRD) -- the small-cell fix for explicit cut-cell
//  schemes.
//
//  Berger & Giuliani, "A state redistribution algorithm for finite volume
//  schemes on cut cell meshes", JCP 428 (2021) 109820; high-order DG treatment
//  and the energy-stability proof in Taylor, Wilcox & Chan, "An Energy Stable
//  High-Order Cut Cell Discontinuous Galerkin Method with State Redistribution
//  for Wave Propagation", arXiv:2404.06630 (2024).
//
//  THE PROBLEM.  A cut cell can be an arbitrarily small fraction of the
//  background cell, and an explicit scheme's stable step scales with that
//  fraction -- a volume ratio of ~1000 is routine, which is fatal.
//
//  THE FIX, in three steps:
//    1. flag every cut element whose volume is below a fraction of the
//       background cell volume (the papers use 1/2),
//    2. grow a MERGE NEIGHBOURHOOD M_k around each flagged element until the
//       neighbourhood's total volume clears the threshold, and compute a single
//       degree-N polynomial over the union by a weighted L2 projection Pi_k,
//    3. set the new solution on element k to the AVERAGE of every projection
//       that element contributed to:
//
//           (S u)|_{D^k} = 1/|C_k| SUM_{j in C_k} (Pi_j u)|_{D^k},
//           C_k = { j : k in M_j }.
//
//  The averaging in step 3 is what distinguishes SRD from plain cell merging /
//  linking: merging has to pick an order and the answer depends on it, whereas
//  averaging over all neighbourhoods an element belongs to is order-free.
//
//  WHY IT IS SAFE.  S is CONTRACTIVE in L2 (Taylor et al. Thm 2.1), so by the
//  Nordstrom-Winters filter argument an energy-stable scheme stays energy
//  stable under it.  It is also conservative and preserves polynomials of
//  degree N exactly, so it costs no formal order.
//
//  SOLVER-AGNOSTIC by construction: it needs element volumes, face neighbours
//  and a cut quadrature rule, and nothing about the equation being solved.  For
//  a static mesh the operator is built ONCE as preprocessing and only applied
//  per step.
// ---------------------------------------------------------------------------

#include <cmath>
#include <vector>

#include "Util.cuh"
#include "SayeQuad.h"

// One element of the cut mesh, as SRD needs to see it.
struct SrdElem {
  i32    nbr[6];      // face neighbours (element indices), -1 where absent
  double vol;         // FLUID volume of this element
  double x0[3];       // lower corner, physical
  double h;           // background cell size
  i32    qOff, qN;    // slice of the shared quadrature pool (REFERENCE coords)
};

// Total-degree-N monomials in 3-D, scaled about a neighbourhood centroid.
// Scaling matters: raw physical monomials over a multi-cell neighbourhood give
// a Vandermonde/mass matrix that is numerically singular by degree 3.
struct SrdBasis {
  i32 N, nb;
  double c[3], s;                        // centroid and scale
  static i32 count(i32 N) { return (N+1)*(N+2)*(N+3)/6; }
  void init(i32 N_, const double cc[3], double ss) {
    N = N_; nb = count(N); c[0]=cc[0]; c[1]=cc[1]; c[2]=cc[2]; s = (ss>0)?ss:1.0;
  }
  void eval(const double X[3], double *psi) const {
    double u=(X[0]-c[0])/s, v=(X[1]-c[1])/s, w=(X[2]-c[2])/s;
    i32 m=0;
    for (i32 d=0; d<=N; d++)
      for (i32 i=d; i>=0; i--)
        for (i32 j=d-i; j>=0; j--) {
          i32 k=d-i-j;
          double t=1;
          for (i32 a=0;a<i;a++) t*=u;
          for (i32 a=0;a<j;a++) t*=v;
          for (i32 a=0;a<k;a++) t*=w;
          psi[m++]=t;
        }
  }
};

// Cholesky solve for a small dense SPD system (in place on a copy).
inline bool srdSolveSPD(std::vector<double> &A, std::vector<double> &b, i32 n) {
  for (i32 j=0;j<n;j++) {
    double d=A[(size_t)j*n+j];
    for (i32 q=0;q<j;q++) d-=A[(size_t)j*n+q]*A[(size_t)j*n+q];
    if (d<=1e-300) return false;
    d=sqrt(d); A[(size_t)j*n+j]=d;
    for (i32 i=j+1;i<n;i++) {
      double s=A[(size_t)i*n+j];
      for (i32 q=0;q<j;q++) s-=A[(size_t)i*n+q]*A[(size_t)j*n+q];
      A[(size_t)i*n+j]=s/d;
    }
  }
  for (i32 i=0;i<n;i++){ double s=b[i]; for(i32 q=0;q<i;q++) s-=A[(size_t)i*n+q]*b[q]; b[i]=s/A[(size_t)i*n+i]; }
  for (i32 i=n-1;i>=0;i--){ double s=b[i]; for(i32 q=i+1;q<n;q++) s-=A[(size_t)q*n+i]*b[q]; b[i]=s/A[(size_t)i*n+i]; }
  return true;
}

struct SrdOperator {
  i32 nElem = 0, N = 0, nb = 0;
  double volFrac = 0.5;                       // "small" threshold, in background volumes
  std::vector<std::vector<i32>> M;            // merge neighbourhood per element
  std::vector<i32>              Ccnt;         // |C_k|
  std::vector<SrdBasis>         basis;        // one per neighbourhood
  std::vector<std::vector<double>> chol;      // factored neighbourhood mass matrix
  std::vector<char>             trivial;      // M_k == {k}: nothing to do

  // ---- step 1+2: flag small elements and grow merge neighbourhoods --------
  // GREEDY BY VOLUME, following Taylor et al. 6.2 rather than the conventional
  // "merge along the wall normal": for each flagged element repeatedly absorb
  // the face neighbour that adds the most volume, until the neighbourhood
  // clears the threshold.  On a Cartesian background this naturally prefers
  // whole uncut cells, which is what a normal-based rule would pick anyway.
  void buildNeighborhoods(const std::vector<SrdElem> &e) {
    nElem = (i32)e.size();
    M.assign(nElem, {}); trivial.assign(nElem, 1);
    const double vFull = e.empty() ? 1.0 : e[0].h*e[0].h*e[0].h;
    const double vTarget = volFrac * vFull;
    for (i32 k=0;k<nElem;k++) {
      M[k].push_back(k);
      if (e[k].vol >= vTarget) continue;       // healthy: M_k = {k}
      double v = e[k].vol;
      std::vector<char> in(nElem, 0); in[k]=1;
      while (v < vTarget) {
        i32 best=-1; double bestV=0;
        for (i32 m : M[k])
          for (i32 f=0; f<6; f++) {
            i32 nn=e[m].nbr[f];
            if (nn<0 || in[nn]) continue;
            if (e[nn].vol > bestV) { bestV=e[nn].vol; best=nn; }
          }
        if (best<0) break;                     // no more neighbours to absorb
        in[best]=1; M[k].push_back(best); v += e[best].vol;
      }
      trivial[k] = (M[k].size()==1);
    }
    // |C_k| = how many neighbourhoods element k belongs to (>= 1, itself)
    Ccnt.assign(nElem, 0);
    for (i32 k=0;k<nElem;k++) for (i32 j : M[k]) Ccnt[j]++;
  }

  // ---- factor the weighted neighbourhood mass matrices --------------------
  //   (u,v)_{M_k} = SUM_{j in M_k} 1/|C_j| INT_{D^j} u v      (their Eq. 24)
  // The 1/|C_j| weighting is what makes the final average contractive.
  // `qx` is the shared quadrature pool in REFERENCE coords; phys = x0 + h*xr.
  void factor(const std::vector<SrdElem> &e, const SayeNode *qx, i32 degree) {
    N = degree; nb = SrdBasis::count(N);
    basis.assign(nElem, SrdBasis{}); chol.assign(nElem, {});
    std::vector<double> psi(nb);
    for (i32 k=0;k<nElem;k++) {
      if (trivial[k]) continue;
      // centroid + scale over the neighbourhood (conditioning, their Eq. 51-52)
      double cc[3]={0,0,0}; double wsum=0;
      for (i32 j : M[k]) for (i32 q=e[j].qOff; q<e[j].qOff+e[j].qN; q++) {
        double w=(double)qx[q].w*e[j].h*e[j].h*e[j].h;
        for (i32 d=0;d<3;d++) cc[d]+= w*(e[j].x0[d]+e[j].h*(double)qx[q].x[d]);
        wsum += w;
      }
      if (wsum<=0) { trivial[k]=1; continue; }
      for (i32 d=0;d<3;d++) cc[d]/=wsum;
      double sc=0;
      for (i32 j : M[k]) for (i32 q=e[j].qOff; q<e[j].qOff+e[j].qN; q++) {
        double r2=0;
        for (i32 d=0;d<3;d++){ double t=e[j].x0[d]+e[j].h*(double)qx[q].x[d]-cc[d]; r2+=t*t; }
        if (r2>sc) sc=r2;
      }
      basis[k].init(N, cc, sqrt(sc));
      std::vector<double> A((size_t)nb*nb, 0.0);
      for (i32 j : M[k]) {
        double wj = 1.0/(double)Ccnt[j], hv = e[j].h*e[j].h*e[j].h;
        for (i32 q=e[j].qOff; q<e[j].qOff+e[j].qN; q++) {
          double X[3];
          for (i32 d=0;d<3;d++) X[d]=e[j].x0[d]+e[j].h*(double)qx[q].x[d];
          basis[k].eval(X, psi.data());
          double w = wj * (double)qx[q].w * hv;
          for (i32 a=0;a<nb;a++) for (i32 b=a;b<nb;b++) A[(size_t)a*nb+b] += w*psi[a]*psi[b];
        }
      }
      for (i32 a=0;a<nb;a++) for (i32 b=0;b<a;b++) A[(size_t)a*nb+b]=A[(size_t)b*nb+a];
      std::vector<double> dummy(nb,0.0);
      if (!srdSolveSPD(A, dummy, nb)) { trivial[k]=1; continue; }   // degenerate: skip
      chol[k].swap(A);                                             // keep the factor
    }
  }

  // solve with the stored Cholesky factor of neighbourhood k
  void applyChol(i32 k, std::vector<double> &b) const {
    const std::vector<double> &L = chol[k];
    for (i32 i=0;i<nb;i++){ double s=b[i]; for(i32 q=0;q<i;q++) s-=L[(size_t)i*nb+q]*b[q]; b[i]=s/L[(size_t)i*nb+i]; }
    for (i32 i=nb-1;i>=0;i--){ double s=b[i]; for(i32 q=i+1;q<nb;q++) s-=L[(size_t)q*nb+i]*b[q]; b[i]=s/L[(size_t)i*nb+i]; }
  }

  // reverse map C_k = { j : k in M_j }, built after buildNeighborhoods()
  std::vector<std::vector<i32>> C;
  void buildReverse() {
    C.assign(nElem, {});
    for (i32 k=0;k<nElem;k++) for (i32 j : M[k]) C[j].push_back(k);
  }
};

// ---------------------------------------------------------------------------
//  Apply S to a nodal field.  u / uOut are [nElem][ndof][nComp], ndof = B.n^3,
//  with the DG nodal (tensor Lagrange) basis B on each element's own reference
//  cell.  Components are redistributed INDEPENDENTLY, as the papers specify for
//  systems.
//
//  A TRIVIAL neighbourhood (M_k == {k}, a healthy element) contributes the
//  ORIGINAL solution, not a projection -- see Taylor et al. Fig. 2b.  That is
//  what stops SRD from degrading Q^N to P^N on elements that never needed
//  stabilizing, and it is why the scheme costs no formal order.
// ---------------------------------------------------------------------------
template <class B_t>
inline void srdApply(const SrdOperator &S, const std::vector<SrdElem> &e,
                     const SayeNode *qx, const B_t &B,
                     const double *u, double *uOut, i32 nComp) {
  const i32 n = B.n, ndof = n*n*n, nb = S.nb;
  std::vector<double> psi(nb), rhs((size_t)nb*nComp);
  // coefficients of Pi_k u, per non-trivial neighbourhood
  std::vector<std::vector<double>> coef(S.nElem);
  std::vector<real> vb(ndof);

  for (i32 k=0;k<S.nElem;k++) {
    if (S.trivial[k]) continue;
    std::fill(rhs.begin(), rhs.end(), 0.0);
    for (i32 j : S.M[k]) {
      double wj = 1.0/(double)S.Ccnt[j], hv = e[j].h*e[j].h*e[j].h;
      const double *uj = u + (size_t)j*ndof*nComp;
      for (i32 q=e[j].qOff; q<e[j].qOff+e[j].qN; q++) {
        real xr[3] = { qx[q].x[0], qx[q].x[1], qx[q].x[2] };
        double X[3];
        for (i32 d=0;d<3;d++) X[d]=e[j].x0[d]+e[j].h*(double)xr[d];
        B.allVal(xr, vb.data());
        S.basis[k].eval(X, psi.data());
        double w = wj * (double)qx[q].w * hv;
        for (i32 c=0;c<nComp;c++) {
          double uq=0;
          for (i32 a=0;a<ndof;a++) uq += uj[(size_t)a*nComp+c]*(double)vb[a];
          double wu = w*uq;
          for (i32 m=0;m<nb;m++) rhs[(size_t)m*nComp+c] += wu*psi[m];
        }
      }
    }
    // one Cholesky solve per component
    coef[k].assign((size_t)nb*nComp, 0.0);
    std::vector<double> col(nb);
    for (i32 c=0;c<nComp;c++) {
      for (i32 m=0;m<nb;m++) col[m]=rhs[(size_t)m*nComp+c];
      S.applyChol(k, col);
      for (i32 m=0;m<nb;m++) coef[k][(size_t)m*nComp+c]=col[m];
    }
  }

  // average every projection that touched element i
  for (i32 i=0;i<S.nElem;i++) {
    double inv = 1.0/(double)S.Ccnt[i];
    double *o = uOut + (size_t)i*ndof*nComp;
    const double *ui = u + (size_t)i*ndof*nComp;
    for (size_t t=0; t<(size_t)ndof*nComp; t++) o[t]=0.0;
    for (i32 j : S.C[i]) {
      if (S.trivial[j]) {                        // healthy element: identity
        for (size_t t=0; t<(size_t)ndof*nComp; t++) o[t] += inv*ui[t];
        continue;
      }
      for (i32 a=0;a<ndof;a++) {
        i32 ii=a%n, jj=(a/n)%n, kk=a/(n*n);
        double X[3] = { e[i].x0[0]+e[i].h*(double)B.t[ii],
                        e[i].x0[1]+e[i].h*(double)B.t[jj],
                        e[i].x0[2]+e[i].h*(double)B.t[kk] };
        S.basis[j].eval(X, psi.data());
        for (i32 c=0;c<nComp;c++) {
          double s=0;
          for (i32 m=0;m<nb;m++) s += coef[j][(size_t)m*nComp+c]*psi[m];
          o[(size_t)a*nComp+c] += inv*s;
        }
      }
    }
  }
}

#endif
