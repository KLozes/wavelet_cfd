//
// Shifted-boundary (GSBM) path of the CutFEM solver -- CutFemSolver::runSbm().
//
// Production realization of the method verified standalone in SbmMms.cu /
// SbmSolve.h, sitting ALONGSIDE the Saye cut-cell path (runQp): same octree +
// oracle (buildMesh), same Qp node lattice and gather/scatter, but
//
//   * the domain is the SURROGATE Omega~ = cells whose CENTRE is inside
//     (phi < 0): every element is FULL -- no cut quadrature, no Saye, no
//     interface polynomial;
//   * boundary conditions are imposed on the surrogate boundary Gamma~ (mesh
//     faces between a surrogate and a non-surrogate cell) by the SHIFTED
//     Nitsche of GSBM Eq. (35) (Colomes, Modderman, Scovazzi, CMAME 452 (2026)
//     118748): plain-v consistency, -adjoint with the Taylor shift S_d u, and
//     the beta1/h penalty on S_d u, with d, nu from the oracle;
//   * the ghost penalty (l = 1..p) acts on interior faces of the boundary BAND
//     (elements touching Gamma~).  It is ESSENTIAL at p >= 2: the shift
//     activates derivatives up to order p and without their jump control the
//     operator is indefinite (verified in M1).
//
// Parameters (verified in SbmMms):  beta1 = 20 (p+1)^2 (2mu+lam) / h,
// beta2 = 0.5 (2mu+lam).  The operator is NON-symmetric (plain-v consistency
// vs shifted adjoint) -- but only MILDLY so once the ghost penalty is in
// (measured asymmetry ~5e-4), and the short-recurrence solvers converge on it:
// default is Jacobi-BiCGStab (O(1) memory; verified same L2 as GMRES to 5
// digits).  SBM_SOLVER=cg|gmres selects the alternatives -- see the solver
// block.  (The historical CG pAp<0 / BiCGStab divergence were artifacts of the
// UNSTABILIZED operator, pre-ghost-penalty.)
//
// Boundary geometry (d, nu) is computed ONCE per face quadrature point and
// cached: with a BVH-backed oracle each evaluation is expensive, and
// recomputing it inside the Krylov loop made blade solves non-terminating.
//
// Cartesian (coordMode 0) only for now; the cylindrical sector needs the
// curved-metric face terms and is the next step.
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#include "CutFemSolver.cuh"
#include "QpBasis.h"
#include "PolyFit.h"
#include "QpElem.h"
#include "SbmShift.h"

static inline u64 sbKey(i32 I, i32 J, i32 K) {
  return (u64)I | ((u64)J << 21) | ((u64)K << 42);
}
static inline void sbBlockDec(u64 loc, i32 &i, i32 &j, i32 &k) {
  k = (i32)((loc >> 40) & ((1u<<20)-1)) - 1;
  j = (i32)((loc >> 20) & ((1u<<20)-1)) - 1;
  i = (i32)( loc        & ((1u<<20)-1)) - 1;
}
static long sbNowUs(void) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

void CutFemSolver::runSbm(void) {
  const i32 p = femOrder;
  QpBasis Bp; Bp.init(p);
  const i32 n = p+1, ndof = n*n*n, ndof3 = 3*ndof, mG = 2*ndof3;
  const real h = cellSize();
  const double mu = prob.mu, lam = prob.lam;
  const double gammaD_ = 20.0*(p+1)*(p+1)*(2*mu+lam);  // GSBM Eq.(35): chi=20
  const bool per = (periodic != 0);
  const double cph = cos((double)pitchAngle), sph = sin((double)pitchAngle);
  const double kapG = getenv("SBM_KAPPA") ? atof(getenv("SBM_KAPPA")) : 0.5;
  const double gammaG_ = kapG*(2*mu+lam);              // GSBM Eq.(17): kappa=0.5 default

  const bool cyl = (ls.coordMode == 1);
  if (ls.coordMode != 0 && !cyl) {
    printf("ERROR: SBM path supports coordMode 0/1 only\n"); return; }
  if (p > QP_MAX) { printf("ERROR: femOrder %d exceeds QP_MAX=%d\n", p, QP_MAX); return; }

  printf("higher : GSBM shifted boundary (Eq.35), p=%d  %s%s  beta1=%.4g/h beta2=%.4g\n",
         p, cyl?"CYLINDRICAL isoparametric":"Cartesian", per?" + cyclic pitch tie":"",
         gammaD_, gammaG_);

  long t0 = sbNowUs();
  initialize();
  buildMesh();

  // -------------------------------------------------------------------------
  //  surrogate elements: cells whose CENTRE is inside (phi < 0).  All full.
  // -------------------------------------------------------------------------
  struct SElem { i32 ci, cj, ck; };
  std::vector<SElem> elems;
  const i32 nB = hashTable.nKeys;
  #pragma omp parallel
  {
    std::vector<SElem> le;
    #pragma omp for schedule(dynamic,4) nowait
    for (i32 b = 0; b < nB; b++) {
      u64 loc = bLocList[b];
      if (loc == kEmpty) continue;
      i32 ib, jb, kb; sbBlockDec(loc, ib, jb, kb);
      for (i32 cz = 0; cz < blockSize; cz++)
      for (i32 cy = 0; cy < blockSize; cy++)
      for (i32 cx = 0; cx < blockSize; cx++) {
        i32 ci = ib*blockSize + cx, cj = jb*blockSize + cy, ck = kb*blockSize + cz;
        if (ls.phi((ci+(real)0.5)*h, (cj+(real)0.5)*h, (ck+(real)0.5)*h) < 0)
          le.push_back({ci,cj,ck});
      }
    }
    #pragma omp critical
    { for (auto &E:le) elems.push_back(E); }
  }
  i32 nE = (i32)elems.size();
  if (nE == 0) { printf("ERROR: no surrogate elements\n"); return; }

  // ---- surrogate connectivity audit + optional pruning ---------------------
  // Thin features (the blade TE at coarse h) can leave cells attached to the
  // body of the surrogate by <=1 face: floppy chains whose near-null modes are
  // held only by boundary penalties -- a Krylov solver's nightmare.  The
  // surrogate is OURS to define, so SBM_PRUNE=1 drops cells with <2 surrogate
  // face-neighbours (iterated to a fixed point; Gamma~ just moves slightly,
  // which the shift absorbs by construction).  The audit prints either way.
  {
    bool prune = getenv("SBM_PRUNE") && atoi(getenv("SBM_PRUNE"))!=0;
    for (i32 pass=0;;pass++){
      std::unordered_map<u64,i32> cid; cid.reserve((size_t)nE*2);
      for (i32 e=0;e<nE;e++) cid[sbKey(elems[e].ci,elems[e].cj,elems[e].ck)]=e;
      std::vector<i32> nnb(nE,0);
      for (i32 e=0;e<nE;e++){ i32 cc[3]={elems[e].ci,elems[e].cj,elems[e].ck};
        for (i32 d=0;d<3;d++) for (i32 s2=0;s2<2;s2++){
          i32 nb[3]={cc[0],cc[1],cc[2]}; nb[d]+= s2?1:-1;
          if (per && d==1){ if (nb[1]==nThetaCells) nb[1]=0; else if (nb[1]<0) nb[1]=nThetaCells-1; }
          if (cid.find(sbKey(nb[0],nb[1],nb[2]))!=cid.end()) nnb[e]++; } }
      i32 c0=0,c1=0;
      for (i32 e=0;e<nE;e++){ if (nnb[e]==0) c0++; else if (nnb[e]==1) c1++; }
      if (pass==0 || c0+c1==0)
        printf("connect: %d elems with 0 nbrs, %d with 1 nbr (of %d)%s\n",
               c0, c1, nE, prune?"":"  [SBM_PRUNE=1 to drop]");
      if (!prune || c0+c1==0) break;
      std::vector<SElem> keep; keep.reserve(nE);
      for (i32 e=0;e<nE;e++) if (nnb[e]>=2) keep.push_back(elems[e]);
      printf("connect: pass %d pruned %d weakly-connected cells\n", pass, nE-(i32)keep.size());
      elems.swap(keep); nE=(i32)elems.size();
      if (nE==0){ printf("ERROR: pruning removed everything\n"); return; }
    }
  }

  // -------------------------------------------------------------------------
  //  node numbering on the p-finer lattice (no cyclic tie: Cartesian only)
  // -------------------------------------------------------------------------
  std::unordered_map<u64,i32> nodeId; nodeId.reserve((size_t)nE*ndof);
  std::vector<i32> eNodeQ((size_t)nE*ndof);
  std::vector<i32> nI, nJ, nK;
  i32 nNodeQ=0;
  for (i32 e=0;e<nE;e++) for (i32 a=0;a<ndof;a++){
    i32 i=a%n, j=(a/n)%n, k=a/(n*n);
    i32 I=p*elems[e].ci+i, J=p*elems[e].cj+j, K=p*elems[e].ck+k;
    u64 key=sbKey(I,J,K);
    auto it=nodeId.find(key); i32 id;
    if (it==nodeId.end()){ id=nNodeQ++; nodeId[key]=id; nI.push_back(I); nJ.push_back(J); nK.push_back(K); }
    else id=it->second;
    eNodeQ[(size_t)e*ndof+a]=id;
  }
  // cyclic dof map: theta-seam slave columns (J = p*nThetaCells) tie to the
  // master at J = 0 through the pitch rotation (identity when non-periodic).
  const i32 Jmax = p*nThetaCells;
  std::vector<i32> realIdx(nNodeQ,-1);
  std::vector<char> rotFlag(nNodeQ,0);
  i32 nDofNode=0, nTie=0, nOrphan=0;
  for (i32 nd=0;nd<nNodeQ;nd++){
    if (per && nJ[nd]==Jmax) continue;
    realIdx[nd]=nDofNode++;
  }
  for (i32 nd=0;nd<nNodeQ;nd++){
    if (realIdx[nd]>=0) continue;
    auto it=nodeId.find(sbKey(nI[nd],0,nK[nd]));
    if (it==nodeId.end()){ realIdx[nd]=nDofNode++; nOrphan++; }
    else { realIdx[nd]=realIdx[it->second]; rotFlag[nd]=1; nTie++; }
  }
  const i32 nDofQ=3*nDofNode;
  if (per) printf("cyclic : %d SBM nodes tied across the pitch (%d unmatched kept free), %d -> %d dofs\n",
                  nTie, nOrphan, 3*nNodeQ, nDofQ);

  // gather/scatter through the tie (rotation-aware; identity when non-periodic)
  auto gather3=[&](const std::vector<real>&x,i32 nd,double u[3]){
    i32 b=3*realIdx[nd];
    if (rotFlag[nd]){ u[0]=cph*x[b]-sph*x[b+1]; u[1]=sph*x[b]+cph*x[b+1]; u[2]=x[b+2]; }
    else { u[0]=x[b]; u[1]=x[b+1]; u[2]=x[b+2]; }
  };
  auto scatter3=[&](std::vector<real>&y,i32 nd,const double c[3]){
    i32 b=3*realIdx[nd]; double a0,a1,a2v;
    if (rotFlag[nd]){ a0=cph*c[0]+sph*c[1]; a1=-sph*c[0]+cph*c[1]; a2v=c[2]; }
    else { a0=c[0]; a1=c[1]; a2v=c[2]; }
    #pragma omp atomic
    y[b]+=(real)a0;
    #pragma omp atomic
    y[b+1]+=(real)a1;
    #pragma omp atomic
    y[b+2]+=(real)a2v;
  };

  std::vector<real> nodeXQ((size_t)3*nNodeQ);
  for (i32 nd=0;nd<nNodeQ;nd++){
    real X0,X1,X2; ls.toPhys((real)nI[nd]*h/p,(real)nJ[nd]*h/p,(real)nK[nd]*h/p,X0,X1,X2);
    nodeXQ[3*nd]=X0; nodeXQ[3*nd+1]=X1; nodeXQ[3*nd+2]=X2;
  }

  // -------------------------------------------------------------------------
  //  surrogate boundary faces + ghost faces (boundary band)
  // -------------------------------------------------------------------------
  std::unordered_map<u64,i32> cellId; cellId.reserve((size_t)nE*2);
  for (i32 e=0;e<nE;e++) cellId[sbKey(elems[e].ci,elems[e].cj,elems[e].ck)]=e;

  struct BF{ i32 e, d, s; };            // owning elem, axis, side (0:-,1:+)
  std::vector<BF> bf;
  for (i32 e=0;e<nE;e++){ i32 cc[3]={elems[e].ci,elems[e].cj,elems[e].ck};
    for (i32 d=0;d<3;d++) for (i32 s=0;s<2;s++){
      i32 nb[3]={cc[0],cc[1],cc[2]}; nb[d]+= s?1:-1;
      if (per && d==1){ if (nb[1]==nThetaCells) nb[1]=0; else if (nb[1]<0) nb[1]=nThetaCells-1; }
      if (cellId.find(sbKey(nb[0],nb[1],nb[2]))==cellId.end()) bf.push_back({e,d,s});
    } }
  i32 nBF=(i32)bf.size();

  std::vector<char> nearB((size_t)nE,0);
  for (i32 f=0;f<nBF;f++) nearB[bf[f].e]=1;
  struct GF{ i32 eM,eP,d; };
  std::vector<GF> gf;
  for (i32 e=0;e<nE;e++){ i32 cc[3]={elems[e].ci,elems[e].cj,elems[e].ck};
    for (i32 d=0;d<3;d++){ i32 nb[3]={cc[0],cc[1],cc[2]}; nb[d]++;
      if (per && d==1 && nb[1]==nThetaCells) nb[1]=0;
      auto it=cellId.find(sbKey(nb[0],nb[1],nb[2])); if (it==cellId.end()) continue;
      i32 ep=it->second; if (nearB[e]||nearB[ep]) gf.push_back({e,ep,d}); } }
  i32 nGFQ=(i32)gf.size();

  // cylindrical metric at a reference point of element E: Jinv (=Jref^{-1}), detJ
  auto inv3s=[&](const double J[3][3], double Ji[3][3])->double{
    double c00=J[1][1]*J[2][2]-J[1][2]*J[2][1];
    double c01=J[1][2]*J[2][0]-J[1][0]*J[2][2];
    double c02=J[1][0]*J[2][1]-J[1][1]*J[2][0];
    double det=J[0][0]*c00+J[0][1]*c01+J[0][2]*c02;
    double id=(fabs(det)>1e-300)?1.0/det:0.0;
    Ji[0][0]=c00*id; Ji[1][0]=c01*id; Ji[2][0]=c02*id;
    Ji[0][1]=(J[0][2]*J[2][1]-J[0][1]*J[2][2])*id;
    Ji[1][1]=(J[0][0]*J[2][2]-J[0][2]*J[2][0])*id;
    Ji[2][1]=(J[0][1]*J[2][0]-J[0][0]*J[2][1])*id;
    Ji[0][2]=(J[0][1]*J[1][2]-J[0][2]*J[1][1])*id;
    Ji[1][2]=(J[0][2]*J[1][0]-J[0][0]*J[1][2])*id;
    Ji[2][2]=(J[0][0]*J[1][1]-J[0][1]*J[1][0])*id;
    return det;
  };
  auto metric=[&](const SElem&E,const real xr[3],double Jinv[3][3],double&detJ){
    double r=(E.ci+xr[0])*h+ls.org[0], s2=(E.cj+xr[1])*h+ls.org[1], z=(E.ck+xr[2])*h+ls.org[2];
    double th=s2/ls.rRef+(double)ls.thc((real)z), thp=(double)ls.thcSlope((real)z);
    double ct=cos(th), st=sin(th);
    double J[3][3]={{ h*ct, h*(-r*st/ls.rRef), h*(-r*st*thp) },
                    { h*st, h*( r*ct/ls.rRef), h*( r*ct*thp) },
                    { 0,    0,                 h            }};
    detJ=inv3s(J,Jinv);
  };

  // -------------------------------------------------------------------------
  //  boundary geometry cache: d (computational frame) and nu at every face
  //  quadrature point, from the oracle (phiGrad + Newton), computed ONCE.
  // -------------------------------------------------------------------------
  GaussRule gq = gaussLegendre(n);
  const i32 NQF = gq.n*gq.n;
  std::vector<double> geoD((size_t)nBF*NQF*6, 0.0);
  long nClamp=0;
  const double dCapF = getenv("SBM_DCAP") ? atof(getenv("SBM_DCAP")) : 3.0;
  const double dCap = dCapF*(double)h;   // safety: a surrogate face's true wall
                                         // should be within ~1 cell; beyond this
                                         // the Newton went wrong -> d=0 (plain
                                         // Nitsche at the face, exact-data MMS)
  #pragma omp parallel for schedule(dynamic,16) reduction(+:nClamp)
  for (i32 f=0;f<nBF;f++){ const BF&F=bf[f];
    const SElem&E=elems[F.e];
    i32 t1=(F.d+1)%3, t2=(F.d+2)%3;
    for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
      real xr[3]; xr[F.d]=F.s?(real)1:(real)0; xr[t1]=gq.x[q1]; xr[t2]=gq.x[q2];
      double c0=(E.ci+xr[0])*h, c1=(E.cj+xr[1])*h, c2=(E.ck+xr[2])*h;
      double cx=c0, cy=c1, cz=c2;
      real g3[3]; double gm2=1;
      for (i32 it=0; it<3; it++){                    // Newton to {phi=0}
        real fv=ls.phiGrad((real)cx,(real)cy,(real)cz,g3);
        gm2=(double)g3[0]*g3[0]+(double)g3[1]*g3[1]+(double)g3[2]*g3[2];
        if (gm2<1e-30) break;
        cx-=fv*g3[0]/gm2; cy-=fv*g3[1]/gm2; cz-=fv*g3[2]/gm2;
      }
      double dv[3]={cx-c0,cy-c1,cz-c2};
      double dm=sqrt(dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2]);
      if (!(dm<=dCap)){ dv[0]=dv[1]=dv[2]=0; nClamp++; }
      ls.phiGrad((real)cx,(real)cy,(real)cz,g3);
      double gm=sqrt((double)g3[0]*g3[0]+(double)g3[1]*g3[1]+(double)g3[2]*g3[2]);
      if (gm<1e-30) gm=1;
      double *g6=&geoD[((size_t)f*NQF + q1*gq.n + q2)*6];
      g6[0]=dv[0]; g6[1]=dv[1]; g6[2]=dv[2];
      g6[3]=g3[0]/gm; g6[4]=g3[1]/gm; g6[5]=g3[2]/gm;
    } }
  if (nClamp) printf("warning: %ld boundary quadrature points clamped (|d| > %g)\n",
                     nClamp, dCap);
  { double dmax=0,dsum=0; long cnt=0; i32 nbig=0;
    for (i32 f=0;f<nBF;f++) for (i32 q=0;q<NQF;q++){
      const double*g6=&geoD[((size_t)f*NQF+q)*6];
      double dm=sqrt(g6[0]*g6[0]+g6[1]*g6[1]+g6[2]*g6[2])/h;
      dmax=fmax(dmax,dm); dsum+=dm; cnt++; if (dm>1.0) nbig++; }
    printf("shift  : |d|/h max %.3f mean %.3f, >1h at %d/%ld face pts\n",
           dmax, dsum/(cnt?cnt:1), nbig, cnt); }

  // -------------------------------------------------------------------------
  //  shift matrices + ghost D^l rows
  // -------------------------------------------------------------------------
  real Vm[QN_MAX][QN_MAX]; sbmDerivMatrix(Bp, Vm);

  // Per-face-quadrature-point basis tables, hoisted out of the Krylov loop:
  // sh (the Taylor shift S_d phi_a -- the expensive one), gb, vb.  Everything
  // the face apply needs per point is then a table read; profiling showed the
  // per-iteration sbmShiftAll evaluation dominated the GMRES cost.
  std::vector<real> shTab, gbTab, vbTab;
  double Dl0[QP_MAX+1][QN_MAX], Dl1[QP_MAX+1][QN_MAX];
  { double Dp[QN_MAX][QN_MAX];
    for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++) Dp[i][a]=Bp.D[i][a];
    for (i32 l=1;l<=p;l++){ for (i32 a=0;a<n;a++){ Dl0[l][a]=Dp[0][a]; Dl1[l][a]=Dp[n-1][a]; }
      if (l<p){ double Nw[QN_MAX][QN_MAX];
        for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++){ double s=0; for(i32 m=0;m<n;m++) s+=Dp[i][m]*Bp.D[m][a]; Nw[i][a]=s; }
        for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++) Dp[i][a]=Nw[i][a]; } } }

  shTab.assign((size_t)nBF*NQF*ndof,0); gbTab.assign((size_t)nBF*NQF*3*ndof,0);
  vbTab.assign((size_t)nBF*NQF*ndof,0);
  std::vector<double> wNTab;                 // cyl: weighted physical normal nraw*|detJ|*w
  if (cyl) wNTab.assign((size_t)nBF*NQF*3,0.0);
  #pragma omp parallel for schedule(dynamic,16)
  for (i32 f=0;f<nBF;f++){ const BF&F=bf[f];
    i32 t1=(F.d+1)%3, t2=(F.d+2)%3;
    real gb[3*QN_MAX*QN_MAX*QN_MAX], sh[QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
    for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
      real xr[3]; xr[F.d]=F.s?(real)1:(real)0; xr[t1]=gq.x[q1]; xr[t2]=gq.x[q2];
      const double*g6=&geoD[((size_t)f*NQF + q1*gq.n + q2)*6];
      real dref[3]={(real)(g6[0]/h),(real)(g6[1]/h),(real)(g6[2]/h)};
      Bp.allGradRef(xr,gb); Bp.allVal(xr,vb); sbmShiftAll(Bp,Vm,xr,dref,sh);
      size_t qp=(size_t)f*NQF + q1*gq.n + q2;
      for (i32 a=0;a<ndof;a++){ shTab[qp*ndof+a]=sh[a]; vbTab[qp*ndof+a]=vb[a]; }
      if (!cyl){
        for (i32 a=0;a<ndof;a++) for (i32 d2=0;d2<3;d2++) gbTab[(qp*3+d2)*ndof+a]=gb[3*a+d2];
      } else {
        // PHYSICAL gradients + Nanson weighted normal (reference face normal = +-e_d)
        double Jinv[3][3],detJ; metric(elems[F.e],xr,Jinv,detJ);
        double nsign=F.s?1.0:-1.0, wf=gq.w[q1]*gq.w[q2];
        for (i32 i2=0;i2<3;i2++)
          wNTab[qp*3+i2]=nsign*Jinv[F.d][i2]*fabs(detJ)*wf;
        for (i32 a=0;a<ndof;a++) for (i32 d2=0;d2<3;d2++)
          gbTab[(qp*3+d2)*ndof+a]=(real)(Jinv[0][d2]*gb[3*a]+Jinv[1][d2]*gb[3*a+1]+Jinv[2][d2]*gb[3*a+2]);
      } } }

  // ghost matrices (one per axis, reference-invariant; same as runQp but kappa=0.5)
  auto ghostLocal=[&](i32 d,const double*uMP,double*yMP){
    for (i32 i=0;i<mG;i++) yMP[i]=0.0;
    i32 t1=(d+1)%3,t2=(d+2)%3; GaussRule g1=gaussLegendre(p+1);
    for (i32 q1=0;q1<g1.n;q1++) for (i32 q2=0;q2<g1.n;q2++){ double w=g1.w[q1]*g1.w[q2];
      real L1[QN_MAX],L2[QN_MAX]; Bp.basis1(g1.x[q1],L1); Bp.basis1(g1.x[q2],L2);
      double cP[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX], cM[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ i32 idx[3]={a%n,(a/n)%n,a/(n*n)}; i32 idn=idx[d];
        double Lt=L1[idx[t1]]*L2[idx[t2]]; for (i32 l=1;l<=p;l++){ cP[l][a]=Dl0[l][idn]*Lt; cM[l][a]=Dl1[l][idn]*Lt; } }
      for (i32 l=1;l<=p;l++){ double cf=gammaG_*h*w; double jU[3]={0,0,0};
        for (i32 a=0;a<ndof;a++) for(i32 comp=0;comp<3;comp++) jU[comp]+=uMP[ndof3+3*a+comp]*cP[l][a]-uMP[3*a+comp]*cM[l][a];
        for (i32 a=0;a<ndof;a++) for(i32 comp=0;comp<3;comp++){ yMP[ndof3+3*a+comp]+=cf*cP[l][a]*jU[comp];
          yMP[3*a+comp]+=cf*(-cM[l][a])*jU[comp]; } } } };
  std::vector<double> Kghost[3];
  for (i32 d=0;d<3;d++){ Kghost[d].assign((size_t)mG*mG,0.0);
    double ue[2*3*QN_MAX*QN_MAX*QN_MAX], ye[2*3*QN_MAX*QN_MAX*QN_MAX];
    for (i32 cq=0;cq<mG;cq++){ for(i32 a=0;a<mG;a++) ue[a]=(a==cq)?1.0:0.0;
      ghostLocal(d,ue,ye); for (i32 r=0;r<mG;r++) Kghost[d][(size_t)r*mG+cq]=ye[r]; } }

  // -------------------------------------------------------------------------
  //  SBM face local apply (GSBM Eq. 35, reference frame; verified in SbmSolve.h)
  // -------------------------------------------------------------------------
  auto sbmFace=[&](i32 f,const double*uloc,double*yloc){
    const BF&F=bf[f];
    i32 d=F.d;
    double nsign=F.s?1.0:-1.0, nn[3]={0,0,0}; nn[d]=nsign;
    for (i32 a=0;a<ndof3;a++) yloc[a]=0.0;
    if (cyl){
      for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
        size_t qp=(size_t)f*NQF + q1*gq.n + q2;
        const real*sh=&shTab[qp*ndof]; const real*vb=&vbTab[qp*ndof];
        const real*gx=&gbTab[(qp*3+0)*ndof], *gy=&gbTab[(qp*3+1)*ndof], *gz=&gbTab[(qp*3+2)*ndof];
        const double*wN=&wNTab[qp*3];
        double dS=sqrt(wN[0]*wN[0]+wN[1]*wN[1]+wN[2]*wN[2]);
        double gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}}, Shu[3]={0,0,0};
        for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ double ua=uloc[3*a+i2];
          gradU[i2][0]+=ua*gx[a]; gradU[i2][1]+=ua*gy[a]; gradU[i2][2]+=ua*gz[a];
          Shu[i2]+=ua*sh[a]; }
        double eps[3][3],tr=0;
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
        tr=eps[0][0]+eps[1][1]+eps[2][2];
        double sig[3][3];
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*tr:0);
        double tw[3]; for(i32 i2=0;i2<3;i2++) tw[i2]=sig[i2][0]*wN[0]+sig[i2][1]*wN[1]+sig[i2][2]*wN[2];
        double ShuN=Shu[0]*wN[0]+Shu[1]*wN[1]+Shu[2]*wN[2];
        double penC=gammaD_/h;
        real gA[3];
        for (i32 a=0;a<ndof;a++){
          gA[0]=gx[a]; gA[1]=gy[a]; gA[2]=gz[a];
          double gan=gA[0]*wN[0]+gA[1]*wN[1]+gA[2]*wN[2];
          double ugb=Shu[0]*gA[0]+Shu[1]*gA[1]+Shu[2]*gA[2];
          for (i32 l=0;l<3;l++){
            double c1=-tw[l]*vb[a];                                   // -(v, sigma(u).n) dS
            double c2=-(mu*(Shu[l]*gan+ugb*wN[l])+lam*gA[l]*ShuN);    // -(sigma(v).n, S_d u) dS
            double c3=penC*Shu[l]*sh[a]*dS;                           // beta1/h penalty
            yloc[3*a+l]+=c1+c2+c3; } }
      }
      return;
    }
    for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
      double hw=gq.w[q1]*gq.w[q2]*h;
      size_t qp=(size_t)f*NQF + q1*gq.n + q2;
      const real*sh=&shTab[qp*ndof]; const real*vb=&vbTab[qp*ndof];
      const real*gx=&gbTab[(qp*3+0)*ndof], *gy=&gbTab[(qp*3+1)*ndof], *gz=&gbTab[(qp*3+2)*ndof];
      const real*gd=&gbTab[(qp*3+d)*ndof];
      double gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}}, Shu[3]={0,0,0};
      for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ double ua=uloc[3*a+i2];
        gradU[i2][0]+=ua*gx[a]; gradU[i2][1]+=ua*gy[a]; gradU[i2][2]+=ua*gz[a];
        Shu[i2]+=ua*sh[a]; }
      double eps[3][3],tr=0;
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2];
      double sig[3][3];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*tr:0);
      double tu[3]; for(i32 i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nn[0]+sig[i2][1]*nn[1]+sig[i2][2]*nn[2];
      double Shun=Shu[0]*nn[0]+Shu[1]*nn[1]+Shu[2]*nn[2];
      real gA[3];
      for (i32 a=0;a<ndof;a++){ double gan=gd[a]*nsign;
        gA[0]=gx[a]; gA[1]=gy[a]; gA[2]=gz[a];
        double ugb=Shu[0]*gA[0]+Shu[1]*gA[1]+Shu[2]*gA[2];
        for (i32 l=0;l<3;l++){
          double c1=-tu[l]*vb[a];                                     // -(v, sigma(u).n~)
          double c2=-(mu*(Shu[l]*gan+ugb*nn[l])+lam*gA[l]*Shun);      // -(sigma(v).n~, S_d u)
          double c3=gammaD_*Shu[l]*sh[a];                             // (beta1/h S_d v, S_d u)
          yloc[3*a+l]+=hw*(c1+c2+c3); } }
    } };

  // -------------------------------------------------------------------------
  //  matrix-free operator apply
  // -------------------------------------------------------------------------
  auto applyA=[&](const std::vector<real>&x,std::vector<real>&y){
    std::fill(y.begin(),y.end(),(real)0);
    #pragma omp parallel for schedule(dynamic,64)
    for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
      real ul[3*QN_MAX*QN_MAX*QN_MAX], yl[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ double u3[3]; gather3(x,nod[a],u3);
        ul[3*a]=(real)u3[0]; ul[3*a+1]=(real)u3[1]; ul[3*a+2]=(real)u3[2]; }
      if (cyl){
        double ylc[3*QN_MAX*QN_MAX*QN_MAX]; for (i32 a2=0;a2<ndof3;a2++) ylc[a2]=0.0;
        real gb2[3*QN_MAX*QN_MAX*QN_MAX]; double gX2[QN_MAX*QN_MAX*QN_MAX][3];
        for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
          real xr[3]={Bp.t[i],Bp.t[j],Bp.t[k]};
          double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
          double wdet=fabs(detJ)*Bp.wq[i]*Bp.wq[j]*Bp.wq[k];
          Bp.allGradRef(xr,gb2);
          for (i32 a2=0;a2<ndof;a2++) for (i32 d2=0;d2<3;d2++)
            gX2[a2][d2]=Jinv[0][d2]*gb2[3*a2]+Jinv[1][d2]*gb2[3*a2+1]+Jinv[2][d2]*gb2[3*a2+2];
          double gU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
          for (i32 a2=0;a2<ndof;a2++) for(i32 i2=0;i2<3;i2++){ double ua=ul[3*a2+i2];
            gU[i2][0]+=ua*gX2[a2][0]; gU[i2][1]+=ua*gX2[a2][1]; gU[i2][2]+=ua*gX2[a2][2]; }
          double ep2[3][3],tr2=0;
          for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) ep2[i2][j2]=0.5*(gU[i2][j2]+gU[j2][i2]);
          tr2=ep2[0][0]+ep2[1][1]+ep2[2][2];
          double sg[3][3];
          for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sg[i2][j2]=2*mu*ep2[i2][j2]+(i2==j2?lam*tr2:0);
          for (i32 a2=0;a2<ndof;a2++) for (i32 l=0;l<3;l++)
            ylc[3*a2+l]+=wdet*(sg[l][0]*gX2[a2][0]+sg[l][1]*gX2[a2][1]+sg[l][2]*gX2[a2][2]);
        }
        for (i32 a2=0;a2<ndof;a2++) scatter3(y,nod[a2],&ylc[3*a2]);
        continue;
      }
      qpElemUncut(Bp,(real)mu,(real)lam,h,ul,yl);
      for (i32 a=0;a<ndof;a++){ double c3[3]={(double)yl[3*a],(double)yl[3*a+1],(double)yl[3*a+2]};
        scatter3(y,nod[a],c3); } }
    #pragma omp parallel for schedule(dynamic,64)
    for (i32 f=0;f<nBF;f++){ const i32*nod=&eNodeQ[(size_t)bf[f].e*ndof];
      double uloc[3*QN_MAX*QN_MAX*QN_MAX], yloc[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++) gather3(x,nod[a],&uloc[3*a]);
      sbmFace(f,uloc,yloc);
      for (i32 a=0;a<ndof;a++) scatter3(y,nod[a],&yloc[3*a]); }
    #pragma omp parallel for schedule(dynamic,64)
    for (i32 f=0;f<nGFQ;f++){ const GF&F=gf[f];
      const i32*nodM=&eNodeQ[(size_t)F.eM*ndof], *nodP=&eNodeQ[(size_t)F.eP*ndof];
      double uMP[2*3*QN_MAX*QN_MAX*QN_MAX], yMP[2*3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ gather3(x,nodM[a],&uMP[3*a]); gather3(x,nodP[a],&uMP[ndof3+3*a]); }
      const double*K=Kghost[F.d].data();
      for (i32 r=0;r<mG;r++){ double s=0; const double*Kr=&K[(size_t)r*mG]; for(i32 c=0;c<mG;c++) s+=Kr[c]*uMP[c]; yMP[r]=s; }
      for (i32 a=0;a<ndof;a++){ scatter3(y,nodM[a],&yMP[3*a]); scatter3(y,nodP[a],&yMP[ndof3+3*a]); } }
  };

  // -------------------------------------------------------------------------
  //  RHS (body force + shifted-Nitsche Dirichlet data) and Jacobi diagonal
  // -------------------------------------------------------------------------
  std::vector<real> bvec(nDofQ,(real)0);
  std::vector<double> diagN((size_t)3*nNodeQ,0.0);   // per-NODE diag (folded to dofs below)
  std::vector<double> diagX((size_t)3*nNodeQ,0.0);   // per-node xy,xz,yz cross terms
  #pragma omp parallel for schedule(dynamic,64)
  for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
    real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
    double gXl[QN_MAX*QN_MAX*QN_MAX][3];
    for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
      real xr[3]={Bp.t[i],Bp.t[j],Bp.t[k]};
      double wv=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]*h*h*h, wb=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]*h;
      Bp.allGradRef(xr,gb); Bp.allVal(xr,vb);
      real X[3]; ls.toPhys((elems[e].ci+xr[0])*h,(elems[e].cj+xr[1])*h,(elems[e].ck+xr[2])*h,X[0],X[1],X[2]);
      real fb[3]; prob.bodyForce(X[0],X[1],X[2],fb);
      if (cyl){
        double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
        double wdet=fabs(detJ)*Bp.wq[i]*Bp.wq[j]*Bp.wq[k];
        for (i32 a=0;a<ndof;a++) for (i32 d2=0;d2<3;d2++)
          gXl[a][d2]=Jinv[0][d2]*gb[3*a]+Jinv[1][d2]*gb[3*a+1]+Jinv[2][d2]*gb[3*a+2];
        for (i32 a=0;a<ndof;a++){
          double gsq=gXl[a][0]*gXl[a][0]+gXl[a][1]*gXl[a][1]+gXl[a][2]*gXl[a][2];
          { double c3[3]={wdet*fb[0]*vb[a],wdet*fb[1]*vb[a],wdet*fb[2]*vb[a]}; scatter3(bvec,nod[a],c3); }
          for (i32 l=0;l<3;l++){
            double dvl=wdet*(mu*(gsq+gXl[a][l]*gXl[a][l])+lam*gXl[a][l]*gXl[a][l]);
            #pragma omp atomic
            diagN[3*nod[a]+l]+=dvl; }
          double c01=wdet*(mu+lam)*gXl[a][0]*gXl[a][1];
          double c02=wdet*(mu+lam)*gXl[a][0]*gXl[a][2];
          double c12=wdet*(mu+lam)*gXl[a][1]*gXl[a][2];
          #pragma omp atomic
          diagX[3*nod[a]  ]+=c01;
          #pragma omp atomic
          diagX[3*nod[a]+1]+=c02;
          #pragma omp atomic
          diagX[3*nod[a]+2]+=c12; }
        continue;
      }
      for (i32 a=0;a<ndof;a++){
        double gsq=(double)gb[3*a]*gb[3*a]+(double)gb[3*a+1]*gb[3*a+1]+(double)gb[3*a+2]*gb[3*a+2];
        { double c3[3]={wv*fb[0]*vb[a],wv*fb[1]*vb[a],wv*fb[2]*vb[a]}; scatter3(bvec,nod[a],c3); }
        for (i32 l=0;l<3;l++){
          double dvl=wb*(mu*(gsq+(double)gb[3*a+l]*gb[3*a+l])+lam*(double)gb[3*a+l]*gb[3*a+l]);
          #pragma omp atomic
          diagN[3*nod[a]+l]+=dvl; }
        // 3x3 nodal-block cross terms (bulk only; penalty/ghost diag are
        // component-isotropic): K_aa[l][m] = wb*(mu+lam)*g_l*g_m for l != m
        double c01=wb*(mu+lam)*(double)gb[3*a]*gb[3*a+1];
        double c02=wb*(mu+lam)*(double)gb[3*a]*gb[3*a+2];
        double c12=wb*(mu+lam)*(double)gb[3*a+1]*gb[3*a+2];
        #pragma omp atomic
        diagX[3*nod[a]  ]+=c01;
        #pragma omp atomic
        diagX[3*nod[a]+1]+=c02;
        #pragma omp atomic
        diagX[3*nod[a]+2]+=c12; } } }
  i32 nNeuSkip=0;
  #pragma omp parallel for schedule(dynamic,16) reduction(+:nNeuSkip)
  for (i32 f=0;f<nBF;f++){ const BF&F=bf[f];
    const SElem&E=elems[F.e]; const i32*nod=&eNodeQ[(size_t)F.e*ndof];
    i32 d=F.d, t1=(d+1)%3, t2=(d+2)%3;
    double nsign=F.s?1.0:-1.0, nn[3]={0,0,0}; nn[d]=nsign;
    real gb[3*QN_MAX*QN_MAX*QN_MAX], sh[QN_MAX*QN_MAX*QN_MAX];
    for (i32 q1=0;q1<gq.n;q1++) for (i32 q2=0;q2<gq.n;q2++){
      real xr[3]; xr[d]=F.s?(real)1:(real)0; xr[t1]=gq.x[q1]; xr[t2]=gq.x[q2];
      double hw=gq.w[q1]*gq.w[q2]*h;
      const double*g6=&geoD[((size_t)f*NQF + q1*gq.n + q2)*6];
      real dref[3]={(real)(g6[0]/h),(real)(g6[1]/h),(real)(g6[2]/h)};
      Bp.allGradRef(xr,gb); sbmShiftAll(Bp,Vm,xr,dref,sh);
      real XT[3]; ls.toPhys((E.ci+xr[0])*h+(real)g6[0],(E.cj+xr[1])*h+(real)g6[1],
                            (E.ck+xr[2])*h+(real)g6[2],XT[0],XT[1],XT[2]);
      if (!prob.isDirichlet(XT[0],XT[1],XT[2])) { nNeuSkip++; continue; }
      real g[3]; prob.dirichletData(XT[0],XT[1],XT[2],g);
      if (cyl){
        size_t qp=(size_t)f*NQF + q1*gq.n + q2;
        const real*shc=&shTab[qp*ndof];
        const real*gx=&gbTab[(qp*3+0)*ndof], *gy=&gbTab[(qp*3+1)*ndof], *gz=&gbTab[(qp*3+2)*ndof];
        const double*wN=&wNTab[qp*3];
        double dS=sqrt(wN[0]*wN[0]+wN[1]*wN[1]+wN[2]*wN[2]);
        double gnw=g[0]*wN[0]+g[1]*wN[1]+g[2]*wN[2];
        double penC=gammaD_/h;
        for (i32 a=0;a<ndof;a++){
          double gA[3]={gx[a],gy[a],gz[a]};
          double gan=gA[0]*wN[0]+gA[1]*wN[1]+gA[2]*wN[2];
          double ggb=g[0]*gA[0]+g[1]*gA[1]+g[2]*gA[2];
          { double c3[3];
            for (i32 l=0;l<3;l++)
              c3[l]=-(mu*(g[l]*gan+ggb*wN[l])+lam*gA[l]*gnw)+penC*g[l]*shc[a]*dS;
            scatter3(bvec,nod[a],c3); }
          for (i32 l=0;l<3;l++){
            double dvl=penC*shc[a]*shc[a]*dS;
            #pragma omp atomic
            diagN[3*nod[a]+l]+=dvl; } }
        continue;
      }
      double gn=g[0]*nn[0]+g[1]*nn[1]+g[2]*nn[2];
      for (i32 a=0;a<ndof;a++){ double gan=gb[3*a+d]*nsign;
        double ggb=g[0]*gb[3*a]+g[1]*gb[3*a+1]+g[2]*gb[3*a+2];
        { double c3[3];
          for (i32 l=0;l<3;l++){
            double rhs=-(mu*(g[l]*gan+ggb*nn[l])+lam*gb[3*a+l]*gn)+gammaD_*g[l]*sh[a];
            c3[l]=hw*rhs; }
          scatter3(bvec,nod[a],c3); }
        for (i32 l=0;l<3;l++){
          double dvl=hw*gammaD_*sh[a]*sh[a];
          #pragma omp atomic
          diagN[3*nod[a]+l]+=dvl; } } } }
  if (nNeuSkip) printf("warning: %d Neumann face points skipped (SBM traction needs the gap path)\n", nNeuSkip);
  // ghost diagonal
  for (i32 f=0;f<nGFQ;f++){ const GF&F=gf[f];
    const i32*nodM=&eNodeQ[(size_t)F.eM*ndof], *nodP=&eNodeQ[(size_t)F.eP*ndof];
    const double*K=Kghost[F.d].data();
    for (i32 a=0;a<ndof;a++) for (i32 l=0;l<3;l++){
      diagN[3*nodM[a]+l]+=K[(size_t)(3*a+l)*mG+(3*a+l)];
      diagN[3*nodP[a]+l]+=K[(size_t)(ndof3+3*a+l)*mG+(ndof3+3*a+l)]; } }
  std::vector<double> diagv(nDofQ,0.0);
  std::vector<double> diagXD((size_t)3*nDofNode,0.0);
  for (i32 nd=0;nd<nNodeQ;nd++){ i32 b=3*realIdx[nd];
    for (i32 l=0;l<3;l++) diagv[b+l]+=diagN[3*nd+l];
    for (i32 l=0;l<3;l++) diagXD[b+l]+=diagX[3*nd+l]; }
  for (i32 i=0;i<nDofQ;i++) if (diagv[i]<=0) diagv[i]=1.0;

  // -------------------------------------------------------------------------
  //  preconditioner: SBM_PC = jacobi (default) | bjac (3x3 nodal blocks) |
  //  poly[k] (k damped-Jacobi sweeps, default k=2).  bjac captures the
  //  (mu+lam) g_l g_m component coupling of elasticity that scalar Jacobi
  //  ignores; poly is a fixed linear polynomial in A (matrix-free friendly).
  // -------------------------------------------------------------------------
  const char* pcName = getenv("SBM_PC") ? getenv("SBM_PC") : "jacobi";
  i32 polyK = 2;
  if (pcName[0]=='p' && pcName[4]) polyK = atoi(pcName+4);
  std::vector<double> Binv;
  if (pcName[0]=='b'){
    Binv.assign((size_t)9*nDofNode,0.0);
    #pragma omp parallel for
    for (i32 nd=0;nd<nDofNode;nd++){
      double B[3][3]={{diagv[3*nd],diagXD[3*nd],diagXD[3*nd+1]},
                      {diagXD[3*nd],diagv[3*nd+1],diagXD[3*nd+2]},
                      {diagXD[3*nd+1],diagXD[3*nd+2],diagv[3*nd+2]}};
      double c00=B[1][1]*B[2][2]-B[1][2]*B[2][1];
      double c01=B[1][2]*B[2][0]-B[1][0]*B[2][2];
      double c02=B[1][0]*B[2][1]-B[1][1]*B[2][0];
      double det=B[0][0]*c00+B[0][1]*c01+B[0][2]*c02;
      double *Bi=&Binv[(size_t)9*nd];
      if (fabs(det)<1e-300){ Bi[0]=1.0/diagv[3*nd]; Bi[4]=1.0/diagv[3*nd+1]; Bi[8]=1.0/diagv[3*nd+2]; continue; }
      double id=1.0/det;
      Bi[0]=c00*id; Bi[3]=c01*id; Bi[6]=c02*id;
      Bi[1]=(B[0][2]*B[2][1]-B[0][1]*B[2][2])*id;
      Bi[4]=(B[0][0]*B[2][2]-B[0][2]*B[2][0])*id;
      Bi[7]=(B[0][1]*B[2][0]-B[0][0]*B[2][1])*id;
      Bi[2]=(B[0][1]*B[1][2]-B[0][2]*B[1][1])*id;
      Bi[5]=(B[0][2]*B[1][0]-B[0][0]*B[1][2])*id;
      Bi[8]=(B[0][0]*B[1][1]-B[0][1]*B[1][0])*id;
    }
    printf("precond: 3x3 nodal block-Jacobi\n");
  } else if (pcName[0]=='p') printf("precond: polynomial (%d damped-Jacobi sweeps)\n", polyK);
  else printf("precond: Jacobi\n");
  std::vector<real> ptmp1, ptmp2;
  if (pcName[0]=='p'){ ptmp1.assign(nDofQ,(real)0); ptmp2.assign(nDofQ,(real)0); }
  auto precond=[&](const std::vector<real>&r,std::vector<real>&z){
    if (pcName[0]=='b'){
      #pragma omp parallel for
      for (i32 nd=0;nd<nDofNode;nd++){ const double*Bi=&Binv[(size_t)9*nd];
        double r0=r[3*nd],r1=r[3*nd+1],r2=r[3*nd+2];
        z[3*nd  ]=(real)(Bi[0]*r0+Bi[1]*r1+Bi[2]*r2);
        z[3*nd+1]=(real)(Bi[3]*r0+Bi[4]*r1+Bi[5]*r2);
        z[3*nd+2]=(real)(Bi[6]*r0+Bi[7]*r1+Bi[8]*r2); }
    } else if (pcName[0]=='p'){
      const double om=2.0/3.0;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) z[i]=(real)(om*r[i]/diagv[i]);
      for (i32 k2=0;k2<polyK;k2++){
        applyA(z,ptmp1);
        #pragma omp parallel for
        for (i32 i=0;i<nDofQ;i++) z[i]+=(real)(om*(r[i]-ptmp1[i])/diagv[i]);
      }
    } else {
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) z[i]=(real)(r[i]/diagv[i]);
    }
  };

  // -------------------------------------------------------------------------
  //  solver.  Default: restarted Jacobi-GMRES(m) -- the SBM operator is
  //  NON-symmetric, and on the UNSTABILIZED operator CG broke down (pAp<0) and
  //  BiCGStab diverged (measured, M1).  BUT the production operator carries the
  //  ghost penalty and its measured asymmetry is ~5e-4, so the short-recurrence
  //  methods may work as inexact solvers here: SBM_SOLVER=bicgstab|cg|gmres
  //  selects, to settle that experimentally.  BiCGStab/CG are O(1) memory
  //  (~8/5 vectors) vs GMRES's (m+1) -- the memory driver of this path.
  // -------------------------------------------------------------------------
  std::vector<real> uv(nDofQ,(real)0);
  i32 it=0;
  // Default: BiCGStab -- verified on the stabilized operator (533 its, same L2 as
  // GMRES to 5 digits), principled for a non-symmetric system, O(1) memory
  // (8 vectors vs GMRES's m+1; the memory driver of this path at scale).
  // "cg" is faster still (inexact: rides the ~5e-4 asymmetry) but theoretically
  // unsound if the shift grows; "gmres" is the fallback if BiCGStab misbehaves.
  const char* sbmSolver = getenv("SBM_SOLVER") ? getenv("SBM_SOLVER") : "bicgstab";
  if (sbmSolver[0]=='b') {                    // ---- Jacobi-BiCGStab ----
    std::vector<real> r(nDofQ),rh(nDofQ),v(nDofQ,(real)0),pd(nDofQ,(real)0),
                      ph(nDofQ),ss(nDofQ),sh2(nDofQ),tt(nDofQ),Ax(nDofQ);
    applyA(uv,Ax);
    #pragma omp parallel for
    for (i32 i=0;i<nDofQ;i++){ r[i]=bvec[i]-Ax[i]; rh[i]=r[i]; }
    double bn=0; for (i32 i=0;i<nDofQ;i++) bn+=(double)bvec[i]*bvec[i]; bn=sqrt(bn); if(bn==0)bn=1;
    double rho=1,al=1,om=1; bool fail=false;
    for (; it<cgMaxIt; it++){
      double rho2=0; for (i32 i=0;i<nDofQ;i++) rho2+=(double)rh[i]*r[i];
      if (fabs(rho2)<1e-290){ printf("warning: BiCGStab breakdown rho=%.2e at it %d\n",rho2,it); fail=true; break; }
      double be=(rho2/rho)*(al/om); rho=rho2;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) pd[i]=(real)(r[i]+be*(pd[i]-om*v[i]));
      precond(pd,ph);
      applyA(ph,v);
      double rhv=0; for (i32 i=0;i<nDofQ;i++) rhv+=(double)rh[i]*v[i];
      al=rho/rhv;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) ss[i]=(real)(r[i]-al*v[i]);
      double sn=0; for (i32 i=0;i<nDofQ;i++) sn+=(double)ss[i]*ss[i]; sn=sqrt(sn);
      if (sn<=cgTol*bn){
        #pragma omp parallel for
        for (i32 i=0;i<nDofQ;i++) uv[i]+=(real)(al*ph[i]);
        it++; cgRes=sn/bn; break; }
      precond(ss,sh2);
      applyA(sh2,tt);
      double ts=0,ttn=0; for (i32 i=0;i<nDofQ;i++){ ts+=(double)tt[i]*ss[i]; ttn+=(double)tt[i]*tt[i]; }
      om=ts/ttn;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++){ uv[i]+=(real)(al*ph[i]+om*sh2[i]); r[i]=(real)(ss[i]-om*tt[i]); }
      double rn=0; for (i32 i=0;i<nDofQ;i++) rn+=(double)r[i]*r[i]; rn=sqrt(rn);
      cgRes=rn/bn;
      if (getenv("SBM_DBG") && it%50==0)
        printf("    [bicgstab it=%d rres=%.3e rho=%.3e al=%.3e om=%.3e]\n",it,rn/bn,rho,al,om);
      if (rn<=cgTol*bn){ it++; break; }
    }
    if (fail) printf("warning: BiCGStab did not converge cleanly\n");
    printf("solver : Jacobi-BiCGStab (O(1) memory)\n");
  } else if (sbmSolver[0]=='l') {        // ---- BiCGStab(ell), Sleijpen-Fokkema ----
    // Left-Jacobi-preconditioned: operates on Ahat = M^-1 A, bhat = M^-1 b.
    // The ell-dimensional MR polynomial replaces plain BiCGStab's degree-1
    // stabilizer -- the standard cure for the "BiCGStab stalls where GMRES
    // works" signature (complex spectrum / non-normality).  Storage:
    // 2(ell+1)+2 vectors, still O(1) in restart-free memory.
    i32 ell = sbmSolver[1] ? atoi(sbmSolver+1) : 2; if (ell<1) ell=2; if (ell>8) ell=8;
    printf("solver : Jacobi-BiCGStab(%d)\n", ell);
    auto applyHat=[&](const std::vector<real>&xx,std::vector<real>&yy){
      applyA(xx,yy);
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) yy[i]=(real)(yy[i]/diagv[i]); };
    auto dotv=[&](const std::vector<real>&a2,const std::vector<real>&b2){
      double s2=0;
      #pragma omp parallel for reduction(+:s2)
      for (i32 i=0;i<nDofQ;i++) s2+=(double)a2[i]*b2[i]; return s2; };
    std::vector<std::vector<real>> rv(ell+1,std::vector<real>(nDofQ,(real)0)),
                                   uv2(ell+1,std::vector<real>(nDofQ,(real)0));
    std::vector<real> rt(nDofQ), bhat(nDofQ);
    #pragma omp parallel for
    for (i32 i=0;i<nDofQ;i++) bhat[i]=(real)(bvec[i]/diagv[i]);
    { std::vector<real> Ax0(nDofQ); applyHat(uv,Ax0);
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++){ rv[0][i]=(real)(bhat[i]-Ax0[i]); rt[i]=rv[0][i]; } }
    double bn=sqrt(dotv(bhat,bhat)); if (bn==0) bn=1;
    double rho=1, alpha=0, omega=1;
    std::vector<double> tau((size_t)(ell+1)*(ell+1),0.0), sg(ell+1,0.0),
                        gp(ell+1,0.0), ga(ell+1,0.0), gpp(ell+1,0.0);
    i32 nRestart=0;
    for (; it<cgMaxIt; ){
      double rn=sqrt(dotv(rv[0],rv[0]));
      cgRes=rn/bn;
      if (getenv("SBM_DBG") && it%50<2*ell)
        printf("    [bl%d it=%d rres=%.3e rho=%.3e om=%.3e]\n",ell,it,rn/bn,rho,omega);
      if (rn<=cgTol*bn) break;
      rho=-omega*rho;
      bool broke=false;
      for (i32 j=0;j<ell;j++){
        double rho1=dotv(rv[j],rt);
        if (fabs(rho)<1e-290||fabs(rho1)<1e-290){ broke=true; break; }
        double beta=alpha*rho1/rho; rho=rho1;
        for (i32 i2=0;i2<=j;i2++){
          #pragma omp parallel for
          for (i32 q=0;q<nDofQ;q++) uv2[i2][q]=(real)(rv[i2][q]-beta*uv2[i2][q]); }
        applyHat(uv2[j],uv2[j+1]); it++;
        double gm=dotv(uv2[j+1],rt);
        if (fabs(gm)<1e-290){ broke=true; break; }
        alpha=rho/gm;
        for (i32 i2=0;i2<=j;i2++){
          #pragma omp parallel for
          for (i32 q=0;q<nDofQ;q++) rv[i2][q]=(real)(rv[i2][q]-alpha*uv2[i2+1][q]); }
        applyHat(rv[j],rv[j+1]); it++;
        #pragma omp parallel for
        for (i32 q=0;q<nDofQ;q++) uv[q]+=(real)(alpha*uv2[0][q]);
      }
      if (broke){ // shadow restart
        nRestart++;
        if (nRestart>50){ printf("warning: BiCGStab(%d) exceeded restarts\n",ell); break; }
        #pragma omp parallel for
        for (i32 q=0;q<nDofQ;q++){ rt[q]=rv[0][q]; uv2[0][q]=(real)0; }
        rho=1; alpha=0; omega=1; continue; }
      // MR part: minimize ||r0 - sum gamma_j r_j|| (modified Gram-Schmidt)
      for (i32 j=1;j<=ell;j++){
        for (i32 i2=1;i2<j;i2++){
          double t=dotv(rv[j],rv[i2])/sg[i2];
          tau[(size_t)i2*(ell+1)+j]=t;
          #pragma omp parallel for
          for (i32 q=0;q<nDofQ;q++) rv[j][q]-=(real)(t*rv[i2][q]); }
        sg[j]=dotv(rv[j],rv[j]);
        if (sg[j]<1e-290){ broke=true; break; }
        gp[j]=dotv(rv[0],rv[j])/sg[j]; }
      if (broke){ nRestart++;
        #pragma omp parallel for
        for (i32 q=0;q<nDofQ;q++){ rt[q]=rv[0][q]; uv2[0][q]=(real)0; }
        rho=1; alpha=0; omega=1; continue; }
      ga[ell]=gp[ell]; omega=ga[ell];
      for (i32 j=ell-1;j>=1;j--){ double s2=gp[j];
        for (i32 i2=j+1;i2<=ell;i2++) s2-=tau[(size_t)j*(ell+1)+i2]*ga[i2];
        ga[j]=s2; }
      for (i32 j=1;j<ell;j++){ double s2=ga[j+1];
        for (i32 i2=j+1;i2<ell;i2++) s2+=tau[(size_t)j*(ell+1)+i2]*ga[i2+1];
        gpp[j]=s2; }
      #pragma omp parallel for
      for (i32 q=0;q<nDofQ;q++){
        uv[q]+=(real)(ga[1]*rv[0][q]);
        rv[0][q]-=(real)(gp[ell]*rv[ell][q]);
        uv2[0][q]-=(real)(ga[ell]*uv2[ell][q]); }
      for (i32 j=1;j<ell;j++){
        #pragma omp parallel for
        for (i32 q=0;q<nDofQ;q++){
          uv2[0][q]-=(real)(ga[j]*uv2[j][q]);
          uv[q]+=(real)(gpp[j]*rv[j][q]);
          rv[0][q]-=(real)(gp[j]*rv[j][q]); } }
    }
    if (nRestart) printf("solver : %d shadow restarts\n",nRestart);
  } else if (sbmSolver[0]=='c') {             // ---- Jacobi-CG (inexact: operator is ~1e-4 asymmetric) ----
    std::vector<real> r(nDofQ),z(nDofQ),pd(nDofQ),Ap(nDofQ);
    applyA(uv,Ap);
    #pragma omp parallel for
    for (i32 i=0;i<nDofQ;i++) r[i]=bvec[i]-Ap[i];
    precond(r,z);
    pd=z;
    double rz=0; for (i32 i=0;i<nDofQ;i++) rz+=(double)r[i]*z[i];
    double bn=0; for (i32 i=0;i<nDofQ;i++) bn+=(double)bvec[i]*bvec[i]; bn=sqrt(bn); if(bn==0)bn=1;
    for (; it<cgMaxIt; it++){
      applyA(pd,Ap);
      double pAp=0; for (i32 i=0;i<nDofQ;i++) pAp+=(double)pd[i]*Ap[i];
      if (!(pAp>0)){ printf("warning: CG indefinite pAp=%.3e at it %d\n",pAp,it); break; }
      double al=rz/pAp;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++){ uv[i]+=(real)(al*pd[i]); r[i]-=(real)(al*Ap[i]); }
      double rn=0; for (i32 i=0;i<nDofQ;i++) rn+=(double)r[i]*r[i]; rn=sqrt(rn);
      cgRes=rn/bn;
      if (rn<=cgTol*bn){ it++; break; }
      precond(r,z);
      double rz2=0; for (i32 i=0;i<nDofQ;i++) rz2+=(double)r[i]*z[i];
      double be=rz2/rz; rz=rz2;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) pd[i]=(real)(z[i]+be*pd[i]);
    }
    printf("solver : Jacobi-CG (inexact on the ~1e-4-asymmetric operator)\n");
  } else {
    const i32 m=200;
    const double tol=(double)cgTol;
    std::vector<std::vector<real>> V(m+1,std::vector<real>(nDofQ));
    std::vector<double> Hm((size_t)(m+1)*m,0.0),cs(m,0.0),sn(m,0.0),ss(m+1,0.0),yv(m,0.0);
    std::vector<real> w(nDofQ),Ax(nDofQ),rr(nDofQ);
    bool conv=false; double beta0=0;
    for (i32 outer=0;outer<2000&&!conv&&it<cgMaxIt;outer++){
      applyA(uv,Ax);
      { std::vector<real> t2(nDofQ);
        #pragma omp parallel for
        for (i32 i=0;i<nDofQ;i++) t2[i]=(real)(bvec[i]-Ax[i]);
        precond(t2,rr); }
      double beta=0;
      #pragma omp parallel for reduction(+:beta)
      for (i32 i=0;i<nDofQ;i++) beta+=(double)rr[i]*rr[i];
      beta=sqrt(beta);
      if (outer==0) beta0=beta>0?beta:1;
      cgRes=beta/beta0;
      if (getenv("SBM_DBG")) printf("    [gmres outer=%d it=%d pres=%.3e]\n",outer,it,beta/beta0);
      if (beta<=tol*beta0){ conv=true; break; }
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) V[0][i]=(real)(rr[i]/beta);
      std::fill(ss.begin(),ss.end(),0.0); ss[0]=beta; i32 jj=0;
      for (i32 j=0;j<m&&it<cgMaxIt;j++){ jj=j; it++;
        applyA(V[j],Ax);
        precond(Ax,w);
        for (i32 i2=0;i2<=j;i2++){ double hij=0;
          #pragma omp parallel for reduction(+:hij)
          for (i32 q=0;q<nDofQ;q++) hij+=(double)w[q]*V[i2][q];
          Hm[(size_t)i2*m+j]=hij;
          #pragma omp parallel for
          for (i32 q=0;q<nDofQ;q++) w[q]-=(real)(hij*V[i2][q]); }
        double hj1=0;
        #pragma omp parallel for reduction(+:hj1)
        for (i32 q=0;q<nDofQ;q++) hj1+=(double)w[q]*w[q];
        hj1=sqrt(hj1); Hm[(size_t)(j+1)*m+j]=hj1;
        if (hj1>1e-300){
          #pragma omp parallel for
          for (i32 q=0;q<nDofQ;q++) V[j+1][q]=(real)(w[q]/hj1); }
        for (i32 i2=0;i2<j;i2++){ double t=cs[i2]*Hm[(size_t)i2*m+j]+sn[i2]*Hm[(size_t)(i2+1)*m+j];
          Hm[(size_t)(i2+1)*m+j]=-sn[i2]*Hm[(size_t)i2*m+j]+cs[i2]*Hm[(size_t)(i2+1)*m+j]; Hm[(size_t)i2*m+j]=t; }
        double d0=Hm[(size_t)j*m+j],d1=Hm[(size_t)(j+1)*m+j],r2=sqrt(d0*d0+d1*d1); if(r2<1e-300)r2=1e-300;
        cs[j]=d0/r2; sn[j]=d1/r2; Hm[(size_t)j*m+j]=cs[j]*d0+sn[j]*d1; Hm[(size_t)(j+1)*m+j]=0;
        double t=cs[j]*ss[j]; ss[j+1]=-sn[j]*ss[j]; ss[j]=t;
        if (fabs(ss[j+1])<=tol*beta0) break; }
      i32 sz=jj+1;
      for (i32 i2=sz-1;i2>=0;i2--){ double s2=ss[i2];
        for (i32 k2=i2+1;k2<sz;k2++) s2-=Hm[(size_t)i2*m+k2]*yv[k2];
        yv[i2]=s2/Hm[(size_t)i2*m+i2]; }
      for (i32 i2=0;i2<sz;i2++){
        #pragma omp parallel for
        for (i32 q=0;q<nDofQ;q++) uv[q]+=(real)(yv[i2]*V[i2][q]); }
    }
  }
  cgIters=it;

  // -------------------------------------------------------------------------
  //  errors over the surrogate (tensor GLL; every element is full)
  // -------------------------------------------------------------------------
  double l2e=0,l2n=0,ene=0,enn=0,vol=0;
  #pragma omp parallel for schedule(dynamic,64) reduction(+:l2e,l2n,ene,enn,vol)
  for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
    real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
    for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
      real xr[3]={Bp.t[i],Bp.t[j],Bp.t[k]};
      double dw=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]*h*h*h;
      double JinvE[3][3],detJE=0;
      if (cyl){ metric(elems[e],xr,JinvE,detJE); dw=fabs(detJE)*Bp.wq[i]*Bp.wq[j]*Bp.wq[k]; }
      Bp.allVal(xr,vb); Bp.allGradRef(xr,gb);
      double uh[3]={0,0,0}, gh[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      double ulocE[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++) gather3(uv,nod[a],&ulocE[3*a]);
      for (i32 a=0;a<ndof;a++) for(i32 l=0;l<3;l++){ double ua=ulocE[3*a+l];
        uh[l]+=ua*vb[a];
        if (cyl){
          double gXa[3];
          for (i32 d2=0;d2<3;d2++) gXa[d2]=JinvE[0][d2]*gb[3*a]+JinvE[1][d2]*gb[3*a+1]+JinvE[2][d2]*gb[3*a+2];
          gh[l][0]+=ua*gXa[0]; gh[l][1]+=ua*gXa[1]; gh[l][2]+=ua*gXa[2];
        } else {
          gh[l][0]+=ua*gb[3*a]/h; gh[l][1]+=ua*gb[3*a+1]/h; gh[l][2]+=ua*gb[3*a+2]/h; } }
      real X[3]; ls.toPhys((elems[e].ci+xr[0])*h,(elems[e].cj+xr[1])*h,(elems[e].ck+xr[2])*h,X[0],X[1],X[2]);
      real ue[3]; prob.exactU(X[0],X[1],X[2],ue);
      real ge[3][3]; prob.exactGradU(X[0],X[1],X[2],ge);
      vol+=dw;
      for (i32 l=0;l<3;l++){ double dd=uh[l]-ue[l]; l2e+=dd*dd*dw; l2n+=(double)ue[l]*ue[l]*dw; }
      double ee[3][3],se[3][3],tre=0;
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) ee[i2][j2]=0.5*((gh[i2][j2]-ge[i2][j2])+(gh[j2][i2]-ge[j2][i2]));
      tre=ee[0][0]+ee[1][1]+ee[2][2];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) se[i2][j2]=2*mu*ee[i2][j2]+(i2==j2?lam*tre:0);
      double en=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) en+=se[i2][j2]*ee[i2][j2]; ene+=en*dw;
      double eeE[3][3],seE[3][3],trE=0;
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eeE[i2][j2]=0.5*(ge[i2][j2]+ge[j2][i2]);
      trE=eeE[0][0]+eeE[1][1]+eeE[2][2];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) seE[i2][j2]=2*mu*eeE[i2][j2]+(i2==j2?lam*trE:0);
      double enE=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) enE+=seE[i2][j2]*eeE[i2][j2]; enn+=enE*dw; } }
  errL2=sqrt(l2e); normL2=sqrt(l2n); errEnergy=sqrt(ene); normEnergy=sqrt(enn);
  volOmega=vol; areaGamma=0;

  // -------------------------------------------------------------------------
  //  report + VTU
  // -------------------------------------------------------------------------
  double ms=(sbNowUs()-t0)/1000.0;
  printf("active : %d surrogate elements (all full), %d boundary faces, %d ghost faces\n", nE, nBF, nGFQ);
  printf("dofs   : %d nodes -> %d unknowns   h = %.6g   p = %d\n", nNodeQ, nDofQ, (double)h, p);
  printf("geom   : |Omega~| = %.8g  (surrogate: centre-in cells; differs from Omega by O(h))\n", volOmega);
  printf("solve  : %s %d iters, rel res %.2e   (%.0f ms)\n", sbmSolver, cgIters, cgRes, ms);
  if ((prob.caseId==CASE_MMS || prob.caseId==CASE_MMS_CYL) && normL2>0)
    printf("error  : L2 %.6e (rel %.4e)   energy %.6e (rel %.4e)   [over Omega~]\n",
           errL2, errL2/normL2, errEnergy, errEnergy/normEnergy);

  if (wantVtu && !outTag.empty()){ mkdir("output",0755);
    std::string fn="output/"+outTag+"_femsbm.vtu"; std::ofstream os(fn.c_str(),std::ios::binary);
    i64 nSub=(i64)nE*p*p*p; static const i32 HEX[8][3]={{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    os<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
      <<"  <UnstructuredGrid>\n    <Piece NumberOfPoints=\""<<nNodeQ<<"\" NumberOfCells=\""<<nSub<<"\">\n";
    os<<"      <Points>\n        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (i32 nd=0;nd<nNodeQ;nd++) os<<(float)nodeXQ[3*nd]<<" "<<(float)nodeXQ[3*nd+1]<<" "<<(float)nodeXQ[3*nd+2]<<"\n";
    os<<"        </DataArray>\n      </Points>\n";
    os<<"      <PointData Vectors=\"u\">\n        <DataArray type=\"Float32\" Name=\"u\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (i32 nd=0;nd<nNodeQ;nd++){ double u3[3]; gather3(uv,nd,u3);
      os<<(float)u3[0]<<" "<<(float)u3[1]<<" "<<(float)u3[2]<<"\n"; }
    os<<"        </DataArray>\n      </PointData>\n";
    os<<"      <Cells>\n        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
      for (i32 sk=0;sk<p;sk++)for(i32 sj=0;sj<p;sj++)for(i32 si=0;si<p;si++){
        for (i32 v=0;v<8;v++){ i32 ii=si+HEX[v][0],jj=sj+HEX[v][1],kk=sk+HEX[v][2]; os<<nod[ii+n*(jj+n*kk)]<<" "; } os<<"\n"; } }
    os<<"        </DataArray>\n        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (i64 s=0;s<nSub;s++) os<<8*(s+1)<<"\n";
    os<<"        </DataArray>\n        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (i64 s=0;s<nSub;s++) os<<"12\n";
    os<<"        </DataArray>\n      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
    printf("wrote %s\n", fn.c_str());
  }
}
