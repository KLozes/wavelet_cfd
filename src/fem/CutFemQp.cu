//
// Higher-order (Qp) path of the CutFEM solver -- CutFemSolver::runQp().
//
// Production realization of the method verified standalone in QpMms.cu, reusing
// the sparse octree + oracle (buildMesh) for the active block set and the
// verified modules QpBasis / PolyFit / SayeQuad / QpElem.  The p=1 GPU path
// (run()) is untouched; run() dispatches here when femOrder>=2.
//
// Two geometry modes:
//   * CARTESIAN (coordMode 0): the element is the cube [0,h]^3, Jacobian h*I;
//     interior elements share one reference matrix, cut elements carry their own.
//   * CYLINDRICAL (coordMode 1): the element is a curved (r, r*theta', z) brick.
//     The physical Jacobian is evaluated ANALYTICALLY from the cylindrical map at
//     every quadrature point (isoparametric-exact), so EVERY element carries its
//     own matrix.  The Nitsche surface uses the Nanson-transformed physical
//     normal and area, and the one-pitch sector is closed by the cyclic node tie
//     u(theta+pitch)=R(pitch)u(theta) -- a rotation applied at gather/scatter.
//
// The level set is sampled by the oracle at the (p+1)^3 GLL solution nodes; the
// Qp normal-derivative-jump ghost penalty (computational frame, l=1..p) on cut-
// element faces restores conditioning and the O(h^{p+1}) rate on slivers.
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
#include "SayeQuad.h"
#include "QpElem.h"

static inline u64 qpKey(i32 I, i32 J, i32 K) {
  return (u64)I | ((u64)J << 21) | ((u64)K << 42);
}
static inline u64 qpCellKey(i32 i, i32 j, i32 k) { return qpKey(i,j,k); }
static inline void qpBlockDec(u64 loc, i32 &i, i32 &j, i32 &k) {
  k = (i32)((loc >> 40) & ((1u<<20)-1)) - 1;
  j = (i32)((loc >> 20) & ((1u<<20)-1)) - 1;
  i = (i32)( loc        & ((1u<<20)-1)) - 1;
}
static long qpNowUs(void) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
// invert a 3x3, return det
static inline double inv3(const double J[3][3], double Ji[3][3]) {
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
}

void CutFemSolver::runQp(void) {
  const i32 p = femOrder;
  QpBasis Bp; Bp.init(p);
  const i32 n = p+1, ndof = n*n*n, ndof3 = 3*ndof, mG = 2*ndof3;
  const real h = cellSize();
  const double mu = prob.mu, lam = prob.lam;
  const double gammaD_ = 100.0*(2*mu+lam)*p*p;   // Nitsche penalty (verified scaling)
  const double gammaG_ = 0.1*(2*mu+lam);         // ghost-penalty coefficient
  const bool cyl = (ls.coordMode == 1);
  const bool per = (periodic != 0);
  const double cph = cos((double)pitchAngle), sph = sin((double)pitchAngle);

  if (ls.coordMode != 0 && !cyl) {
    printf("ERROR: Qp path supports coordMode 0 (Cartesian) or 1 (cylindrical) only\n"); return; }
  if (p > QP_MAX) { printf("ERROR: femOrder %d exceeds QP_MAX=%d\n", p, QP_MAX); return; }

  printf("higher : Qp CutFEM (Saye cut quadrature), p=%d  %s%s  gammaD=%.4g gammaG=%.4g\n",
         p, cyl?"CYLINDRICAL isoparametric":"Cartesian", per?" + cyclic pitch tie":"",
         gammaD_, gammaG_);

  long t0 = qpNowUs();
  initialize();
  buildMesh();
  real tgll[PNC]; gllNodes(p, tgll);

  // -------------------------------------------------------------------------
  //  active elements: sample phi at the (p+1)^3 GLL nodes via the oracle
  // -------------------------------------------------------------------------
  struct QElem { i32 ci, cj, ck; bool cut; };
  std::vector<QElem> elems;
  std::vector<real>  ephi;
  const i32 nB = hashTable.nKeys;
  #pragma omp parallel
  {
    std::vector<QElem> le; std::vector<real> lp;
    #pragma omp for schedule(dynamic,4) nowait
    for (i32 b = 0; b < nB; b++) {
      u64 loc = bLocList[b];
      if (loc == kEmpty) continue;
      i32 ib, jb, kb; qpBlockDec(loc, ib, jb, kb);
      for (i32 cz = 0; cz < blockSize; cz++)
      for (i32 cy = 0; cy < blockSize; cy++)
      for (i32 cx = 0; cx < blockSize; cx++) {
        i32 ci = ib*blockSize + cx, cj = jb*blockSize + cy, ck = kb*blockSize + cz;
        real v[PNC*PNC*PNC];
        bool anyNeg=false, anyPos=false;
        for (i32 k=0;k<n;k++) for(i32 j=0;j<n;j++) for(i32 i=0;i<n;i++){
          real f = ls.phi((ci+tgll[i])*h, (cj+tgll[j])*h, (ck+tgll[k])*h);
          v[i+n*(j+n*k)]=f; if(f<0)anyNeg=true; else anyPos=true;
        }
        if (!anyNeg) continue;
        QElem E; E.ci=ci; E.cj=cj; E.ck=ck; E.cut=anyPos;
        le.push_back(E);
        for (i32 t=0;t<ndof;t++) lp.push_back(v[t]);
      }
    }
    #pragma omp critical
    { for (auto &E:le) elems.push_back(E); for (real f:lp) ephi.push_back(f); }
  }
  i32 nE = (i32)elems.size();
  if (nE == 0) { printf("ERROR: no active elements\n"); return; }

  // -------------------------------------------------------------------------
  //  node numbering (p-finer lattice) + cyclic dof map (master + rotation)
  // -------------------------------------------------------------------------
  std::unordered_map<u64,i32> nodeId;
  nodeId.reserve((size_t)nE*ndof);
  std::vector<i32> eNodeQ((size_t)nE*ndof);
  std::vector<i32> nI, nJ, nK;                 // per-node lattice coords
  auto gcoord=[&](const QElem&E,i32 a,i32&I,i32&J,i32&K){ i32 i=a%n,j=(a/n)%n,k=a/(n*n);
    I=p*E.ci+i; J=p*E.cj+j; K=p*E.ck+k; };
  i32 nNodeQ=0;
  for (i32 e=0;e<nE;e++) for (i32 a=0;a<ndof;a++){
    i32 I,J,K; gcoord(elems[e],a,I,J,K); u64 key=qpKey(I,J,K);
    auto it=nodeId.find(key); i32 id;
    if (it==nodeId.end()){ id=nNodeQ++; nodeId[key]=id; nI.push_back(I); nJ.push_back(J); nK.push_back(K); }
    else id=it->second;
    eNodeQ[(size_t)e*ndof+a]=id;
  }
  const i32 Jmax = p*nThetaCells;              // theta slave column (periodic)
  std::vector<i32> realIdx(nNodeQ,-1);
  std::vector<char> rotFlag(nNodeQ,0);
  i32 nDofNode=0, nTie=0, nOrphan=0;
  for (i32 nd=0;nd<nNodeQ;nd++){
    if (per && nJ[nd]==Jmax) continue;         // slave, resolved below
    realIdx[nd]=nDofNode++;
  }
  for (i32 nd=0;nd<nNodeQ;nd++){
    if (realIdx[nd]>=0) continue;
    auto it=nodeId.find(qpKey(nI[nd],0,nK[nd]));   // master at J=0
    if (it==nodeId.end()){ realIdx[nd]=nDofNode++; nOrphan++; }   // no partner: own dof
    else { realIdx[nd]=realIdx[it->second]; rotFlag[nd]=1; nTie++; }
  }
  const i32 nDofQ=3*nDofNode;
  if (per) printf("cyclic : %d Qp nodes tied across the pitch (%d unmatched kept free), %d -> %d dofs\n",
                  nTie, nOrphan, 3*nNodeQ, nDofQ);

  // gather/scatter with the pitch rotation (identity when non-periodic).
  // Vectors are stored in `real` (fp32 in the wavefem build, fp64 in wavefem_dp):
  // with the matrix bank gone (matrix-free below), storage is the memory driver,
  // so single precision halves it.  Element APPLY still computes in double.
  auto gather3=[&](const std::vector<real>&x,i32 nd,double u[3]){
    i32 b=3*realIdx[nd];
    if (rotFlag[nd]){ u[0]=cph*x[b]-sph*x[b+1]; u[1]=sph*x[b]+cph*x[b+1]; u[2]=x[b+2]; }
    else { u[0]=x[b]; u[1]=x[b+1]; u[2]=x[b+2]; }
  };
  auto scatter3=[&](std::vector<real>&y,i32 nd,const double c[3]){
    i32 b=3*realIdx[nd]; double a0,a1,a2;
    if (rotFlag[nd]){ a0=cph*c[0]+sph*c[1]; a1=-sph*c[0]+cph*c[1]; a2=c[2]; }
    else { a0=c[0]; a1=c[1]; a2=c[2]; }
    #pragma omp atomic
    y[b]+=a0;
    #pragma omp atomic
    y[b+1]+=a1;
    #pragma omp atomic
    y[b+2]+=a2;
  };

  // node physical positions; per-dof "owner" position (for rigid-mode build/VTU)
  std::vector<real> nodeXQ((size_t)3*nNodeQ);
  std::vector<double> dofPos((size_t)3*nDofNode, 0.0);
  for (i32 nd=0;nd<nNodeQ;nd++){
    real X0,X1,X2; ls.toPhys((real)nI[nd]*h/p,(real)nJ[nd]*h/p,(real)nK[nd]*h/p,X0,X1,X2);
    nodeXQ[3*nd]=X0; nodeXQ[3*nd+1]=X1; nodeXQ[3*nd+2]=X2;
    if (!rotFlag[nd]){ i32 d=realIdx[nd]; dofPos[3*d]=X0; dofPos[3*d+1]=X1; dofPos[3*d+2]=X2; }
  }

  // -------------------------------------------------------------------------
  //  per-element level-set polynomial + Saye cut quadrature
  // -------------------------------------------------------------------------
  std::vector<PolyND> ePoly(nE);
  std::vector<i32> cutIdx(nE,-1); i32 nCutQ=0;
  for (i32 e=0;e<nE;e++){ ePoly[e]=fitPoly3(p,&ephi[(size_t)e*ndof]); if (elems[e].cut) cutIdx[e]=nCutQ++; }
  std::vector<SayeNode> volPool, surfPool;
  std::vector<i32> volOff(nCutQ+1,0), surfOff(nCutQ+1,0);
  { std::vector<SayeNode> arena(1<<18), out(1<<16);
    for (i32 e=0;e<nE;e++) if (elems[e].cut){ i32 c=cutIdx[e];
      SayeArena ar; ar.buf=arena.data(); ar.cap=1<<18; ar.top=0;
      SayeSet ov; ov.p=out.data(); ov.n=0; ov.cap=1<<16; ov.ovf=false;
      sayeVolume(ePoly[e],&ov,&ar,SayeCfg::def());
      for (i32 q=0;q<ov.n;q++) volPool.push_back(ov.p[q]); volOff[c+1]=(i32)volPool.size();
      SayeArena ar2; ar2.buf=arena.data(); ar2.cap=1<<18; ar2.top=0;
      SayeSet sv; sv.p=out.data(); sv.n=0; sv.cap=1<<16; sv.ovf=false;
      sayeSurface(ePoly[e],&sv,&ar2,SayeCfg::def());
      for (i32 q=0;q<sv.n;q++) surfPool.push_back(sv.p[q]); surfOff[c+1]=(i32)surfPool.size();
    } }

  // ghost faces (interior faces of cut elements; wrap the theta seam if periodic)
  std::unordered_map<u64,i32> cellId; cellId.reserve((size_t)nE*2);
  for (i32 e=0;e<nE;e++) cellId[qpCellKey(elems[e].ci,elems[e].cj,elems[e].ck)]=e;
  struct GF{ i32 eM,eP,d; };
  std::vector<GF> gf;
  for (i32 e=0;e<nE;e++){ i32 cc[3]={elems[e].ci,elems[e].cj,elems[e].ck};
    for (i32 d=0;d<3;d++){ i32 nb[3]={cc[0],cc[1],cc[2]}; nb[d]++;
      if (per && d==1 && nb[1]==nThetaCells) nb[1]=0;      // theta seam wraps
      auto it=cellId.find(qpCellKey(nb[0],nb[1],nb[2])); if (it==cellId.end()) continue;
      i32 ep=it->second; if (elems[e].cut||elems[ep].cut) gf.push_back({e,ep,d}); } }
  i32 nGFQ=(i32)gf.size();

  double Dl0[QP_MAX+1][QN_MAX], Dl1[QP_MAX+1][QN_MAX];
  { double Dp[QN_MAX][QN_MAX];
    for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++) Dp[i][a]=Bp.D[i][a];
    for (i32 l=1;l<=p;l++){ for (i32 a=0;a<n;a++){ Dl0[l][a]=Dp[0][a]; Dl1[l][a]=Dp[n-1][a]; }
      if (l<p){ double Nw[QN_MAX][QN_MAX];
        for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++){ double s=0; for(i32 m=0;m<n;m++) s+=Dp[i][m]*Bp.D[m][a]; Nw[i][a]=s; }
        for (i32 i=0;i<n;i++) for (i32 a=0;a<n;a++) Dp[i][a]=Nw[i][a]; } } }

  auto physOf=[&](const QElem&E,const real xr[3],real X[3]){
    ls.toPhys((E.ci+xr[0])*h,(E.cj+xr[1])*h,(E.ck+xr[2])*h,X[0],X[1],X[2]); };

  // cylindrical metric at a reference point: Jinv (=Jref^{-1}), detJ (analytic)
  auto metric=[&](const QElem&E,const real xr[3],double Jinv[3][3],double&detJ){
    double r=(E.ci+xr[0])*h+ls.org[0], s=(E.cj+xr[1])*h+ls.org[1], z=(E.ck+xr[2])*h+ls.org[2];
    double th=s/ls.rRef+(double)ls.thc((real)z), thp=(double)ls.thcSlope((real)z);
    double ct=cos(th), st=sin(th);
    double J[3][3]={{ h*ct, h*(-r*st/ls.rRef), h*(-r*st*thp) },
                    { h*st, h*( r*ct/ls.rRef), h*( r*ct*thp) },
                    { 0,    0,                 h            }};
    detJ=inv3(J,Jinv);
  };

  // -------------------------------------------------------------------------
  //  local element operators
  // -------------------------------------------------------------------------
  // CARTESIAN cut-element local apply (bulk Saye + Nitsche), reference frame
  auto applyCutCart=[&](i32 e,const double*uloc,double*yloc){
    i32 c=cutIdx[e]; i32 nv=volOff[c+1]-volOff[c]; const SayeNode*vn=&volPool[volOff[c]];
    real ul[3*QN_MAX*QN_MAX*QN_MAX], yl[3*QN_MAX*QN_MAX*QN_MAX];
    for (i32 a=0;a<ndof3;a++) ul[a]=(real)uloc[a];
    qpElemCoreSaye(Bp,(real)mu,(real)lam,h,vn,nv,ul,yl);
    for (i32 a=0;a<ndof3;a++) yloc[a]=yl[a];
    i32 ns=surfOff[c+1]-surfOff[c]; const SayeNode*sn=&surfPool[surfOff[c]];
    real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
    for (i32 q=0;q<ns;q++){ real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]};
      real X[3]; physOf(elems[e],xr,X); if (!prob.isDirichlet(X[0],X[1],X[2])) continue;
      double nn[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]};
      Bp.allGradRef(xr,gb); Bp.allVal(xr,vb); double hw=sn[q].w*h;
      double uval[3]={0,0,0}, gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ uval[i2]+=uloc[3*a+i2]*vb[a];
        gradU[i2][0]+=uloc[3*a+i2]*gb[3*a+0]; gradU[i2][1]+=uloc[3*a+i2]*gb[3*a+1]; gradU[i2][2]+=uloc[3*a+i2]*gb[3*a+2]; }
      double eps[3][3],tr=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2]; double sig[3][3];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*tr:0);
      double tu[3]; for(i32 i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nn[0]+sig[i2][1]*nn[1]+sig[i2][2]*nn[2];
      double un=uval[0]*nn[0]+uval[1]*nn[1]+uval[2]*nn[2];
      for (i32 a=0;a<ndof;a++){ double gan=gb[3*a+0]*nn[0]+gb[3*a+1]*nn[1]+gb[3*a+2]*nn[2];
        double ugb=uval[0]*gb[3*a+0]+uval[1]*gb[3*a+1]+uval[2]*gb[3*a+2];
        for (i32 l=0;l<3;l++){ double t1=-tu[l]*vb[a];
          double t2=-(mu*(uval[l]*gan+ugb*nn[l])+lam*gb[3*a+l]*un); double t3=gammaD_*uval[l]*vb[a];
          yloc[3*a+l]+=hw*(t1+t2+t3); } } }
  };

  // CYLINDRICAL element local apply (curved bulk + Nanson Nitsche).  nitsche=false
  // gives bulk only (for the rigid-body self-check).
  auto applyElemCyl=[&](i32 e,const double*uloc,double*yloc,bool nitsche){
    for (i32 a=0;a<ndof3;a++) yloc[a]=0;
    // volume quadrature: tensor GLL (uncut) or Saye (cut)
    std::vector<SayeNode> tens; const SayeNode*vn; i32 nv;
    if (!elems[e].cut){ tens.resize(ndof); i32 qi=0;
      for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){
        tens[qi].x[0]=Bp.t[i];tens[qi].x[1]=Bp.t[j];tens[qi].x[2]=Bp.t[k];
        tens[qi].w=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]; qi++; } vn=tens.data(); nv=ndof;
    } else { i32 c=cutIdx[e]; nv=volOff[c+1]-volOff[c]; vn=&volPool[volOff[c]]; }
    real gb[3*QN_MAX*QN_MAX*QN_MAX];
    for (i32 q=0;q<nv;q++){ real xr[3]={vn[q].x[0],vn[q].x[1],vn[q].x[2]};
      double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
      Bp.allGradRef(xr,gb);
      double gX[QN_MAX*QN_MAX*QN_MAX][3];
      for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
      double gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ gradU[i2][0]+=uloc[3*a+i2]*gX[a][0];
        gradU[i2][1]+=uloc[3*a+i2]*gX[a][1]; gradU[i2][2]+=uloc[3*a+i2]*gX[a][2]; }
      double eps[3][3],tr=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2]; double sig[3][3];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*tr:0);
      double wdet=fabs(detJ)*vn[q].w;
      for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++)
        yloc[3*a+i2]+=wdet*(sig[i2][0]*gX[a][0]+sig[i2][1]*gX[a][1]+sig[i2][2]*gX[a][2]); }
    if (!nitsche || !elems[e].cut) return;
    // Nitsche on the physical (Nanson) surface at Dirichlet points
    i32 c=cutIdx[e]; i32 ns=surfOff[c+1]-surfOff[c]; const SayeNode*sn=&surfPool[surfOff[c]];
    real vb[QN_MAX*QN_MAX*QN_MAX];
    for (i32 q=0;q<ns;q++){ real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]};
      real X[3]; physOf(elems[e],xr,X); if (!prob.isDirichlet(X[0],X[1],X[2])) continue;
      double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
      double nref[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]};
      double nraw[3]; for(i32 i2=0;i2<3;i2++) nraw[i2]=Jinv[0][i2]*nref[0]+Jinv[1][i2]*nref[1]+Jinv[2][i2]*nref[2];
      double nmag=sqrt(nraw[0]*nraw[0]+nraw[1]*nraw[1]+nraw[2]*nraw[2]); if (nmag<=0) continue;
      double nP[3]={nraw[0]/nmag,nraw[1]/nmag,nraw[2]/nmag};
      double dS=fabs(detJ)*nmag*sn[q].w;
      Bp.allGradRef(xr,gb); Bp.allVal(xr,vb);
      double gX[QN_MAX*QN_MAX*QN_MAX][3];
      for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
      double uval[3]={0,0,0}, gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
      for (i32 a=0;a<ndof;a++) for(i32 i2=0;i2<3;i2++){ uval[i2]+=uloc[3*a+i2]*vb[a];
        gradU[i2][0]+=uloc[3*a+i2]*gX[a][0]; gradU[i2][1]+=uloc[3*a+i2]*gX[a][1]; gradU[i2][2]+=uloc[3*a+i2]*gX[a][2]; }
      double eps[3][3],tr=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
      tr=eps[0][0]+eps[1][1]+eps[2][2]; double sig[3][3];
      for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) sig[i2][j2]=2*mu*eps[i2][j2]+(i2==j2?lam*tr:0);
      double tu[3]; for(i32 i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nP[0]+sig[i2][1]*nP[1]+sig[i2][2]*nP[2];
      double un=uval[0]*nP[0]+uval[1]*nP[1]+uval[2]*nP[2];
      // penalty carries the 1/h Nitsche scaling (Cartesian folds it into hw=sn.w*h;
      // here dS is the FULL physical area, so divide the penalty coefficient by h)
      double penC=gammaD_/h;
      for (i32 a=0;a<ndof;a++){ double gan=gX[a][0]*nP[0]+gX[a][1]*nP[1]+gX[a][2]*nP[2];
        double ugb=uval[0]*gX[a][0]+uval[1]*gX[a][1]+uval[2]*gX[a][2];
        for (i32 l=0;l<3;l++){ double t1=-tu[l]*vb[a];
          double t2=-(mu*(uval[l]*gan+ugb*nP[l])+lam*gX[a][l]*un); double t3=penC*uval[l]*vb[a];
          yloc[3*a+l]+=dS*(t1+t2+t3); } } }
  };

  // ---- consistency self-check (cylindrical curved bulk operator) ----
  // A CONSTANT displacement is in Q_p exactly, so eps=0 and K_bulk*u_const must
  // be machine-zero regardless of the curved metric -- this validates that the
  // analytic Jacobian is applied consistently (partition-of-unity of grad).  (A
  // rigid ROTATION is only O(h^{p+1}) here because the geometry is exact, not
  // isoparametric-interpolated, so it is not a machine-zero test.)  Also checks
  // the reference-cell volume integral of |detJ| against the analytic curved
  // cell volume on a sample element, which validates the determinant.
  if (cyl) {
    double worstC=0;
    for (i32 e=0;e<nE;e++){
      double uc[3*QN_MAX*QN_MAX*QN_MAX], yc[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ uc[3*a]=1.0; uc[3*a+1]=-0.7; uc[3*a+2]=0.3; }  // const
      applyElemCyl(e,uc,yc,false);
      double yn2=0; for (i32 a=0;a<ndof3;a++) yn2+=yc[a]*yc[a];
      worstC=std::max(worstC,sqrt(yn2)); }
    // |detJ| volume of an uncut sample cell vs the analytic curved-brick volume
    // int_cell dV = int_{r0}^{r1} int_{s0}^{s1} int_{z0}^{z1} r dr ds dz
    //             = (r1^2-r0^2)/2 * (s1-s0) * (z1-z0)   (theta = s/rRef+thc(z))
    double volErr=-1;
    for (i32 e=0;e<nE && volErr<0;e++) if (!elems[e].cut){
      double num=0; GaussRule g=gaussLegendre(5);
      for (i32 i=0;i<g.n;i++)for(i32 j=0;j<g.n;j++)for(i32 k=0;k<g.n;k++){
        real xr[3]={(real)g.x[i],(real)g.x[j],(real)g.x[k]}; double Jinv[3][3],detJ;
        metric(elems[e],xr,Jinv,detJ); num+=fabs(detJ)*g.w[i]*g.w[j]*g.w[k]; }
      double r0=elems[e].ci*h+ls.org[0], r1=r0+h;
      double ana=0.5*(r1*r1-r0*r0)*h*h/ls.rRef;   // (s1-s0)=h, (z1-z0)=h, ds=h -> /rRef in theta but dV=r dr ds dz uses s directly
      // dV_phys = r dr dtheta dz = r dr (ds/rRef) dz ; here s is the arc coord so dtheta=ds/rRef
      volErr=fabs(num-ana)/std::max(1e-30,ana);
    }
    printf("check  : curved-bulk const-field residual %.3e %s;  |detJ| cell-volume rel err %.3e\n",
           worstC, worstC<1e-9?"[OK]":"[WARN]", volErr);
  }

  // -------------------------------------------------------------------------
  //  MATRIX-FREE operator.  The cylindrical per-element matrix bank
  //  (~n_elem * (3(p+1)^3)^2 reals) was the memory driver; instead of storing it,
  //  recompute each element's action per CG apply by calling the SAME verified
  //  local applies used to build it (so the operator is algebraically identical,
  //  just recomputed).  Only the tiny reference-invariant per-axis ghost matrices
  //  are kept.  Combined with the fp32 `real` vectors, this removes the tens-of-GB
  //  bank and leaves the Saye pools + O(dof) vectors as the footprint.
  // -------------------------------------------------------------------------
  auto applyElem=[&](i32 e,const double*uloc,double*yloc){
    if (cyl) { applyElemCyl(e,uloc,yloc,true); return; }
    if (elems[e].cut) { applyCutCart(e,uloc,yloc); return; }
    real ul[3*QN_MAX*QN_MAX*QN_MAX], yl[3*QN_MAX*QN_MAX*QN_MAX];
    for (i32 a=0;a<ndof3;a++) ul[a]=(real)uloc[a];
    qpElemUncut(Bp,(real)mu,(real)lam,h,ul,yl);
    for (i32 a=0;a<ndof3;a++) yloc[a]=yl[a];
  };

  // ghost matrices (computational frame, one per axis; shared, tiny)
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
  //  operator apply (dense local matvecs + rotation-aware gather/scatter)
  // -------------------------------------------------------------------------
  auto applyA=[&](const std::vector<real>&x,std::vector<real>&y){
    std::fill(y.begin(),y.end(),(real)0);
    #pragma omp parallel for schedule(dynamic,64)
    for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
      double uloc[3*QN_MAX*QN_MAX*QN_MAX], yloc[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ double u3[3]; gather3(x,nod[a],u3); uloc[3*a]=u3[0]; uloc[3*a+1]=u3[1]; uloc[3*a+2]=u3[2]; }
      applyElem(e,uloc,yloc);                    // matrix-free: recompute the action
      for (i32 a=0;a<ndof;a++){ double c3[3]={yloc[3*a],yloc[3*a+1],yloc[3*a+2]}; scatter3(y,nod[a],c3); } }
    #pragma omp parallel for schedule(dynamic,64)
    for (i32 f=0;f<nGFQ;f++){ const GF&F=gf[f];
      const i32*nodM=&eNodeQ[(size_t)F.eM*ndof], *nodP=&eNodeQ[(size_t)F.eP*ndof];
      double uMP[2*3*QN_MAX*QN_MAX*QN_MAX], yMP[2*3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ double u3[3]; gather3(x,nodM[a],u3); uMP[3*a]=u3[0];uMP[3*a+1]=u3[1];uMP[3*a+2]=u3[2];
        gather3(x,nodP[a],u3); uMP[ndof3+3*a]=u3[0];uMP[ndof3+3*a+1]=u3[1];uMP[ndof3+3*a+2]=u3[2]; }
      const double*K=Kghost[F.d].data();
      for (i32 r=0;r<mG;r++){ double s=0; const double*Kr=&K[(size_t)r*mG]; for(i32 c=0;c<mG;c++) s+=Kr[c]*uMP[c]; yMP[r]=s; }
      for (i32 a=0;a<ndof;a++){ double cm[3]={yMP[3*a],yMP[3*a+1],yMP[3*a+2]}; scatter3(y,nodM[a],cm);
        double cp[3]={yMP[ndof3+3*a],yMP[ndof3+3*a+1],yMP[ndof3+3*a+2]}; scatter3(y,nodP[a],cp); } }
  };

  // -------------------------------------------------------------------------
  //  RHS (body force + Nitsche data + Neumann traction) and Jacobi diagonal
  // -------------------------------------------------------------------------
  std::vector<real> bvec(nDofQ,(real)0);                 // fp32 storage (wavefem build)
  std::vector<double> diagNode((size_t)3*nNodeQ,0.0);   // per-node, folded to dof below (temp, freed)
  {
    for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
      double bloc[3*QN_MAX*QN_MAX*QN_MAX]; for (i32 a=0;a<ndof3;a++) bloc[a]=0;
      real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
      // volume quadrature source
      std::vector<SayeNode> tens; const SayeNode*vn; i32 nv;
      if (!elems[e].cut){ tens.resize(ndof); i32 qi=0;
        for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){ tens[qi].x[0]=Bp.t[i];tens[qi].x[1]=Bp.t[j];tens[qi].x[2]=Bp.t[k];
          tens[qi].w=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]; qi++; } vn=tens.data(); nv=ndof;
      } else { i32 c=cutIdx[e]; nv=volOff[c+1]-volOff[c]; vn=&volPool[volOff[c]]; }
      for (i32 q=0;q<nv;q++){ real xr[3]={vn[q].x[0],vn[q].x[1],vn[q].x[2]};
        Bp.allGradRef(xr,gb); Bp.allVal(xr,vb); real X[3]; physOf(elems[e],xr,X);
        real f[3]; prob.bodyForce(X[0],X[1],X[2],f);
        double gX[QN_MAX*QN_MAX*QN_MAX][3]; double wdet, wdiag;
        if (cyl){ double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
          for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
          wdet=fabs(detJ)*vn[q].w; wdiag=wdet;
        } else { for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=gb[3*a+d]/h;
          wdet=vn[q].w*h*h*h; wdiag=vn[q].w*h; }
        for (i32 a=0;a<ndof;a++){ for (i32 l=0;l<3;l++) bloc[3*a+l]+=wdet*f[l]*vb[a];
          double gsq=gX[a][0]*gX[a][0]+gX[a][1]*gX[a][1]+gX[a][2]*gX[a][2];
          double sc = cyl ? 1.0 : (h*h);   // diag uses physical grads; Cart gX=gb/h so rescale to match wdiag*h powers
          for (i32 l=0;l<3;l++) diagNode[3*nod[a]+l]+=wdiag*(mu*(gsq+gX[a][l]*gX[a][l])+lam*gX[a][l]*gX[a][l])*sc; } }
      // surface source
      if (elems[e].cut){ i32 c=cutIdx[e]; i32 ns=surfOff[c+1]-surfOff[c]; const SayeNode*sn=&surfPool[surfOff[c]];
        for (i32 q=0;q<ns;q++){ real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]}; real X[3]; physOf(elems[e],xr,X);
          Bp.allGradRef(xr,gb); Bp.allVal(xr,vb);
          double nP[3], dS; double gX[QN_MAX*QN_MAX*QN_MAX][3];
          if (cyl){ double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
            double nref[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]}, nraw[3];
            for(i32 i2=0;i2<3;i2++) nraw[i2]=Jinv[0][i2]*nref[0]+Jinv[1][i2]*nref[1]+Jinv[2][i2]*nref[2];
            double nmag=sqrt(nraw[0]*nraw[0]+nraw[1]*nraw[1]+nraw[2]*nraw[2]); if (nmag<=0) continue;
            nP[0]=nraw[0]/nmag; nP[1]=nraw[1]/nmag; nP[2]=nraw[2]/nmag; dS=fabs(detJ)*nmag*sn[q].w;
            for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
          } else { nP[0]=sn[q].n[0]; nP[1]=sn[q].n[1]; nP[2]=sn[q].n[2]; dS=sn[q].w*h;   // Cart: reference frame, one h
            for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=gb[3*a+d]; }
          // penalty coefficient: Cartesian folds 1/h into dS=sn.w*h; cylindrical
          // uses the full physical dS, so the penalty carries the explicit 1/h
          double penC = cyl ? (gammaD_/h) : gammaD_;
          bool dir=prob.isDirichlet(X[0],X[1],X[2]);
          if (dir){ real g[3]; prob.dirichletData(X[0],X[1],X[2],g);
            double gn=g[0]*nP[0]+g[1]*nP[1]+g[2]*nP[2];
            for (i32 a=0;a<ndof;a++){ double gan=gX[a][0]*nP[0]+gX[a][1]*nP[1]+gX[a][2]*nP[2];
              double ggb=g[0]*gX[a][0]+g[1]*gX[a][1]+g[2]*gX[a][2];
              for (i32 l=0;l<3;l++){ double rhs=-(mu*(g[l]*gan+ggb*nP[l])+lam*gX[a][l]*gn)+penC*g[l]*vb[a];
                bloc[3*a+l]+=dS*rhs; diagNode[3*nod[a]+l]+=dS*penC*vb[a]*vb[a]; } }
          } else { real nr[3]={(real)nP[0],(real)nP[1],(real)nP[2]}, g[3]; prob.neumannData(X[0],X[1],X[2],nr,g);
            for (i32 a=0;a<ndof;a++) for(i32 l=0;l<3;l++) bloc[3*a+l]+=dS*g[l]*vb[a]; } } }
      // scatter the element load through the cyclic tie
      for (i32 a=0;a<ndof;a++){ double c3[3]={bloc[3*a],bloc[3*a+1],bloc[3*a+2]}; scatter3(bvec,nod[a],c3); }
    }
    // ghost-penalty diagonal (computational frame)
    for (i32 f=0;f<nGFQ;f++){ const GF&F=gf[f];
      const i32*nodM=&eNodeQ[(size_t)F.eM*ndof], *nodP=&eNodeQ[(size_t)F.eP*ndof];
      i32 d=F.d,t1=(d+1)%3,t2=(d+2)%3; GaussRule g1=gaussLegendre(p+1);
      for (i32 q1=0;q1<g1.n;q1++) for (i32 q2=0;q2<g1.n;q2++){ double w=g1.w[q1]*g1.w[q2];
        real L1[QN_MAX],L2[QN_MAX]; Bp.basis1(g1.x[q1],L1); Bp.basis1(g1.x[q2],L2);
        for (i32 a=0;a<ndof;a++){ i32 idx[3]={a%n,(a/n)%n,a/(n*n)}; i32 idn=idx[d]; double Lt=L1[idx[t1]]*L2[idx[t2]];
          for (i32 l=1;l<=p;l++){ double cP=Dl0[l][idn]*Lt, cM=Dl1[l][idn]*Lt, cf=gammaG_*h*w;
            for (i32 comp=0;comp<3;comp++){ diagNode[3*nodP[a]+comp]+=cf*cP*cP; diagNode[3*nodM[a]+comp]+=cf*cM*cM; } } } } }
    // fold node diagonal into dof space (rotation ignored: preconditioner only)
    std::vector<real> diagv(nDofQ,(real)0);
    for (i32 nd=0;nd<nNodeQ;nd++){ i32 b=3*realIdx[nd];
      for (i32 l=0;l<3;l++) diagv[b+l]+=(real)diagNode[3*nd+l]; }
    for (i32 i=0;i<nDofQ;i++) if (diagv[i]<=0) diagv[i]=(real)1;

    // ---- Jacobi-PCG (fp32 vectors, fp64 scalars/accumulators = mixed precision) ----
    std::vector<real> uv(nDofQ,(real)0), r=bvec, z(nDofQ), pd(nDofQ), Ap(nDofQ);
    #pragma omp parallel for
    for (i32 i=0;i<nDofQ;i++) z[i]=r[i]/diagv[i];
    pd=z;
    double rz=0; for (i32 i=0;i<nDofQ;i++) rz+=r[i]*z[i];
    double bnorm=0; for (i32 i=0;i<nDofQ;i++) bnorm+=bvec[i]*bvec[i]; bnorm=sqrt(bnorm);
    i32 it=0;
    for (; it<cgMaxIt && bnorm>0; it++){
      applyA(pd,Ap);
      double pAp=0;
      #pragma omp parallel for reduction(+:pAp)
      for (i32 i=0;i<nDofQ;i++) pAp+=pd[i]*Ap[i];
      if (!(pAp>0)){ printf("WARNING: Qp CG breakdown pAp=%.3e\n",pAp); break; }
      double al=rz/pAp;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++){ uv[i]+=al*pd[i]; r[i]-=al*Ap[i]; }
      double rn=0;
      #pragma omp parallel for reduction(+:rn)
      for (i32 i=0;i<nDofQ;i++) rn+=r[i]*r[i];
      rn=sqrt(rn);
      if (rn<=cgTol*bnorm){ it++; cgRes=rn/bnorm; break; }
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) z[i]=r[i]/diagv[i];
      double rz2=0;
      #pragma omp parallel for reduction(+:rz2)
      for (i32 i=0;i<nDofQ;i++) rz2+=r[i]*z[i];
      double be=rz2/rz; rz=rz2;
      #pragma omp parallel for
      for (i32 i=0;i<nDofQ;i++) pd[i]=z[i]+be*pd[i];
      cgRes=rn/bnorm;
    }
    cgIters=it;

    // ---- errors + geometry ----
    double l2e=0,l2n=0,ene=0,enn=0,vol=0,area=0;
    for (i32 e=0;e<nE;e++){ const i32*nod=&eNodeQ[(size_t)e*ndof];
      double uloc[3*QN_MAX*QN_MAX*QN_MAX];
      for (i32 a=0;a<ndof;a++){ double u3[3]; gather3(uv,nod[a],u3); uloc[3*a]=u3[0];uloc[3*a+1]=u3[1];uloc[3*a+2]=u3[2]; }
      std::vector<SayeNode> tens; const SayeNode*vn; i32 nv;
      if (!elems[e].cut){ tens.resize(ndof); i32 qi=0;
        for (i32 k=0;k<n;k++)for(i32 j=0;j<n;j++)for(i32 i=0;i<n;i++){ tens[qi].x[0]=Bp.t[i];tens[qi].x[1]=Bp.t[j];tens[qi].x[2]=Bp.t[k];
          tens[qi].w=Bp.wq[i]*Bp.wq[j]*Bp.wq[k]; qi++; } vn=tens.data(); nv=ndof;
      } else { i32 c=cutIdx[e]; nv=volOff[c+1]-volOff[c]; vn=&volPool[volOff[c]]; }
      real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
      for (i32 q=0;q<nv;q++){ real xr[3]={vn[q].x[0],vn[q].x[1],vn[q].x[2]};
        Bp.allVal(xr,vb); Bp.allGradRef(xr,gb);
        double gX[QN_MAX*QN_MAX*QN_MAX][3], dw;
        if (cyl){ double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
          for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=Jinv[0][d]*gb[3*a+0]+Jinv[1][d]*gb[3*a+1]+Jinv[2][d]*gb[3*a+2];
          dw=fabs(detJ)*vn[q].w; }
        else { for (i32 a=0;a<ndof;a++) for(i32 d=0;d<3;d++) gX[a][d]=gb[3*a+d]/h; dw=vn[q].w*h*h*h; }
        double uh[3]={0,0,0}, gh[3][3]={{0,0,0},{0,0,0},{0,0,0}};
        for (i32 a=0;a<ndof;a++) for(i32 l=0;l<3;l++){ uh[l]+=uloc[3*a+l]*vb[a];
          gh[l][0]+=uloc[3*a+l]*gX[a][0]; gh[l][1]+=uloc[3*a+l]*gX[a][1]; gh[l][2]+=uloc[3*a+l]*gX[a][2]; }
        real X[3]; physOf(elems[e],xr,X); real ue[3]; prob.exactU(X[0],X[1],X[2],ue);
        real ge[3][3]; prob.exactGradU(X[0],X[1],X[2],ge);
        vol+=dw; for (i32 l=0;l<3;l++){ double d=uh[l]-ue[l]; l2e+=d*d*dw; l2n+=ue[l]*ue[l]*dw; }
        double ee[3][3],se[3][3],tre=0;
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) ee[i2][j2]=0.5*((gh[i2][j2]-ge[i2][j2])+(gh[j2][i2]-ge[j2][i2]));
        tre=ee[0][0]+ee[1][1]+ee[2][2];
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) se[i2][j2]=2*mu*ee[i2][j2]+(i2==j2?lam*tre:0);
        double en=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) en+=se[i2][j2]*ee[i2][j2]; ene+=en*dw;
        double eeE[3][3],seE[3][3],trE=0;
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) eeE[i2][j2]=0.5*(ge[i2][j2]+ge[j2][i2]);
        trE=eeE[0][0]+eeE[1][1]+eeE[2][2];
        for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) seE[i2][j2]=2*mu*eeE[i2][j2]+(i2==j2?lam*trE:0);
        double enE=0; for(i32 i2=0;i2<3;i2++)for(i32 j2=0;j2<3;j2++) enE+=seE[i2][j2]*eeE[i2][j2]; enn+=enE*dw; }
      if (elems[e].cut){ i32 c=cutIdx[e]; i32 ns=surfOff[c+1]-surfOff[c]; const SayeNode*sn=&surfPool[surfOff[c]];
        for (i32 q=0;q<ns;q++){ if (!cyl){ area+=sn[q].w*h*h; }
          else { real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]}; double Jinv[3][3],detJ; metric(elems[e],xr,Jinv,detJ);
            double nref[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]}, nraw[3];
            for(i32 i2=0;i2<3;i2++) nraw[i2]=Jinv[0][i2]*nref[0]+Jinv[1][i2]*nref[1]+Jinv[2][i2]*nref[2];
            area+=fabs(detJ)*sqrt(nraw[0]*nraw[0]+nraw[1]*nraw[1]+nraw[2]*nraw[2])*sn[q].w; } } } }
    errL2=sqrt(l2e); normL2=sqrt(l2n); errEnergy=sqrt(ene); normEnergy=sqrt(enn);
    volOmega=vol; areaGamma=area;

    // ---- report ----
    double ms=(qpNowUs()-t0)/1000.0; i32 nCutTrueQ=0; for (i32 e=0;e<nE;e++) if (elems[e].cut) nCutTrueQ++;
    printf("active : %d Qp elements (%d cut), %d ghost faces\n", nE, nCutTrueQ, nGFQ);
    printf("dofs   : %d nodes -> %d unknowns   h = %.6g   p = %d\n", nNodeQ, nDofQ, (double)h, p);
    printf("geom   : |Omega_h| = %.8g", volOmega);
    if (volExact>0) printf("   exact %.8g   err %.3e (%.3f%%)", volExact, volOmega-volExact, 100.0*fabs(volOmega-volExact)/volExact);
    printf("\n         |Gamma_h| = %.8g", areaGamma);
    if (areaExact>0) printf("   exact %.8g   err %.3e (%.3f%%)", areaExact, areaGamma-areaExact, 100.0*fabs(areaGamma-areaExact)/areaExact);
    printf("\nsolve  : CG %d iters, rel res %.2e   (%.0f ms)\n", cgIters, cgRes, ms);
    if ((prob.caseId==CASE_MMS || prob.caseId==CASE_MMS_CYL) && normL2>0)
      printf("error  : L2 %.6e (rel %.4e)   energy %.6e (rel %.4e)\n", errL2, errL2/normL2, errEnergy, errEnergy/normEnergy);

    // ---- VTU (p^3 sub-hexes) ----
    if (wantVtu && !outTag.empty()){ mkdir("output",0755);
      std::string fn="output/"+outTag+"_femqp.vtu"; std::ofstream os(fn.c_str(),std::ios::binary);
      i64 nSub=(i64)nE*p*p*p; static const i32 HEX[8][3]={{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
      os<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
        <<"  <UnstructuredGrid>\n    <Piece NumberOfPoints=\""<<nNodeQ<<"\" NumberOfCells=\""<<nSub<<"\">\n";
      os<<"      <Points>\n        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
      for (i32 nd=0;nd<nNodeQ;nd++) os<<(float)nodeXQ[3*nd]<<" "<<(float)nodeXQ[3*nd+1]<<" "<<(float)nodeXQ[3*nd+2]<<"\n";
      os<<"        </DataArray>\n      </Points>\n";
      os<<"      <PointData Vectors=\"u\">\n        <DataArray type=\"Float32\" Name=\"u\" NumberOfComponents=\"3\" format=\"ascii\">\n";
      for (i32 nd=0;nd<nNodeQ;nd++){ double u3[3]; gather3(uv,nd,u3); os<<(float)u3[0]<<" "<<(float)u3[1]<<" "<<(float)u3[2]<<"\n"; }
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
}
