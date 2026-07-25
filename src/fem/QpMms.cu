//
// End-to-end MMS convergence test for the higher-order (Qp) CutFEM method, on a
// STANDALONE structured cut mesh -- the decisive gate before wiring into the
// production sparse-grid solver.  Reuses the three verified modules:
//   QpBasis.h  (basis)   QpElem.h/SayeQuad.h (bulk + cut quadrature)  PolyFit.h
//
// Domain: a sphere immersed in a uniform N^3 grid of Q_p cubes over [-1,1]^3.
// Whole boundary Gamma = {sphere} is Dirichlet, imposed weakly by Nitsche.
// Manufactured solution is divergence-free so f = -div sigma(u) = -mu*lap(u):
//    u = ( sin(k y) sin(k z),  sin(k z) sin(k x),  sin(k x) sin(k y) ),
//    div u = 0,  lap u_i = -2k^2 u_i  =>  f = 2 mu k^2 u.
// Pass: ||u-u_h||_L2 slope -> p+1.
//
//   build: nvcc -O2 -DUSE_DOUBLE -I src/common -I src/fem src/fem/QpMms.cu -o qp_mms
//   run:   ./qp_mms <p> <N1> <N2> ...
//

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "QpElem.h"

// CG vector precision: -DFP32CG stores the CG vectors in float (fp32) while
// keeping fp64 dot-product accumulators -- mirrors the production runQp fp32 path,
// so we can check the fp32 mixed-precision CG converges (host-run, no GPU needed).
#ifdef FP32CG
typedef float  cgv;
#else
typedef double cgv;
#endif
#include "PolyFit.h"

static double MU = 0.8, LAM = 1.2;
static double KK = M_PI;
static double SPH_R = 0.75, SPH_C[3] = {0.0123, -0.0071, 0.0055};

static double sdf(double x,double y,double z){
  double dx=x-SPH_C[0],dy=y-SPH_C[1],dz=z-SPH_C[2];
  return sqrt(dx*dx+dy*dy+dz*dz)-SPH_R;
}
static void uex(double x,double y,double z,double u[3]){
  u[0]=sin(KK*y)*sin(KK*z); u[1]=sin(KK*z)*sin(KK*x); u[2]=sin(KK*x)*sin(KK*y);
}
static void fbody(double x,double y,double z,double f[3]){
  double u[3]; uex(x,y,z,u);
  for(int i=0;i<3;i++) f[i]=2*MU*KK*KK*u[i];   // -mu*lap(u), lap u_i=-2k^2 u_i
}

// ---- global CG state (host, double) ---------------------------------------
struct Mesh {
  int p, n, N;
  double h, dom0;                       // cell size, domain origin
  int nAx;                              // nodes per axis = p*N+1
  std::vector<int> nodeDof;             // per lattice node -> dof (-1 if unused)
  int nDof;                             // active nodes
  // active elements
  std::vector<int>    eNodes;           // [elem*ndof + a] global dof (node id)
  std::vector<char>   eCut;             // 0 interior, 1 cut
  std::vector<PolyND> ePhi;             // per elem level-set fit
  // stored Saye rules for cut elements (flat, CSR by cut-element order)
  std::vector<int> eVolOff, eSurfOff;   // per elem, offsets into vol/surf pools
  std::vector<SayeNode> volPool, surfPool;
  int ndof;                             // (p+1)^3
};

int main(int argc, char** argv){
  int p = argc>1 ? atoi(argv[1]) : 2;
  std::vector<int> Ns;
  for (int a=2; a<argc; a++) Ns.push_back(atoi(argv[a]));
  if (Ns.empty()) Ns = {8,16,32};

  QpBasis B; B.init(p);
  int n=B.n, ndof=n*n*n;
  double gammaD = 100.0*(2*MU+LAM)*p*p;   // Nitsche penalty
  double gammaG = 0.1*(2*MU+LAM);         // ghost-penalty coefficient gamma_l

  // l-th normal-derivative of each 1-D basis at the two faces (xi=0, xi=1):
  //   Dl0[l][a] = (D^l)[0][a],  Dl1[l][a] = (D^l)[n-1][a],  l=1..p.
  // D^l is the repeated differentiation matrix (l-th derivative at the nodes).
  static double Dl0[QN_MAX+1][QN_MAX], Dl1[QN_MAX+1][QN_MAX];
  {
    double Dp[QN_MAX][QN_MAX];
    for(int i=0;i<n;i++) for(int a=0;a<n;a++) Dp[i][a]=B.D[i][a];   // D^1
    for(int l=1;l<=p;l++){
      for(int a=0;a<n;a++){ Dl0[l][a]=Dp[0][a]; Dl1[l][a]=Dp[n-1][a]; }
      if(l<p){
        double Nw[QN_MAX][QN_MAX];
        for(int i=0;i<n;i++) for(int a=0;a<n;a++){
          double s=0; for(int m=0;m<n;m++) s+=Dp[i][m]*B.D[m][a]; Nw[i][a]=s;
        }
        for(int i=0;i<n;i++) for(int a=0;a<n;a++) Dp[i][a]=Nw[i][a];
      }
    }
  }

  printf("Qp CutFEM MMS  p=%d  mu=%.2f lam=%.2f  gammaD=%.1f\n", p, MU, LAM, gammaD);
  printf("  %4s  %10s  %12s  %6s  %8s  %6s\n","N","nDof","L2err","ord","cgIt","nCut");

  static SayeNode arenaBuf[1<<18], outBuf[1<<16];
  double prevE=0, prevH=0;

  for (int N : Ns){
    Mesh M; M.p=p; M.n=n; M.N=N; M.ndof=ndof;
    double dom0=-1.0, domL=2.0; M.h=domL/N; M.dom0=dom0;
    int nAx=p*N+1; M.nAx=nAx;
    M.nodeDof.assign((size_t)nAx*nAx*nAx, -1);

    auto gnode=[&](int cx,int cy,int cz,int i,int j,int k)->long{
      long I=p*cx+i, J=p*cy+j, K=p*cz+k;
      return I + (long)nAx*(J + (long)nAx*K);
    };

    // ---- pass 1: classify elements, mark used nodes ----
    std::vector<int> actEl;             // active element linear ids
    for (int cz=0;cz<N;cz++) for(int cy=0;cy<N;cy++) for(int cx=0;cx<N;cx++){
      double x0=dom0+cx*M.h, y0=dom0+cy*M.h, z0=dom0+cz*M.h;
      real v[PNC*PNC*PNC]; real t[PNC]; gllNodes(p,t);
      bool anyNeg=false, anyPos=false;
      for(int k=0;k<n;k++) for(int j=0;j<n;j++) for(int i=0;i<n;i++){
        double f=sdf(x0+t[i]*M.h, y0+t[j]*M.h, z0+t[k]*M.h);
        v[i+n*(j+n*k)]=(real)f; if(f<0)anyNeg=true; else anyPos=true;
      }
      if(!anyNeg) continue;                       // element entirely outside
      int eid = cx + N*(cy + N*cz);
      actEl.push_back(eid);
      for(int k=0;k<n;k++) for(int j=0;j<n;j++) for(int i=0;i<n;i++)
        M.nodeDof[gnode(cx,cy,cz,i,j,k)] = 0;     // mark used
    }
    // compact dof ids
    M.nDof=0;
    for (size_t q=0;q<M.nodeDof.size();q++) if(M.nodeDof[q]==0) M.nodeDof[q]=M.nDof++;

    // ---- pass 2: per active element, node maps + Saye rules ----
    int nE=actEl.size();
    M.eNodes.resize((size_t)nE*ndof);
    M.eCut.resize(nE); M.ePhi.resize(nE);
    M.eVolOff.assign(nE+1,0); M.eSurfOff.assign(nE+1,0);
    int nCut=0;
    for (int e=0;e<nE;e++){
      int eid=actEl[e]; int cx=eid%N, cy=(eid/N)%N, cz=eid/(N*N);
      double x0=dom0+cx*M.h, y0=dom0+cy*M.h, z0=dom0+cz*M.h;
      real v[PNC*PNC*PNC]; real t[PNC]; gllNodes(p,t);
      bool anyPos=false;
      for(int k=0;k<n;k++) for(int j=0;j<n;j++) for(int i=0;i<n;i++){
        double f=sdf(x0+t[i]*M.h, y0+t[j]*M.h, z0+t[k]*M.h);
        v[i+n*(j+n*k)]=(real)f; if(f>=0)anyPos=true;
        M.eNodes[(size_t)e*ndof + (i+n*(j+n*k))] = M.nodeDof[gnode(cx,cy,cz,i,j,k)];
      }
      M.ePhi[e]=fitPoly3(p,v);
      M.eCut[e]=anyPos?1:0;
      if(anyPos){
        nCut++;
        // volume rule
        SayeArena ar; ar.buf=arenaBuf; ar.cap=1<<18; ar.top=0;
        SayeSet out; out.p=outBuf; out.n=0; out.cap=1<<16; out.ovf=false;
        sayeVolume(M.ePhi[e], &out, &ar, SayeCfg::def());
        M.eVolOff[e+1]=M.eVolOff[e]; // filled below via push
        for(int q=0;q<out.n;q++) M.volPool.push_back(out.p[q]);
        // surface rule
        SayeArena ar2; ar2.buf=arenaBuf; ar2.cap=1<<18; ar2.top=0;
        SayeSet sout; sout.p=outBuf; sout.n=0; sout.cap=1<<16; sout.ovf=false;
        sayeSurface(M.ePhi[e], &sout, &ar2, SayeCfg::def());
        for(int q=0;q<sout.n;q++) M.surfPool.push_back(sout.p[q]);
      }
      M.eVolOff[e+1]=M.volPool.size();
      M.eSurfOff[e+1]=M.surfPool.size();
    }

    // ---- ghost-penalty face list: interior faces of cut elements ----
    // eid -> active index, to find same-level neighbours
    std::vector<int> elemIdx((size_t)N*N*N, -1);
    for(int e=0;e<nE;e++) elemIdx[actEl[e]]=e;
    struct GFace{ int eM, eP, d; };     // minus/plus element (along +d), axis d
    std::vector<GFace> gfaces;
    for(int e=0;e<nE;e++){
      int eid=actEl[e]; int cx=eid%N, cy=(eid/N)%N, cz=eid/(N*N);
      int cc[3]={cx,cy,cz};
      for(int d=0;d<3;d++){
        if(cc[d]+1>=N) continue;
        int nb[3]={cx,cy,cz}; nb[d]++;
        int eid2=nb[0]+N*(nb[1]+N*nb[2]);
        int ep=elemIdx[eid2];
        if(ep<0) continue;
        if(M.eCut[e]||M.eCut[ep]) gfaces.push_back({e,ep,d});
      }
    }
    int nGF=gfaces.size();

    int ndof3=3*ndof;
    long nd3=(long)3*M.nDof;

    // ---- Jacobi diagonal ----
    std::vector<cgv> diag(nd3, (cgv)0);
    // ---- RHS ----
    std::vector<cgv> b(nd3, (cgv)0);

    // ---- ghost-penalty face contribution (shared by diag and applyA) ----
    //   j_h = sum_l gammaG * h * int_F [d_n^l u][d_n^l v] dS_ref  (h factor after
    //   the 1/h^{2l} and h^2 from dS collapse).  Dg!=0 -> diagonal, else apply.
    auto ghostFace=[&](const GFace& gf, const std::vector<cgv>* X,
                       std::vector<cgv>* Y, std::vector<cgv>* Dg){
      int d=gf.d, t1=(d+1)%3, t2=(d+2)%3;
      const int* nodM=&M.eNodes[(size_t)gf.eM*ndof];
      const int* nodP=&M.eNodes[(size_t)gf.eP*ndof];
      GaussRule g1=gaussLegendre(p+1);
      double cP[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX], cM[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX];
      for(int q1=0;q1<g1.n;q1++) for(int q2=0;q2<g1.n;q2++){
        double w=g1.w[q1]*g1.w[q2];
        real L1[QN_MAX],L2[QN_MAX]; B.basis1(g1.x[q1],L1); B.basis1(g1.x[q2],L2);
        for(int a=0;a<ndof;a++){
          int idx[3]={a%n,(a/n)%n,a/(n*n)};
          int idn=idx[d];
          double Ltang=L1[idx[t1]]*L2[idx[t2]];
          for(int l=1;l<=p;l++){ cP[l][a]=Dl0[l][idn]*Ltang; cM[l][a]=Dl1[l][idn]*Ltang; }
        }
        if(Dg){
          for(int l=1;l<=p;l++){
            double cf=gammaG*M.h*w;
            for(int a=0;a<ndof;a++) for(int comp=0;comp<3;comp++){
              (*Dg)[3*nodP[a]+comp]+= cf*cP[l][a]*cP[l][a];
              (*Dg)[3*nodM[a]+comp]+= cf*cM[l][a]*cM[l][a];
            }
          }
        } else {
          for(int l=1;l<=p;l++){
            double cf=gammaG*M.h*w;
            double jU[3]={0,0,0};
            for(int a=0;a<ndof;a++) for(int comp=0;comp<3;comp++)
              jU[comp]+= (*X)[3*nodP[a]+comp]*cP[l][a] - (*X)[3*nodM[a]+comp]*cM[l][a];
            for(int a=0;a<ndof;a++) for(int comp=0;comp<3;comp++){
              #pragma omp atomic
              (*Y)[3*nodP[a]+comp]+= cf*cP[l][a]*jU[comp];
              #pragma omp atomic
              (*Y)[3*nodM[a]+comp]+= cf*(-cM[l][a])*jU[comp];
            }
          }
        }
      }
    };

    real gbq[3*QN_MAX*QN_MAX*QN_MAX], vbq[QN_MAX*QN_MAX*QN_MAX];
    for (int e=0;e<nE;e++){
      int eid=actEl[e]; int cx=eid%N, cy=(eid/N)%N, cz=eid/(N*N);
      double x0=dom0+cx*M.h, y0=dom0+cy*M.h, z0=dom0+cz*M.h;
      const int* nod=&M.eNodes[(size_t)e*ndof];

      // --- volume quadrature source: uncut=tensor GLL, cut=stored Saye ---
      int nv; const SayeNode* vn=nullptr;
      std::vector<SayeNode> tens;
      if(!M.eCut[e]){
        tens.resize(ndof); int q=0;
        for(int k=0;k<n;k++)for(int j=0;j<n;j++)for(int i=0;i<n;i++){
          tens[q].x[0]=B.t[i];tens[q].x[1]=B.t[j];tens[q].x[2]=B.t[k];
          tens[q].w=B.wq[i]*B.wq[j]*B.wq[k]; q++;
        }
        nv=ndof; vn=tens.data();
      } else { nv=M.eVolOff[e+1]-M.eVolOff[e]; vn=&M.volPool[M.eVolOff[e]]; }

      for(int q=0;q<nv;q++){
        real xr[3]={vn[q].x[0],vn[q].x[1],vn[q].x[2]};
        B.allGradRef(xr,gbq); B.allVal(xr,vbq);
        double xp=x0+xr[0]*M.h, yp=y0+xr[1]*M.h, zp=z0+xr[2]*M.h;
        double f[3]; fbody(xp,yp,zp,f);
        double wv=vn[q].w*M.h*M.h*M.h;                       // dx
        for(int a=0;a<ndof;a++){
          for(int l=0;l<3;l++) b[3*nod[a]+l]+=wv*f[l]*vbq[a];
          // bulk diagonal
          double gg=gbq[3*a+0]*gbq[3*a+0]+gbq[3*a+1]*gbq[3*a+1]+gbq[3*a+2]*gbq[3*a+2];
          double wb=vn[q].w*M.h;
          for(int l=0;l<3;l++)
            diag[3*nod[a]+l]+= wb*(MU*(gg+gbq[3*a+l]*gbq[3*a+l])+LAM*gbq[3*a+l]*gbq[3*a+l]);
        }
      }

      // --- Nitsche surface source (cut only) ---
      if(M.eCut[e]){
        int ns=M.eSurfOff[e+1]-M.eSurfOff[e];
        const SayeNode* sn=&M.surfPool[M.eSurfOff[e]];
        for(int q=0;q<ns;q++){
          real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]};
          double nn[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]};
          B.allGradRef(xr,gbq); B.allVal(xr,vbq);
          double xp=x0+xr[0]*M.h, yp=y0+xr[1]*M.h, zp=z0+xr[2]*M.h;
          double g[3]; uex(xp,yp,zp,g);
          double hw=sn[q].w*M.h;
          double gn=g[0]*nn[0]+g[1]*nn[1]+g[2]*nn[2];
          for(int a=0;a<ndof;a++){
            double gan=gbq[3*a+0]*nn[0]+gbq[3*a+1]*nn[1]+gbq[3*a+2]*nn[2];
            double ggb=g[0]*gbq[3*a+0]+g[1]*gbq[3*a+1]+g[2]*gbq[3*a+2];
            for(int l=0;l<3;l++){
              // RHS: -g.(sig(phi_a e_l) n) + gammaD g.(phi_a e_l)
              double rhs = -(MU*(g[l]*gan + ggb*nn[l]) + LAM*gbq[3*a+l]*gn)
                           + gammaD*g[l]*vbq[a];
              b[3*nod[a]+l]+= hw*rhs;
              // penalty diagonal
              diag[3*nod[a]+l]+= hw*gammaD*vbq[a]*vbq[a];
            }
          }
        }
      }
    }
    // ghost-penalty diagonal
    for(int gf=0; gf<nGF; gf++) ghostFace(gfaces[gf], nullptr, nullptr, &diag);
    for(long i=0;i<nd3;i++) if(diag[i]<=0) diag[i]=1.0;

    // ---- precompute dense local element matrices (production design) ----
    // Kuncut is shared by every interior element; each cut element stores its own
    // (bulk Saye + Nitsche); one ghost-face matrix per axis (reference-invariant).
    // CG then reduces to dense local matvecs -- feasible at p=3.
    int mE=ndof3, mG=2*ndof3;

    // local apply for a cut element (bulk + Nitsche) on local dof arrays
    auto applyCutLocal=[&](int e, const double* uloc, double* yloc){
      int nv=M.eVolOff[e+1]-M.eVolOff[e]; const SayeNode* vn=&M.volPool[M.eVolOff[e]];
      real ul[3*QN_MAX*QN_MAX*QN_MAX], yl[3*QN_MAX*QN_MAX*QN_MAX];
      for(int a=0;a<ndof3;a++) ul[a]=(real)uloc[a];
      qpElemCoreSaye(B,MU,LAM,M.h,vn,nv,ul,yl);
      for(int a=0;a<ndof3;a++) yloc[a]=yl[a];
      int ns=M.eSurfOff[e+1]-M.eSurfOff[e]; const SayeNode* sn=&M.surfPool[M.eSurfOff[e]];
      real gb[3*QN_MAX*QN_MAX*QN_MAX], vb[QN_MAX*QN_MAX*QN_MAX];
      for(int q=0;q<ns;q++){
        real xr[3]={sn[q].x[0],sn[q].x[1],sn[q].x[2]};
        double nn[3]={sn[q].n[0],sn[q].n[1],sn[q].n[2]};
        B.allGradRef(xr,gb); B.allVal(xr,vb);
        double hw=sn[q].w*M.h;
        double uval[3]={0,0,0}, gradU[3][3]={{0,0,0},{0,0,0},{0,0,0}};
        for(int a=0;a<ndof;a++) for(int i2=0;i2<3;i2++){
          uval[i2]+=uloc[3*a+i2]*vb[a];
          gradU[i2][0]+=uloc[3*a+i2]*gb[3*a+0];
          gradU[i2][1]+=uloc[3*a+i2]*gb[3*a+1];
          gradU[i2][2]+=uloc[3*a+i2]*gb[3*a+2];
        }
        double eps[3][3],tr=0;
        for(int i2=0;i2<3;i2++)for(int j2=0;j2<3;j2++) eps[i2][j2]=0.5*(gradU[i2][j2]+gradU[j2][i2]);
        tr=eps[0][0]+eps[1][1]+eps[2][2];
        double sig[3][3];
        for(int i2=0;i2<3;i2++)for(int j2=0;j2<3;j2++) sig[i2][j2]=2*MU*eps[i2][j2]+(i2==j2?LAM*tr:0);
        double tu[3]; for(int i2=0;i2<3;i2++) tu[i2]=sig[i2][0]*nn[0]+sig[i2][1]*nn[1]+sig[i2][2]*nn[2];
        double un=uval[0]*nn[0]+uval[1]*nn[1]+uval[2]*nn[2];
        for(int a=0;a<ndof;a++){
          double gan=gb[3*a+0]*nn[0]+gb[3*a+1]*nn[1]+gb[3*a+2]*nn[2];
          double ugb=uval[0]*gb[3*a+0]+uval[1]*gb[3*a+1]+uval[2]*gb[3*a+2];
          for(int l=0;l<3;l++){
            double t1=-tu[l]*vb[a];
            double t2=-(MU*(uval[l]*gan+ugb*nn[l])+LAM*gb[3*a+l]*un);
            double t3= gammaD*uval[l]*vb[a];
            yloc[3*a+l]+= hw*(t1+t2+t3);
          }
        }
      }
    };
    // Kuncut
    std::vector<double> Kuncut((size_t)mE*mE,0.0);
    {
      real ul[3*QN_MAX*QN_MAX*QN_MAX], yl[3*QN_MAX*QN_MAX*QN_MAX];
      for(int c=0;c<mE;c++){
        for(int a=0;a<mE;a++) ul[a]=(a==c)?1:0;
        qpElemUncut(B,MU,LAM,M.h,ul,yl);
        for(int r=0;r<mE;r++) Kuncut[(size_t)r*mE+c]=yl[r];
      }
    }
    // per-cut-element matrices
    std::vector<int> cutIdx(nE,-1); int nCutE=0;
    for(int e=0;e<nE;e++) if(M.eCut[e]) cutIdx[e]=nCutE++;
    std::vector<double> cutMat((size_t)nCutE*mE*mE,0.0);
    #pragma omp parallel for schedule(dynamic,4)
    for(int e=0;e<nE;e++) if(M.eCut[e]){
      double* K=&cutMat[(size_t)cutIdx[e]*mE*mE];
      double ue[3*QN_MAX*QN_MAX*QN_MAX], ye[3*QN_MAX*QN_MAX*QN_MAX];
      for(int c=0;c<mE;c++){
        for(int a=0;a<mE;a++) ue[a]=(a==c)?1.0:0.0;
        applyCutLocal(e, ue, ye);
        for(int r=0;r<mE;r++) K[(size_t)r*mE+c]=ye[r];
      }
    }
    // ghost-face matrix per axis (reference-invariant)
    auto ghostLocal=[&](int d, const double* uMP, double* yMP){
      for(int i=0;i<mG;i++) yMP[i]=0.0;
      int t1=(d+1)%3,t2=(d+2)%3; GaussRule g1=gaussLegendre(p+1);
      for(int q1=0;q1<g1.n;q1++) for(int q2=0;q2<g1.n;q2++){
        double w=g1.w[q1]*g1.w[q2];
        real L1[QN_MAX],L2[QN_MAX]; B.basis1(g1.x[q1],L1); B.basis1(g1.x[q2],L2);
        double cP[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX], cM[QP_MAX+1][QN_MAX*QN_MAX*QN_MAX];
        for(int a=0;a<ndof;a++){ int idx[3]={a%n,(a/n)%n,a/(n*n)}; int idn=idx[d];
          double Lt=L1[idx[t1]]*L2[idx[t2]];
          for(int l=1;l<=p;l++){ cP[l][a]=Dl0[l][idn]*Lt; cM[l][a]=Dl1[l][idn]*Lt; } }
        for(int l=1;l<=p;l++){ double cf=gammaG*M.h*w; double jU[3]={0,0,0};
          for(int a=0;a<ndof;a++) for(int comp=0;comp<3;comp++)
            jU[comp]+= uMP[ndof3+3*a+comp]*cP[l][a] - uMP[3*a+comp]*cM[l][a];
          for(int a=0;a<ndof;a++) for(int comp=0;comp<3;comp++){
            yMP[ndof3+3*a+comp]+= cf*cP[l][a]*jU[comp];
            yMP[3*a+comp]+= cf*(-cM[l][a])*jU[comp];
          }
        }
      }
    };
    std::vector<double> Kghost[3];
    for(int d=0;d<3;d++){ Kghost[d].assign((size_t)mG*mG,0.0);
      double ue[2*3*QN_MAX*QN_MAX*QN_MAX], ye[2*3*QN_MAX*QN_MAX*QN_MAX];
      for(int c=0;c<mG;c++){ for(int a=0;a<mG;a++) ue[a]=(a==c)?1.0:0.0;
        ghostLocal(d,ue,ye);
        for(int r=0;r<mG;r++) Kghost[d][(size_t)r*mG+c]=ye[r];
      }
    }

    // ---- operator apply: dense local matvecs ----
    auto applyA=[&](const std::vector<cgv>& x, std::vector<cgv>& y){
      std::fill(y.begin(),y.end(),0.0);
      #pragma omp parallel for schedule(dynamic,64)
      for(int e=0;e<nE;e++){
        const int* nod=&M.eNodes[(size_t)e*ndof];
        double uloc[3*QN_MAX*QN_MAX*QN_MAX], yloc[3*QN_MAX*QN_MAX*QN_MAX];
        for(int a=0;a<ndof;a++) for(int l=0;l<3;l++) uloc[3*a+l]=x[3*nod[a]+l];
        const double* K = M.eCut[e]? &cutMat[(size_t)cutIdx[e]*mE*mE] : Kuncut.data();
        for(int r=0;r<mE;r++){ double s=0; const double* Kr=&K[(size_t)r*mE];
          for(int c=0;c<mE;c++) s+=Kr[c]*uloc[c]; yloc[r]=s; }
        for(int a=0;a<ndof;a++) for(int l=0;l<3;l++){
          #pragma omp atomic
          y[3*nod[a]+l]+=yloc[3*a+l];
        }
      }
      #pragma omp parallel for schedule(dynamic,64)
      for(int gf=0; gf<nGF; gf++){
        const GFace& F=gfaces[gf];
        const int* nodM=&M.eNodes[(size_t)F.eM*ndof];
        const int* nodP=&M.eNodes[(size_t)F.eP*ndof];
        double uMP[2*3*QN_MAX*QN_MAX*QN_MAX], yMP[2*3*QN_MAX*QN_MAX*QN_MAX];
        for(int a=0;a<ndof;a++) for(int l=0;l<3;l++){ uMP[3*a+l]=x[3*nodM[a]+l]; uMP[ndof3+3*a+l]=x[3*nodP[a]+l]; }
        const double* K=Kghost[F.d].data();
        for(int r=0;r<mG;r++){ double s=0; const double* Kr=&K[(size_t)r*mG];
          for(int c=0;c<mG;c++) s+=Kr[c]*uMP[c]; yMP[r]=s; }
        for(int a=0;a<ndof;a++) for(int l=0;l<3;l++){
          #pragma omp atomic
          y[3*nodM[a]+l]+=yMP[3*a+l];
          #pragma omp atomic
          y[3*nodP[a]+l]+=yMP[ndof3+3*a+l];
        }
      }
    };

    fprintf(stderr,"    [N=%d setup done: nE=%d nCut=%d nGF=%d volPts=%zu surfPts=%zu nDof=%d]\n",
            N,nE,nCut,nGF,M.volPool.size(),M.surfPool.size(),M.nDof);
    // ---- Jacobi-preconditioned CG ----
    std::vector<cgv> u(nd3,(cgv)0), r=b, z(nd3), pd(nd3), Ap(nd3);
    #pragma omp parallel for
    for(long i=0;i<nd3;i++) z[i]=r[i]/diag[i];
    pd=z;
    double rz=0;    for(long i=0;i<nd3;i++) rz+=r[i]*z[i];
    double bnorm=0; for(long i=0;i<nd3;i++) bnorm+=b[i]*b[i]; bnorm=sqrt(bnorm);
    const char* mi=getenv("CG_MAXIT"); int maxit = mi?atoi(mi):20000;
    const char* ct=getenv("CG_TOL");   double tol = ct?atof(ct):1e-10;
    int it=0;
    for(; it<maxit; it++){
      applyA(pd,Ap);
      double pAp=0;
      #pragma omp parallel for reduction(+:pAp)
      for(long i=0;i<nd3;i++) pAp+=pd[i]*Ap[i];
      double al=rz/pAp;
      #pragma omp parallel for
      for(long i=0;i<nd3;i++){ u[i]+=al*pd[i]; r[i]-=al*Ap[i]; }
      double rn=0;
      #pragma omp parallel for reduction(+:rn)
      for(long i=0;i<nd3;i++) rn+=r[i]*r[i];
      rn=sqrt(rn);
      if(it%500==0) fprintf(stderr,"    [N=%d it=%d relres=%.3e]\n",N,it,rn/bnorm);
      if(rn<=tol*bnorm) { it++; break; }
      #pragma omp parallel for
      for(long i=0;i<nd3;i++) z[i]=r[i]/diag[i];
      double rz2=0;
      #pragma omp parallel for reduction(+:rz2)
      for(long i=0;i<nd3;i++) rz2+=r[i]*z[i];
      double be=rz2/rz; rz=rz2;
      #pragma omp parallel for
      for(long i=0;i<nd3;i++) pd[i]=z[i]+be*pd[i];
    }

    // ---- L2 error over Omega ----
    double l2=0;
    std::vector<SayeNode> tens;
    for(int e=0;e<nE;e++){
      int eid=actEl[e]; int cx=eid%N, cy=(eid/N)%N, cz=eid/(N*N);
      double x0=dom0+cx*M.h, y0=dom0+cy*M.h, z0=dom0+cz*M.h;
      const int* nod=&M.eNodes[(size_t)e*ndof];
      int nv; const SayeNode* vn=nullptr;
      if(!M.eCut[e]){
        tens.resize(ndof); int q=0;
        for(int k=0;k<n;k++)for(int j=0;j<n;j++)for(int i=0;i<n;i++){
          tens[q].x[0]=B.t[i];tens[q].x[1]=B.t[j];tens[q].x[2]=B.t[k];
          tens[q].w=B.wq[i]*B.wq[j]*B.wq[k]; q++;
        }
        nv=ndof; vn=tens.data();
      } else { nv=M.eVolOff[e+1]-M.eVolOff[e]; vn=&M.volPool[M.eVolOff[e]]; }
      real vb[QN_MAX*QN_MAX*QN_MAX];
      for(int q=0;q<nv;q++){
        real xr[3]={vn[q].x[0],vn[q].x[1],vn[q].x[2]};
        B.allVal(xr,vb);
        double uh[3]={0,0,0};
        for(int a=0;a<ndof;a++) for(int l=0;l<3;l++) uh[l]+=u[3*nod[a]+l]*vb[a];
        double xp=x0+xr[0]*M.h, yp=y0+xr[1]*M.h, zp=z0+xr[2]*M.h;
        double ue[3]; uex(xp,yp,zp,ue);
        double d2=0; for(int l=0;l<3;l++){ double d=uh[l]-ue[l]; d2+=d*d; }
        l2 += d2*vn[q].w*M.h*M.h*M.h;
      }
    }
    l2=sqrt(l2);
    double ord = (prevE>0)? log(prevE/l2)/log(prevH/M.h) : 0.0;
    printf("  %4d  %10ld  %12.4e  %6.2f  %8d  %6d\n", N, nd3, l2, ord, it, nCut);
    fflush(stdout);
    prevE=l2; prevH=M.h;
  }
  return 0;
}
