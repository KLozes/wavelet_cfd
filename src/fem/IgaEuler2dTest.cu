//
// IGA COMPRESSIBLE FLOW, step 2: 2-D Euler on a uniform tensor grid with a
// CUT-CELL immersed cylinder.
//
//   basis     tensor-product uniform C^{p-1} B-splines, control-point
//             conservative dofs, CONSISTENT mass (ELL-assembled, solved by
//             CG with the full-grid Kronecker mass as preconditioner).
//   geometry  circle SDF, EXACT analytic cut quadrature: per cut cell a
//             height-function rule with tangency/topology splits (machine-
//             accurate), wall rule on the exact arc with exact normals.
//   scheme    Galerkin flux term + Guermond-Popov ENTROPY VISCOSITY (the
//             1-D A/B winner: shock-only sensor, zero at contacts), cell-
//             wise with 3x3 neighbour-max smoothing.
//   walls     mirror-state Rusanov flux on the arc (zero mass/energy flux
//             by construction); far field: Rusanov vs free stream on the
//             box edges -- NO frozen rows (the 1-D drift lesson).
//   time      SSP-RK3.
//   gates     vortex : isentropic vortex on the UNCUT grid, L2(rho) orders
//             fsp    : free-stream preservation through the CUT mesh
//                      (transparent wall flux isolates quadrature errors)
//             cyl    : M=0.3 cylinder, real wall -- Cp vs potential theory,
//                      wall-normal velocity, conservation, EV quiescence
//
// build:  make iga_euler2d       run:  ./iga_euler2d [vortex|fsp|cyl|all] [p]
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "Util.cuh"
#include "IgaBasis.h"
#include "SayeQuad.h"   // GaussRule/gaussLegendre

static constexpr double GAM = 1.4;
static constexpr i32 NF = 4;               // rho, rho u, rho v, E

// ---------------------------------------------------------------------------
//  Euler helpers
// ---------------------------------------------------------------------------
__host__ __device__ static inline void primEval(const double U[NF], double &r, double &u, double &v,
                            double &pr, double &c) {
  r = fmax(U[0], 1e-12); u = U[1]/r; v = U[2]/r;
  pr = (GAM-1.0)*(U[3] - 0.5*r*(u*u+v*v)); pr = fmax(pr, 1e-12);
  c = sqrt(GAM*pr/r);
}
__host__ __device__ static inline void eulerFlux2(const double U[NF], double Fx[NF], double Fy[NF],
                              double &u, double &v, double &c) {
  double r, pr; primEval(U, r, u, v, pr, c);
  Fx[0]=U[1]; Fx[1]=U[1]*u+pr; Fx[2]=U[1]*v;    Fx[3]=(U[3]+pr)*u;
  Fy[0]=U[2]; Fy[1]=U[2]*u;    Fy[2]=U[2]*v+pr; Fy[3]=(U[3]+pr)*v;
}
static inline double entEta2(const double U[NF]) {
  double r,u,v,pr,c; primEval(U,r,u,v,pr,c);
  return -r*(log(pr)-GAM*log(r))/(GAM-1.0);
}
static inline double entQdir(const double U[NF], double dx, double dy) {
  double r,u,v,pr,c; primEval(U,r,u,v,pr,c);
  return (u*dx+v*dy)*entEta2(U);
}
// analytic Euler flux Jacobians A_x = dFx/dU, A_y = dFy/dU
__host__ __device__ static inline void eulerJac2(const double U[NF],
                                double Ax[NF][NF], double Ay[NF][NF]) {
  double r,u,v,pr,c; primEval(U,r,u,v,pr,c);
  double q2=u*u+v*v, H=(U[3]+pr)/r, ph=0.5*(GAM-1.0)*q2, g1=GAM-1.0;
  Ax[0][0]=0;        Ax[0][1]=1;           Ax[0][2]=0;        Ax[0][3]=0;
  Ax[1][0]=ph-u*u;   Ax[1][1]=(3.0-GAM)*u; Ax[1][2]=-g1*v;    Ax[1][3]=g1;
  Ax[2][0]=-u*v;     Ax[2][1]=v;           Ax[2][2]=u;        Ax[2][3]=0;
  Ax[3][0]=u*(ph-H); Ax[3][1]=H-g1*u*u;    Ax[3][2]=-g1*u*v;  Ax[3][3]=GAM*u;
  Ay[0][0]=0;        Ay[0][1]=0;           Ay[0][2]=1;        Ay[0][3]=0;
  Ay[1][0]=-u*v;     Ay[1][1]=v;           Ay[1][2]=u;        Ay[1][3]=0;
  Ay[2][0]=ph-v*v;   Ay[2][1]=-g1*u;       Ay[2][2]=(3.0-GAM)*v; Ay[2][3]=g1;
  Ay[3][0]=v*(ph-H); Ay[3][1]=-g1*u*v;     Ay[3][2]=H-g1*v*v; Ay[3][3]=GAM*v;
}

// Rusanov flux through unit normal n, exterior state Ue
__host__ __device__ static inline void rusanov(const double Ub[NF], const double Ue[NF],
                           double nx, double ny, double Fh[NF]) {
  double Fxb[NF],Fyb[NF],Fxe[NF],Fye[NF],ub,vb,cb,ue,ve,ce;
  eulerFlux2(Ub,Fxb,Fyb,ub,vb,cb); eulerFlux2(Ue,Fxe,Fye,ue,ve,ce);
  double lb=fabs(ub*nx+vb*ny)+cb, le=fabs(ue*nx+ve*ny)+ce, lam=fmax(lb,le);
  for (i32 k=0;k<NF;k++)
    Fh[k] = 0.5*((Fxb[k]+Fxe[k])*nx + (Fyb[k]+Fye[k])*ny) - 0.5*lam*(Ue[k]-Ub[k]);
}

// ---------------------------------------------------------------------------
//  1-D spline machinery (identical role to the 1-D test)
// ---------------------------------------------------------------------------
struct Spline1d {
  i32 p, N, n;
  IgaBasis B;
  void init(i32 p_, i32 N_) { p=p_; N=N_; n=N+p; B.init(p_); }
  void val(double xi, real *Nv) const { B.val((real)xi, Nv); }
  void der(double xi, real *Dv) const { B.der((real)xi, Dv); }
};

struct BandChol {
  i32 n, b;
  std::vector<double> L;
  void factor(const std::vector<double> &Ab) {
    L = Ab;
    for (i32 j=0;j<n;j++) {
      i32 j0 = (j-b>0)?(j-b):0;
      double d = L[(size_t)j*(b+1)+b];
      for (i32 k=j0;k<j;k++){ double l=L[(size_t)j*(b+1)+(k-j+b)]; d-=l*l; }
      d = sqrt(d); L[(size_t)j*(b+1)+b]=d;
      for (i32 i=j+1;i<=j+b && i<n;i++){
        i32 i0=(i-b>0)?(i-b):0;
        double s2=L[(size_t)i*(b+1)+(j-i+b)];
        for (i32 k=(i0>j0?i0:j0);k<j;k++)
          s2-=L[(size_t)i*(b+1)+(k-i+b)]*L[(size_t)j*(b+1)+(k-j+b)];
        L[(size_t)i*(b+1)+(j-i+b)]=s2/d;
      }
    }
  }
  void solve(double *x, i32 stride=1) const {
    for (i32 i=0;i<n;i++){ i32 i0=(i-b>0)?(i-b):0; double s2=x[(size_t)i*stride];
      for (i32 k=i0;k<i;k++) s2-=L[(size_t)i*(b+1)+(k-i+b)]*x[(size_t)k*stride];
      x[(size_t)i*stride]=s2/L[(size_t)i*(b+1)+b]; }
    for (i32 i=n-1;i>=0;i--){ double s2=x[(size_t)i*stride];
      for (i32 k=i+1;k<=i+b && k<n;k++) s2-=L[(size_t)k*(b+1)+(i-k+b)]*x[(size_t)k*stride];
      x[(size_t)i*stride]=s2/L[(size_t)i*(b+1)+b]; }
  }
};

// ---------------------------------------------------------------------------
//  exact circle cut quadrature.  Fluid is OUTSIDE the circle (phi = |x-c|-R).
//  Height-function rule per cut cell: transverse Gauss segments split at every
//  tangency (ct +- R) and interval-topology change (circle crossing a cell
//  edge in the height direction), so every 1-D integrand is smooth and the
//  tensor Gauss rule is machine-accurate.  Wall rule on exact arc segments.
// ---------------------------------------------------------------------------
struct CutCellQ {
  std::vector<double> vx, vy, vw;              // fluid-part volume rule
  std::vector<double> wx, wy, ww, wnx, wny;    // wall rule, n outward from fluid
  double area = 0;
};

struct Circle { double cx, cy, R; };

static void buildCutCell(const Circle &G, double x0, double y0, double h,
                         i32 ng, CutCellQ &Q) {
  GaussRule g = gaussLegendre(ng);
  // transverse = direction with larger |center offset| (interface more
  // perpendicular to it -> height function along the other dir is smooth)
  double mx = x0+0.5*h - G.cx, my = y0+0.5*h - G.cy;
  bool tIsY = fabs(mx) >= fabs(my);   // transverse coordinate is y; height along x
  double tc  = tIsY ? G.cy : G.cx;    // circle center in transverse coord
  double hc  = tIsY ? G.cx : G.cy;    // circle center in height coord
  double T0  = tIsY ? y0 : x0, T1 = T0 + h;
  double H0  = tIsY ? x0 : y0, H1 = H0 + h;
  // split points in the transverse interval
  std::vector<double> sp = {T0, T1};
  auto push=[&](double t){ if (t>T0+1e-13*h && t<T1-1e-13*h) sp.push_back(t); };
  push(tc-G.R); push(tc+G.R);
  for (double H : {H0, H1}) {
    double a2 = G.R*G.R-(H-hc)*(H-hc);
    if (a2 > 0) { double s=sqrt(a2); push(tc-s); push(tc+s); }
  }
  std::sort(sp.begin(), sp.end());
  for (size_t i=0;i+1<sp.size();i++) {
    double ta=sp[i], tb=sp[i+1]; if (tb-ta < 1e-13*h) continue;
    for (i32 q=0;q<g.n;q++) {
      double t = ta+(tb-ta)*(double)g.x[q], wt=(tb-ta)*(double)g.w[q];
      // fluid part of the height segment [H0,H1]: remove (hm,hp) inside circle
      double a2 = G.R*G.R-(t-tc)*(t-tc);
      double seg[2][2]; i32 nseg=0;
      if (a2 <= 0) { seg[0][0]=H0; seg[0][1]=H1; nseg=1; }
      else {
        double s=sqrt(a2), hm=hc-s, hp=hc+s;
        if (hm > H0) { seg[nseg][0]=H0; seg[nseg][1]=fmin(hm,H1); nseg++; }
        if (hp < H1) { seg[nseg][0]=fmax(hp,H0); seg[nseg][1]=H1; nseg++; }
      }
      for (i32 s2=0;s2<nseg;s2++) {
        double ha=seg[s2][0], hb=seg[s2][1]; if (hb-ha<=0) continue;
        for (i32 r=0;r<g.n;r++) {
          double hh=ha+(hb-ha)*(double)g.x[r], wh=(hb-ha)*(double)g.w[r]*wt;
          Q.vx.push_back(tIsY ? hh : t);
          Q.vy.push_back(tIsY ? t  : hh);
          Q.vw.push_back(wh); Q.area += wh;
        }
      }
    }
  }
  // wall: arc segments of the circle inside the cell box
  std::vector<double> th;
  auto pushTh=[&](double v){ th.push_back(atan2(sin(v),cos(v))); };
  for (double X : {x0, x0+h}) { double c=(X-G.cx)/G.R;
    if (fabs(c)<1.0) { pushTh(acos(c)); pushTh(-acos(c)); } }
  for (double Y : {y0, y0+h}) { double s=(Y-G.cy)/G.R;
    if (fabs(s)<1.0) { pushTh(asin(s)); pushTh(M_PI-asin(s)); } }
  if (th.empty()) return;                      // no wall in this cell
  std::sort(th.begin(), th.end());
  th.push_back(th.front()+2*M_PI);
  for (size_t i=0;i+1<th.size();i++) {
    double a=th[i], b=th[i+1]; if (b-a < 1e-14) continue;
    double tm=0.5*(a+b);
    double px=G.cx+G.R*cos(tm), py=G.cy+G.R*sin(tm);
    if (px<x0-1e-13*h||px>x0+h+1e-13*h||py<y0-1e-13*h||py>y0+h+1e-13*h) continue;
    for (i32 q=0;q<g.n;q++) {
      double tq=a+(b-a)*(double)g.x[q], w=G.R*(b-a)*(double)g.w[q];
      Q.wx.push_back(G.cx+G.R*cos(tq)); Q.wy.push_back(G.cy+G.R*sin(tq));
      Q.ww.push_back(w);
      Q.wnx.push_back(-cos(tq)); Q.wny.push_back(-sin(tq));  // outward from fluid
    }
  }
}

// ---------------------------------------------------------------------------
//  solver
// ---------------------------------------------------------------------------
struct Solver2d {
  i32 p, Nx, Ny, nx, ny, n;                    // spans / control points
  double x0d, y0d, h;
  Spline1d Sx, Sy;
  bool hasBody=false; Circle body;
  std::vector<i32> cls;                        // per cell: 0 fluid 1 cut 2 solid
  std::vector<i32> cutIdx;                     // cell -> cut record (-1)
  std::vector<CutCellQ> cq;
  std::vector<char> act;                       // control point active flag
  // ELL mass (2p+1)^2 stencil over control lattice + Kronecker preconditioner
  std::vector<double> Mell;
  BandChol Mx1, My1;
  std::vector<double> sdiag;      // sqrt(Kronecker_ii / M_ii): rescales the
                                  // small-support rows the Kronecker prec
                                  // cannot see (support fraction ~1e-2 ->
                                  // unpreconditioned tail, CG hit 200 iters)
  GaussRule gv;                                // volume rule (full cells)
  double Uinf[NF];
  // EV state
  std::vector<double> nuCell, nuS;
  double evNorm = 1.0, evDelta = 0.05;
  std::vector<double> Udot;
  // knobs
  double C_DC=1.0, C_MAX=0.5, C_SUPG=1.0, epsM=0.0, wallBeta=1.0, gpM=0.0; i32 fsp=0;
  i32 nuFrozen=0;
  double bandEVscale=1.0;   // JFNK: freeze nuCell/evNorm during J*v differencing
                    // (max/abs in the sensor are non-differentiable; lagged
                    // viscosity is the standard Newton treatment)

  i32 aidx(i32 i, i32 j) const { return i + nx*j; }

  void init(i32 p_, i32 Nx_, i32 Ny_, double x0_, double y0_, double h_) {
    p=p_; Nx=Nx_; Ny=Ny_; nx=Nx+p; ny=Ny+p; n=nx*ny;
    x0d=x0_; y0d=y0_; h=h_;
    Sx.init(p,Nx); Sy.init(p,Ny);
    gv = gaussLegendre(p+2);
    cls.assign((size_t)Nx*Ny, 0); cutIdx.assign((size_t)Nx*Ny, -1);
    act.assign(n, 1);
    nuCell.assign((size_t)Nx*Ny, 0.0); nuS.assign((size_t)Nx*Ny, 0.0);
    Udot.assign((size_t)NF*n, 0.0);
  }

  void classify(const Circle &G, i32 ngCut=0) {
    hasBody = true; body = G;
    if (ngCut<=0) ngCut=p+4;   // closure 8e-13 at p+4 vs 3e-9 at p+2
    cq.clear(); std::fill(cutIdx.begin(),cutIdx.end(),-1);
    for (i32 cy=0;cy<Ny;cy++) for (i32 cx=0;cx<Nx;cx++) {
      double bx0=x0d+cx*h, by0=y0d+cy*h, bx1=bx0+h, by1=by0+h;
      double ddx=fmax(fmax(bx0-G.cx, G.cx-bx1),0.0);
      double ddy=fmax(fmax(by0-G.cy, G.cy-by1),0.0);
      double dmin=sqrt(ddx*ddx+ddy*ddy);
      double dmax=0;
      for (double X : {bx0,bx1}) for (double Y : {by0,by1})
        dmax=fmax(dmax, sqrt((X-G.cx)*(X-G.cx)+(Y-G.cy)*(Y-G.cy)));
      i32 c = (dmin >= G.R) ? 0 : (dmax <= G.R ? 2 : 1);
      cls[(size_t)cx+Nx*cy]=c;
      if (c==1) {
        CutCellQ Q; buildCutCell(G, bx0, by0, h, ngCut, Q);
        if (Q.area < 1e-14*h*h) { cls[(size_t)cx+Nx*cy]=2; continue; }
        if (Q.area > h*h*(1-1e-14) && Q.ww.empty()) { cls[(size_t)cx+Nx*cy]=0; continue; }
        cutIdx[(size_t)cx+Nx*cy]=(i32)cq.size(); cq.push_back(std::move(Q));
      }
    }
    // active control points: support touches any fluid measure
    std::fill(act.begin(), act.end(), 0);
    for (i32 cy=0;cy<Ny;cy++) for (i32 cx=0;cx<Nx;cx++) {
      if (cls[(size_t)cx+Nx*cy]==2) continue;
      for (i32 a=0;a<=p;a++) for (i32 b=0;b<=p;b++) act[aidx(cx+a,cy+b)]=1;
    }
  }

  void buildMass() {
    const i32 W=2*p+1, WW=W*W;
    Mell.assign((size_t)n*WW, 0.0);
    real Nvx[BS_NMAX], Nvy[BS_NMAX];
    // GHOST-PENALTY mass (the elasticity stack's stabilizer, ported): for
    // C^{p-1} splines only the l=p normal-derivative jump survives at knot
    // lines; gamma h^{2p+1} * sum_F int_F [d^p_n psi_a][d^p_n psi_b] ds over
    // faces of cut cells.  Spectrally equivalent to h^2 on ALL active dofs
    // (ties small-support rows to their neighbours THROUGH the solid) and --
    // unlike the eps-mass -- row sums against constants are ZERO (the jump of
    // d^p(sum psi)=0), so conservation is untouched and no solid inertia is
    // added.  p=2 only here (analytic B'' knot jumps {+1,-3,+3,-1}).
    if (gpM > 0 && p==2) {
      const double J[4]={1.0,-3.0,3.0,-1.0};
      // 1-D span mass of the tangential direction (unit span, scaled by h)
      double Mt[3][3]={{0}};
      { GaussRule g1=gaussLegendre(p+1); real Nv[BS_NMAX];
        for (i32 q=0;q<g1.n;q++){ Sx.val(g1.x[q],Nv);
          for (i32 a2=0;a2<=p;a2++) for (i32 b2=0;b2<=p;b2++)
            Mt[a2][b2]+=(double)g1.w[q]*h*(double)Nv[a2]*(double)Nv[b2]; } }
      const double gam = gpM*pow(h,2*p+1)/(h*h*h*h);  // h^{2p+1} * (1/h^2)^2
      auto addFaceX=[&](i32 gx, i32 cy){        // vertical knot line x-index gx,
        for (i32 ia=gx-p; ia<=gx; ia++) {       // tangential span cy
          if (ia<0||ia>=nx) continue;
          double Ja=J[gx-ia];
          for (i32 ib=gx-p; ib<=gx; ib++) {
            if (ib<0||ib>=nx) continue;
            double Jb=J[gx-ib];
            for (i32 ja=0;ja<=p;ja++) for (i32 jb=0;jb<=p;jb++) {
              i32 arow=aidx(ia,cy+ja);
              i32 di=ib-ia+p, dj=(cy+jb)-(cy+ja)+p;
              if (di<0||di>2*p||dj<0||dj>2*p) continue;
              Mell[(size_t)arow*(2*p+1)*(2*p+1) + dj*(2*p+1)+di]
                += gam*Ja*Jb*Mt[ja][jb];
            } }
        } };
      auto addFaceY=[&](i32 gy, i32 cx){
        for (i32 ja=gy-p; ja<=gy; ja++) {
          if (ja<0||ja>=ny) continue;
          double Ja=J[gy-ja];
          for (i32 jb=gy-p; jb<=gy; jb++) {
            if (jb<0||jb>=ny) continue;
            double Jb=J[gy-jb];
            for (i32 ia=0;ia<=p;ia++) for (i32 ib=0;ib<=p;ib++) {
              i32 arow=aidx(cx+ia,ja);
              i32 di=(cx+ib)-(cx+ia)+p, dj=jb-ja+p;
              if (di<0||di>2*p||dj<0||dj>2*p) continue;
              Mell[(size_t)arow*(2*p+1)*(2*p+1) + dj*(2*p+1)+di]
                += gam*Ja*Jb*Mt[ia][ib];
            } }
        } };
      std::vector<char> fx((size_t)(Nx+1)*Ny,0), fy((size_t)Nx*(Ny+1),0);
      for (i32 cy=0;cy<Ny;cy++) for (i32 cx=0;cx<Nx;cx++) {
        if (cls[(size_t)cx+Nx*cy]!=1) continue;
        fx[(size_t)cx  +(Nx+1)*cy]=1; fx[(size_t)cx+1+(Nx+1)*cy]=1;
        fy[(size_t)cx+Nx*cy]=1;       fy[(size_t)cx+Nx*(cy+1)]=1;
      }
      for (i32 cy=0;cy<Ny;cy++) for (i32 g=0;g<=Nx;g++)
        if (fx[(size_t)g+(Nx+1)*cy]) addFaceX(g,cy);
      for (i32 cx=0;cx<Nx;cx++) for (i32 g=0;g<=Ny;g++)
        if (fy[(size_t)cx+Nx*g]) addFaceY(g,cx);
      // NOTE: do NOT widen the active set here.  A first version activated
      // every dof of solid cells adjacent to cut cells, but GP faces only tie
      // dofs whose support CROSSES a cut-cell face -- the rest became active
      // rows with ~zero mass, GMRES filled them with garbage, and every
      // Newton step tripped the positivity guard (dtau collapsed to 1e-7).
      // The fluid-support active set is the correct one; GP just conditions
      // its small rows.
    }
    // fictitious-domain mass (finite-cell alpha): epsM * solid-part mass on
    // cut cells and on solid cells with active dofs.  Bounds every active
    // mass row below (min support fraction 6.7e-3 otherwise), at the price
    // of an epsM-weighted solid-extension bias in conservation.
    if (epsM > 0) {
      for (i32 cy=0;cy<Ny;cy++) for (i32 cx=0;cx<Nx;cx++) {
        i32 c = cls[(size_t)cx+Nx*cy]; if (c==0) continue;
        bool any=false;
        for (i32 a=0;a<=p&&!any;a++) for (i32 b=0;b<=p;b++)
          if (act[aidx(cx+a,cy+b)]) { any=true; break; }
        if (!any) continue;
        for (i32 qx=0;qx<gv.n;qx++) for (i32 qy=0;qy<gv.n;qy++) {
          double X=x0d+(cx+(double)gv.x[qx])*h, Y=y0d+(cy+(double)gv.x[qy])*h;
          if (!hasBody) continue;
          double d=sqrt((X-body.cx)*(X-body.cx)+(Y-body.cy)*(Y-body.cy));
          if (d >= body.R) continue;                    // fluid part handled below
          Sx.val(gv.x[qx],Nvx); Sy.val(gv.x[qy],Nvy);
          double w=epsM*(double)gv.w[qx]*(double)gv.w[qy]*h*h;
          for (i32 a=0;a<=p;a++) for (i32 b=0;b<=p;b++) {
            double Pa=(double)Nvx[a]*(double)Nvy[b];
            i32 ia=cx+a, ja=cy+b;
            for (i32 a2=0;a2<=p;a2++) for (i32 b2=0;b2<=p;b2++) {
              double Pb=(double)Nvx[a2]*(double)Nvy[b2];
              i32 di=(cx+a2)-ia+p, dj=(cy+b2)-ja+p;
              Mell[(size_t)aidx(ia,ja)*WW + dj*W+di] += w*Pa*Pb;
            } }
        }
      }
      // solid-supported dofs become genuinely active through the eps mass
      for (i32 cy=0;cy<Ny;cy++) for (i32 cx=0;cx<Nx;cx++) {
        if (cls[(size_t)cx+Nx*cy]!=1) continue;
        for (i32 dj=-1;dj<=1;dj++) for (i32 di=-1;di<=1;di++) {
          i32 ii=cx+di, jj=cy+dj;
          if (ii<0||ii>=Nx||jj<0||jj>=Ny) continue;
          if (cls[(size_t)ii+Nx*jj]!=2) continue;
          for (i32 a=0;a<=p;a++) for (i32 b=0;b<=p;b++) act[aidx(ii+a,jj+b)]=1;
        } }
    }
    for (i32 cy=0;cy<Ny;cy++) for (i32 cx=0;cx<Nx;cx++) {
      i32 c = cls[(size_t)cx+Nx*cy]; if (c==2) continue;
      if (c==0) {
        for (i32 qx=0;qx<gv.n;qx++) for (i32 qy=0;qy<gv.n;qy++) {
          Sx.val(gv.x[qx],Nvx); Sy.val(gv.x[qy],Nvy);
          double w=(double)gv.w[qx]*(double)gv.w[qy]*h*h;
          for (i32 a=0;a<=p;a++) for (i32 b=0;b<=p;b++) {
            double Pa=(double)Nvx[a]*(double)Nvy[b];
            i32 ia=cx+a, ja=cy+b;
            for (i32 a2=0;a2<=p;a2++) for (i32 b2=0;b2<=p;b2++) {
              double Pb=(double)Nvx[a2]*(double)Nvy[b2];
              i32 di=(cx+a2)-ia+p, dj=(cy+b2)-ja+p;
              Mell[(size_t)aidx(ia,ja)*WW + dj*W+di] += w*Pa*Pb;
            } }
        }
      } else {
        const CutCellQ &Q = cq[cutIdx[(size_t)cx+Nx*cy]];
        for (size_t q=0;q<Q.vw.size();q++) {
          double xi=(Q.vx[q]-x0d)/h-cx, yi=(Q.vy[q]-y0d)/h-cy;
          Sx.val(xi,Nvx); Sy.val(yi,Nvy);
          for (i32 a=0;a<=p;a++) for (i32 b=0;b<=p;b++) {
            double Pa=(double)Nvx[a]*(double)Nvy[b];
            i32 ia=cx+a, ja=cy+b;
            for (i32 a2=0;a2<=p;a2++) for (i32 b2=0;b2<=p;b2++) {
              double Pb=(double)Nvx[a2]*(double)Nvy[b2];
              i32 di=(cx+a2)-ia+p, dj=(cy+b2)-ja+p;
              Mell[(size_t)aidx(ia,ja)*WW + dj*W+di] += Q.vw[q]*Pa*Pb;
            } }
        }
      }
    }
    for (i32 a=0;a<n;a++) if (!act[a]) Mell[(size_t)a*(2*p+1)*(2*p+1)+((2*p+1)*(2*p+1))/2]=1.0;
    // 1-D Kronecker factors of the FULL uniform grid (preconditioner)
    auto oneD=[&](Spline1d &S, i32 N, BandChol &M){
      M.n=S.n; M.b=p;
      std::vector<double> Ab((size_t)S.n*(p+1),0.0);
      GaussRule g1=gaussLegendre(p+1); real Nv[BS_NMAX];
      for (i32 s=0;s<N;s++) for (i32 q=0;q<g1.n;q++){
        S.val(g1.x[q],Nv);
        for (i32 a=0;a<=p;a++) for (i32 b2=0;b2<=a;b2++){
          i32 i=s+a, j=s+b2;
          Ab[(size_t)i*(p+1)+(j-i+p)] += (double)g1.w[q]*h*(double)Nv[a]*(double)Nv[b2];
        } }
      M.factor(Ab);
    };
    oneD(Sx,Nx,Mx1); oneD(Sy,Ny,My1);
    // diagonal of the full Kronecker mass from the 1-D diagonals
    { std::vector<double> dx(nx,0.0), dy(ny,0.0);
      GaussRule g1=gaussLegendre(p+1); real Nv[BS_NMAX];
      for (i32 s2=0;s2<Nx;s2++) for (i32 q=0;q<g1.n;q++){ Sx.val(g1.x[q],Nv);
        for (i32 a=0;a<=p;a++) dx[s2+a]+=(double)g1.w[q]*h*(double)Nv[a]*(double)Nv[a]; }
      for (i32 s2=0;s2<Ny;s2++) for (i32 q=0;q<g1.n;q++){ Sy.val(g1.x[q],Nv);
        for (i32 a=0;a<=p;a++) dy[s2+a]+=(double)g1.w[q]*h*(double)Nv[a]*(double)Nv[a]; }
      const i32 WW=(2*p+1)*(2*p+1);
      sdiag.assign(n,1.0);
      for (i32 a=0;a<n;a++){
        if (!act[a]) continue;
        double mii=Mell[(size_t)a*WW+WW/2], kii=dx[a%nx]*dy[a/nx]/h;
        if (mii>0 && kii>0) sdiag[a]=sqrt(kii/mii);
      } }
  }

  void massApply(const double *v, double *out) const {
    const i32 W=2*p+1, WW=W*W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i32 a=0;a<n;a++) {
      i32 i=a%nx, j=a/nx;
      double s[NF]={0,0,0,0};
      const double *row=&Mell[(size_t)a*WW];
      for (i32 dj=-p;dj<=p;dj++) {
        i32 jj=j+dj; if (jj<0||jj>=ny) continue;
        for (i32 di=-p;di<=p;di++) {
          i32 ii=i+di; if (ii<0||ii>=nx) continue;
          double m=row[(dj+p)*W+(di+p)]; if (m==0.0) continue;
          const double *vv=&v[(size_t)NF*(ii+nx*jj)];
          for (i32 k=0;k<NF;k++) s[k]+=m*vv[k];
        } }
      for (i32 k=0;k<NF;k++) out[(size_t)NF*a+k]=s[k];
    }
  }

  // preconditioner: z = D (My (x) Mx)^-1 D r, D = diag(sdiag), masked to active
  void precApply(const double *r, double *z, std::vector<double> &scr) const {
    scr.resize((size_t)n);
    for (i32 k=0;k<NF;k++) {
      for (i32 a=0;a<n;a++) scr[a]=sdiag[a]*r[(size_t)NF*a+k];
      for (i32 j=0;j<ny;j++) Mx1.solve(&scr[(size_t)j*nx], 1);
      for (i32 i=0;i<nx;i++) My1.solve(&scr[(size_t)i], nx);
      for (i32 a=0;a<n;a++) z[(size_t)NF*a+k]= act[a]?sdiag[a]*scr[a]:0.0;
    }
  }

  // PCG solve M x = b (in place on b), returns iterations
  i32 massSolve(std::vector<double> &b, std::vector<double> &x) const {
    static thread_local std::vector<double> r, zv, pv, Ap, scr;
    size_t m=b.size();
    r=b; x.assign(m,0.0); zv.assign(m,0.0); Ap.assign(m,0.0);
    for (size_t i=0;i<m;i++) if (!act[i/NF]) r[i]=0;
    double b2=0; for (size_t i=0;i<m;i++) b2+=r[i]*r[i];
    if (b2==0) return 0;
    precApply(r.data(), zv.data(), scr);
    pv=zv;
    double rz=0; for (size_t i=0;i<m;i++) rz+=r[i]*zv[i];
    i32 it=0;
    for (; it<400; it++) {
      massApply(pv.data(), Ap.data());
      for (size_t i=0;i<m;i++) if (!act[i/NF]) Ap[i]=pv[i];
      double pAp=0; for (size_t i=0;i<m;i++) pAp+=pv[i]*Ap[i];
      double al=rz/pAp;
      double r2=0;
      for (size_t i=0;i<m;i++){ x[i]+=al*pv[i]; r[i]-=al*Ap[i]; r2+=r[i]*r[i]; }
      if (r2 < 1e-24*b2) { it++; break; }
      precApply(r.data(), zv.data(), scr);
      double rz2=0; for (size_t i=0;i<m;i++) rz2+=r[i]*zv[i];
      double be=rz2/rz; rz=rz2;
      for (size_t i=0;i<m;i++) pv[i]=zv[i]+be*pv[i];
    }
    return it;
  }

  // ------- RHS ---------------------------------------------------------------
  void evalCell(const std::vector<double> &U, i32 cx, i32 cy,
                double xi, double yi, double Uq[NF], double Ux[NF],
                double Uy[NF], double Ut[NF]) const {
    real Nvx[BS_NMAX], Nvy[BS_NMAX], Dvx[BS_NMAX], Dvy[BS_NMAX];
    Sx.val(xi,Nvx); Sx.der(xi,Dvx); Sy.val(yi,Nvy); Sy.der(yi,Dvy);
    for (i32 k2=0;k2<NF;k2++){Uq[k2]=Ux[k2]=Uy[k2]=Ut[k2]=0;}
    for (i32 a=0;a<=p;a++) for (i32 b=0;b<=p;b++) {
      double P=(double)Nvx[a]*(double)Nvy[b];
      double Px=(double)Dvx[a]/h*(double)Nvy[b];
      double Py=(double)Nvx[a]*(double)Dvy[b]/h;
      const double *uu=&U[(size_t)NF*aidx(cx+a,cy+b)];
      const double *ud=&Udot[(size_t)NF*aidx(cx+a,cy+b)];
      for (i32 k2=0;k2<NF;k2++){
        Uq[k2]+=P*uu[k2]; Ux[k2]+=Px*uu[k2]; Uy[k2]+=Py*uu[k2]; Ut[k2]+=P*ud[k2]; }
    }
  }

  double sensorAt(const double Uq[NF], const double Ux[NF], const double Uy[NF],
                  const double Ut[NF], double &eAbs) const {
    double du=1e-7, Up[NF];
    for (i32 k=0;k<NF;k++) Up[k]=Uq[k]+du*Ut[k];
    double e1=entEta2(Up);
    for (i32 k=0;k<NF;k++) Up[k]=Uq[k]-du*Ut[k];
    double e2=entEta2(Up);
    double eta_t=(e1-e2)/(2*du);
    for (i32 k=0;k<NF;k++) Up[k]=Uq[k]+du*Ux[k];
    double q1=entQdir(Up,1,0);
    for (i32 k=0;k<NF;k++) Up[k]=Uq[k]-du*Ux[k];
    double q2=entQdir(Up,1,0);
    for (i32 k=0;k<NF;k++) Up[k]=Uq[k]+du*Uy[k];
    double q3=entQdir(Up,0,1);
    for (i32 k=0;k<NF;k++) Up[k]=Uq[k]-du*Uy[k];
    double q4=entQdir(Up,0,1);
    eAbs = fabs(entEta2(Uq));
    return fabs(eta_t + (q1-q2)/(2*du) + (q3-q4)/(2*du));
  }

  void rhs(const std::vector<double> &U, std::vector<double> &R) {
    R.assign((size_t)NF*n, 0.0);
    if (!nuFrozen) rhsPass1(U);
    rhsPass2(U, R);
  }
  void rhsPass1(const std::vector<double> &U) {
    // ---- pass 1: cell-wise entropy viscosity -------------------------------
    double eMax=-1e300, eMin=1e300, eInt=0, fluidArea=0, rhoMax=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,16) reduction(max:eMax,rhoMax) reduction(min:eMin) reduction(+:eInt,fluidArea)
#endif
    for (i32 cc=0;cc<Nx*Ny;cc++) {
      i32 cx=cc%Nx, cy=cc/Nx;
      i32 c=cls[cc]; if (c==2){ nuS[cc]=0; continue; }
      double nmax=0, lamM=0;
      double Uq[NF],Ux[NF],Uy[NF],Ut[NF];
      if (c==0) {
        for (i32 qx=0;qx<gv.n;qx++) for (i32 qy=0;qy<gv.n;qy++) {
          evalCell(U,cx,cy,gv.x[qx],gv.x[qy],Uq,Ux,Uy,Ut);
          double u,v,cs,r,pr; primEval(Uq,r,u,v,pr,cs);
          lamM=fmax(lamM, sqrt(u*u+v*v)+cs);
          double eA, res=sensorAt(Uq,Ux,Uy,Ut,eA);
          double ev=entEta2(Uq);
          eMax=fmax(eMax,ev); eMin=fmin(eMin,ev); rhoMax=fmax(rhoMax,Uq[0]);
          eInt+=(double)gv.w[qx]*(double)gv.w[qy]*h*h*ev;
          fluidArea+=(double)gv.w[qx]*(double)gv.w[qy]*h*h;
          nmax=fmax(nmax, C_DC*h*h*res/evNorm);
        }
      } else {
        const CutCellQ &Q=cq[cutIdx[cc]];
        for (size_t q=0;q<Q.vw.size();q++) {
          double xi=(Q.vx[q]-x0d)/h-cx, yi=(Q.vy[q]-y0d)/h-cy;
          evalCell(U,cx,cy,xi,yi,Uq,Ux,Uy,Ut);
          double u,v,cs,r,pr; primEval(Uq,r,u,v,pr,cs);
          lamM=fmax(lamM, sqrt(u*u+v*v)+cs);
          double eA, res=sensorAt(Uq,Ux,Uy,Ut,eA);
          double ev=entEta2(Uq);
          eMax=fmax(eMax,ev); eMin=fmin(eMin,ev); rhoMax=fmax(rhoMax,Uq[0]);
          eInt+=Q.vw[q]*ev; fluidArea+=Q.vw[q];
          nmax=fmax(nmax, C_DC*h*h*res/evNorm);
        }
      }
      nuS[cc]=fmin(nmax, C_MAX*h*lamM);
    }
    // Guermond normalization ||eta - eta_bar||_inf with a PHYSICAL floor.
    // For entropy-flat flows (isentropic vortex: eta == 0 analytically) the
    // deviation is only the O(h^{p+1}) discrete wiggle -- dividing by it makes
    // the sensor O(1) in SMOOTH flow and cost the vortex gate a full order
    // (measured 1.8 vs design 3).  The floor delta*rhoMax (s-variations of
    // O(delta) count as significant) restores nu ~ h^{p+2} there while leaving
    // Sod untouched (its deviation ~0.09 dominates the floor).
    { double ebar = eInt/fmax(fluidArea,1e-300);
      double dev  = fmax(eMax-ebar, ebar-eMin);
      evNorm = fmax(dev, evDelta*fmax(rhoMax,1e-8)); }
    // 3x3 neighbour-max smoothing
    for (i32 cy=0;cy<Ny;cy++) for (i32 cx=0;cx<Nx;cx++) {
      double v=0;
      for (i32 dj=-1;dj<=1;dj++) for (i32 di=-1;di<=1;di++) {
        i32 ii=cx+di, jj=cy+dj;
        if (ii<0||ii>=Nx||jj<0||jj>=Ny) continue;
        v=fmax(v,nuS[(size_t)ii+Nx*jj]);
      }
      nuCell[(size_t)cx+Nx*cy]=v;
    }
    if (bandEVscale != 1.0) {
      // The EV band smears the wall (measured: killing it lifts the suction
      // peak -1.93 -> -2.39 and fixes stagnation) BUT zero band dissipation
      // leaves the aft flow's neutral modes undamped and PTC STALLS (dtau
      // 9e-6, residual frozen 1.3e-3): steady inviscid cylinder needs SOME
      // dissipation to select the steady solution.  Scale, don't switch.
      for (i32 cy=0;cy<Ny;cy++) for (i32 cx=0;cx<Nx;cx++) {
        bool band=false;
        for (i32 dj=-1;dj<=1&&!band;dj++) for (i32 di=-1;di<=1;di++) {
          i32 ii=cx+di, jj=cy+dj;
          if (ii<0||ii>=Nx||jj<0||jj>=Ny) continue;
          if (cls[(size_t)ii+Nx*jj]==1) { band=true; break; }
        }
        if (band) nuCell[(size_t)cx+Nx*cy]*=bandEVscale;
      }
    }
  }
  void rhsPass2(const std::vector<double> &U, std::vector<double> &R) {
    // ---- pass 2: assembly ---------------------------------------------------
#ifdef _OPENMP
    i32 nth=omp_get_max_threads();
#else
    i32 nth=1;
#endif
    static std::vector<std::vector<double>> Rloc;
    Rloc.resize(nth);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      i32 tid=omp_get_thread_num();
#else
      i32 tid=0;
#endif
      std::vector<double> &Rt=Rloc[tid];
      Rt.assign(R.size(),0.0);
      real Nvx[BS_NMAX],Nvy[BS_NMAX],Dvx[BS_NMAX],Dvy[BS_NMAX];
      auto scatterVol=[&](i32 cx,i32 cy,double xi,double yi,double w,
                          const std::vector<double> &Uc){
        double Uq[NF],Ux[NF],Uy[NF],Ut[NF];
        evalCell(Uc,cx,cy,xi,yi,Uq,Ux,Uy,Ut);
        double Fx[NF],Fy[NF],u,v,cs;
        eulerFlux2(Uq,Fx,Fy,u,v,cs);
        double nu=nuCell[(size_t)cx+Nx*cy];
        // GLS/SUPG baseline stabilization, steady residual, FULL Jacobian
        // pairing: -tau * (A grad psi)^T (A grad U).  Testing with psi=U the
        // form is -tau ||A grad U||^2 <= 0: dissipative BY CONSTRUCTION.  The
        // spectral shortcut -tau*lam*(grad psi)-pairing used first is sign-
        // INDEFINITE (A is neither symmetric nor definite) and blew the
        // vortex up in 8 steps (lam 1e12); the same scalarization is what the
        // 1-D test measured as "SUPG unstable".  Needed at all because the
        // entropy viscosity is correctly ~0 in smooth flow, and the central
        // Galerkin interior + characteristic-incomplete wall dissipation
        // carries a dt-independent unstable mode without it (measured blowup
        // at CFL 0.3 and 0.02 alike).  Residual-based: vanishes at steady
        // state; conservative (sum_a grad psi_a = 0).
        double gx[NF]={0,0,0,0}, gy[NF]={0,0,0,0};
        { double Ax[NF][NF], Ay[NF][NF], Res[NF];
          eulerJac2(Uq,Ax,Ay);
          for (i32 k=0;k<NF;k++){ Res[k]=0;
            for (i32 m=0;m<NF;m++) Res[k]+=Ax[k][m]*Ux[m]+Ay[k][m]*Uy[m]; }
          double lam=sqrt(u*u+v*v)+cs, tau=0.5*C_SUPG*h/fmax(lam,1e-12);
          for (i32 k=0;k<NF;k++) for (i32 m=0;m<NF;m++){
            gx[k]+=tau*Ax[m][k]*Res[m]; gy[k]+=tau*Ay[m][k]*Res[m]; }
        }
        Sx.val(xi,Nvx); Sx.der(xi,Dvx); Sy.val(yi,Nvy); Sy.der(yi,Dvy);
        for (i32 a=0;a<=p;a++) for (i32 b=0;b<=p;b++) {
          double Px=(double)Dvx[a]/h*(double)Nvy[b];
          double Py=(double)Nvx[a]*(double)Dvy[b]/h;
          double *out=&Rt[(size_t)NF*aidx(cx+a,cy+b)];
          for (i32 k=0;k<NF;k++)
            out[k]+= w*( Fx[k]*Px + Fy[k]*Py - nu*(Ux[k]*Px+Uy[k]*Py)
                         - Px*gx[k] - Py*gy[k] );
        }
      };
#ifdef _OPENMP
#pragma omp for schedule(dynamic,16) nowait
#endif
      for (i32 cc=0;cc<Nx*Ny;cc++) {
        i32 cx=cc%Nx, cy=cc/Nx;
        i32 c=cls[cc]; if (c==2) continue;
        if (c==0) {
          for (i32 qx=0;qx<gv.n;qx++) for (i32 qy=0;qy<gv.n;qy++)
            scatterVol(cx,cy,gv.x[qx],gv.x[qy],
                       (double)gv.w[qx]*(double)gv.w[qy]*h*h, U);
        } else {
          const CutCellQ &Q=cq[cutIdx[cc]];
          for (size_t q=0;q<Q.vw.size();q++)
            scatterVol(cx,cy,(Q.vx[q]-x0d)/h-cx,(Q.vy[q]-y0d)/h-cy,Q.vw[q],U);
          // wall flux: mirror-state Rusanov (zero mass/energy flux) or
          // transparent (free-stream-preservation mode)
          for (size_t q=0;q<Q.ww.size();q++) {
            double xi=(Q.wx[q]-x0d)/h-cx, yi=(Q.wy[q]-y0d)/h-cy;
            double Uq[NF],Ux[NF],Uy[NF],Ut[NF];
            evalCell(U,cx,cy,xi,yi,Uq,Ux,Uy,Ut);
            double nxq=Q.wnx[q], nyq=Q.wny[q], Fh[NF];
            if (fsp) {
              double Fx[NF],Fy[NF],u,v,cs; eulerFlux2(Uq,Fx,Fy,u,v,cs);
              for (i32 k=0;k<NF;k++) Fh[k]=Fx[k]*nxq+Fy[k]*nyq;
            } else {
              double Um[NF]; double un=(Uq[1]*nxq+Uq[2]*nyq);
              Um[0]=Uq[0]; Um[1]=Uq[1]-2*un*nxq; Um[2]=Uq[2]-2*un*nyq; Um[3]=Uq[3];
              rusanov(Uq,Um,nxq,nyq,Fh);
              // wallBeta scales ONLY the u.n penalty (the Rusanov dissipation
              // term; central part untouched): the mirror jump is purely the
              // normal momentum, so the extra term is beta-1 times it
              if (wallBeta != 1.0) {
                double u2,v2,cs2,r2,pr2; primEval(Uq,r2,u2,v2,pr2,cs2);
                double lam=fabs(u2*nxq+v2*nyq)+cs2;
                double ex=(wallBeta-1.0)*0.5*lam*2.0*un;   // -0.5*lam*(Um-Uq) extra
                Fh[1]+=ex*nxq; Fh[2]+=ex*nyq;
              }
            }
            Sx.val(xi,Nvx); Sy.val(yi,Nvy);
            for (i32 a=0;a<=p;a++) for (i32 b=0;b<=p;b++) {
              double P=(double)Nvx[a]*(double)Nvy[b];
              double *out=&Rt[(size_t)NF*aidx(cx+a,cy+b)];
              for (i32 k=0;k<NF;k++) out[k]-= Q.ww[q]*P*Fh[k];
            }
          }
        }
      }
      // far-field box edges (Rusanov vs free stream), split across threads by edge
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
      for (i32 e=0;e<4;e++) {
        i32 tang = (e<2)?1:0;                        // 0/1: x edges vary y; 2/3 vary x
        i32 nsp = tang? Ny : Nx;
        for (i32 s=0;s<nsp;s++) {
          for (i32 q=0;q<gv.n;q++) {
            double xi, yi, nxq, nyq;
            if (e==0){ xi=0;              yi=(double)gv.x[q]; nxq=-1; nyq=0; }
            else if (e==1){ xi=1;         yi=(double)gv.x[q]; nxq= 1; nyq=0; }
            else if (e==2){ xi=(double)gv.x[q]; yi=0;         nxq=0; nyq=-1; }
            else          { xi=(double)gv.x[q]; yi=1;         nxq=0; nyq= 1; }
            i32 cx = tang? (e==0?0:Nx-1) : s;
            i32 cy = tang? s : (e==2?0:Ny-1);
            double lx = tang? (e==0?0.0:1.0) : xi;
            double ly = tang? yi : (e==2?0.0:1.0);
            double Uq[NF],Ux[NF],Uy[NF],Ut[NF];
            evalCell(U,cx,cy,lx,ly,Uq,Ux,Uy,Ut);
            double Fh[NF]; rusanov(Uq,Uinf,nxq,nyq,Fh);
            Sx.val(lx,Nvx); Sy.val(ly,Nvy);
            double w=(double)gv.w[q]*h;
            for (i32 a=0;a<=p;a++) for (i32 b=0;b<=p;b++) {
              double P=(double)Nvx[a]*(double)Nvy[b];
              double *out=&Rt[(size_t)NF*aidx(cx+a,cy+b)];
              for (i32 k=0;k<NF;k++) out[k]-= w*P*Fh[k];
            }
          }
        }
      }
    }
    for (i32 t=0;t<nth;t++)
      for (size_t i=0;i<R.size();i++) R[i]+=Rloc[t][i];
    for (i32 a=0;a<n;a++) if (!act[a]) for (i32 k=0;k<NF;k++) R[(size_t)NF*a+k]=0;
  }

  // integral of a conserved component over the fluid
  double integrate(const std::vector<double> &U, i32 comp) const {
    double s=0;
    for (i32 cc=0;cc<Nx*Ny;cc++) {
      i32 cx=cc%Nx, cy=cc/Nx, c=cls[cc]; if (c==2) continue;
      double Uq[NF],Ux[NF],Uy[NF],Ut[NF];
      if (c==0) {
        for (i32 qx=0;qx<gv.n;qx++) for (i32 qy=0;qy<gv.n;qy++) {
          evalCell(U,cx,cy,gv.x[qx],gv.x[qy],Uq,Ux,Uy,Ut);
          s+=(double)gv.w[qx]*(double)gv.w[qy]*h*h*Uq[comp];
        }
      } else {
        const CutCellQ &Q=cq[cutIdx[cc]];
        for (size_t q=0;q<Q.vw.size();q++) {
          evalCell(U,cx,cy,(Q.vx[q]-x0d)/h-cx,(Q.vy[q]-y0d)/h-cy,Uq,Ux,Uy,Ut);
          s+=Q.vw[q]*Uq[comp];
        }
      }
    }
    return s;
  }
};

static double g_csupg = 0.0;
static double g_epsm = 0.0;

// SSP-RK3 driver with lagged Udot
struct Stepper {
  Solver2d &S;
  std::vector<double> U0, Us, R, X;
  i32 lastIt=0;
  Stepper(Solver2d &s):S(s){}
  double wavespeedMax(const std::vector<double> &U) {
    double lm=0;
    for (i32 a=0;a<S.n;a++) {
      if (!S.act[a]) continue;
      double r,u,v,pr,c; primEval(&U[(size_t)NF*a],r,u,v,pr,c);
      lm=fmax(lm, sqrt(u*u+v*v)+c);
    }
    return lm;
  }
  void step(std::vector<double> &U, double dt) {
    size_t m=U.size(); U0=U;
    S.rhs(U,R); S.massSolve(R,X); S.Udot=X;
    Us.resize(m);
    for (size_t i=0;i<m;i++) Us[i]=U0[i]+dt*X[i];
    S.rhs(Us,R); lastIt=S.massSolve(R,X);
    for (size_t i=0;i<m;i++) Us[i]=0.75*U0[i]+0.25*(Us[i]+dt*X[i]);
    S.rhs(Us,R); S.massSolve(R,X);
    for (size_t i=0;i<m;i++) U[i]=(U0[i]+2.0*(Us[i]+dt*X[i]))/3.0;
  }
};

// ---------------------------------------------------------------------------
//  PTC-JFNK: backward-Euler pseudo-transient continuation, Jacobian-free
//  GMRES on (M/dtau - J) delta = R(U), right-preconditioned by the Kronecker
//  mass (P^-1 = dtau * Ktilde^-1).  SER timestep growth, positivity-guarded.
// ---------------------------------------------------------------------------
struct Ptc {
  Solver2d &S;
  std::vector<double> Rbase;
  double normv(const std::vector<double> &v){ double s2=0;
    for (size_t i=0;i<v.size();i++) s2+=v[i]*v[i]; return sqrt(s2); }
  Ptc(Solver2d &s):S(s){}
  // A*v = M v/dtau - (R(U+eps v)-R(U))/eps   (nu frozen by caller)
  void applyA(const std::vector<double> &U, const std::vector<double> &v,
              double dtau, double normU, std::vector<double> &out,
              std::vector<double> &scr1, std::vector<double> &scr2) {
    double nv=normv(v);
    double eps = sqrt(1e-13)*sqrt(1.0+normU)/fmax(nv,1e-300);
    scr1.resize(v.size());
    for (size_t i=0;i<v.size();i++) scr1[i]=U[i]+eps*v[i];
    S.rhs(scr1, scr2);                       // nuFrozen=1 during solves
    S.massApply(v.data(), scr1.data());      // reuse scr1 for M v
    out.resize(v.size());
    for (size_t i=0;i<v.size();i++)
      out[i]= scr1[i]/dtau - (scr2[i]-Rbase[i])/eps;
    for (size_t i=0;i<v.size();i++) if (!S.act[i/NF]) out[i]=v[i];
  }
  // right-preconditioned GMRES(m): solve A x = b, returns iterations (<0 fail)
  i32 gmres(const std::vector<double> &U, const std::vector<double> &b,
            double dtau, std::vector<double> &x, i32 m, double rtol) {
    size_t Nd=b.size();
    double normU=normv(U);
    x.assign(Nd,0.0);
    std::vector<std::vector<double>> V; std::vector<double> H((size_t)(m+1)*m,0.0);
    std::vector<double> cs(m), sn(m), g(m+1), scr1, scr2, w(Nd), pz(Nd), scr3;
    double b2=normv(b); if (b2==0) return 0;
    i32 total=0;
    for (i32 restart=0; restart<3; restart++) {
      // r = b - A x  (x=0 first pass)
      std::vector<double> r=b;
      if (total>0) { applyA(U,xPrec(x,dtau,pz,scr3),dtau,normU,w,scr1,scr2);
        for (size_t i=0;i<Nd;i++) r[i]=b[i]-w[i]; }
      double beta=normv(r); if (beta/b2<rtol) return total;
      V.assign(1,r); for (size_t i=0;i<Nd;i++) V[0][i]/=beta;
      std::fill(g.begin(),g.end(),0.0); g[0]=beta;
      i32 k=0;
      for (; k<m; k++) {
        applyA(U,xPrec(V[k],dtau,pz,scr3),dtau,normU,w,scr1,scr2);
        for (i32 j=0;j<=k;j++){ double hjk=0;
          for (size_t i=0;i<Nd;i++) hjk+=w[i]*V[j][i];
          H[(size_t)j*m+k]=hjk;
          for (size_t i=0;i<Nd;i++) w[i]-=hjk*V[j][i]; }
        double hk1=normv(w); H[(size_t)(k+1)*m+k]=hk1;
        if (hk1>1e-30){ V.push_back(w); for (size_t i=0;i<Nd;i++) V[k+1][i]/=hk1; }
        for (i32 j=0;j<k;j++){ double t=cs[j]*H[(size_t)j*m+k]+sn[j]*H[(size_t)(j+1)*m+k];
          H[(size_t)(j+1)*m+k]=-sn[j]*H[(size_t)j*m+k]+cs[j]*H[(size_t)(j+1)*m+k];
          H[(size_t)j*m+k]=t; }
        double d=sqrt(H[(size_t)k*m+k]*H[(size_t)k*m+k]+hk1*hk1);
        cs[k]=H[(size_t)k*m+k]/d; sn[k]=hk1/d;
        H[(size_t)k*m+k]=d; g[k+1]=-sn[k]*g[k]; g[k]=cs[k]*g[k];
        total++;
        if (fabs(g[k+1])/b2 < rtol || hk1<=1e-30) { k++; break; }
      }
      // back substitution, x += V y
      std::vector<double> y(k);
      for (i32 i=k-1;i>=0;i--){ double t=g[i];
        for (i32 j=i+1;j<k;j++) t-=H[(size_t)i*m+j]*y[j];
        y[i]=t/H[(size_t)i*m+i]; }
      for (i32 j=0;j<k;j++) for (size_t i=0;i<Nd;i++) x[i]+=y[j]*V[j][i];
      if (fabs(g[k>0?k:0])/b2 < rtol || (k<m)) break;
    }
    return total;
  }
  // helper: z = P^-1 v = dtau * Ktilde^-1 v (masked)
  const std::vector<double>& xPrec(const std::vector<double> &v, double dtau,
                                   std::vector<double> &pz, std::vector<double> &scr) {
    pz.resize(v.size());
    S.precApply(v.data(), pz.data(), scr);
    for (size_t i=0;i<pz.size();i++) pz[i]*=dtau;
    return pz;
  }
};

// ---------------------------------------------------------------------------
//  gates
// ---------------------------------------------------------------------------
static void vortexExact(double x, double y, double t, double beta,
                        double u0, double v0, double U[NF]) {
  double xc=5.0+u0*t, yc=5.0+v0*t;
  double dx=x-xc, dy=y-yc, r2=dx*dx+dy*dy;
  double ex=exp(0.5*(1.0-r2));
  double du=-beta/(2*M_PI)*ex*dy, dv=beta/(2*M_PI)*ex*dx;
  double T=1.0-(GAM-1)*beta*beta/(8*GAM*M_PI*M_PI)*exp(1.0-r2);
  double rho=pow(T,1.0/(GAM-1)), u=u0+du, v=v0+dv, pr=rho*T;
  U[0]=rho; U[1]=rho*u; U[2]=rho*v; U[3]=pr/(GAM-1)+0.5*rho*(u*u+v*v);
}

static i32 gateVortex(i32 p, double CFL, double CDC, double CMAX) {
  printf("\n[vortex] isentropic vortex, UNCUT grid, L2(rho) orders (design %d)\n", p+1);
  printf("%6s %10s %12s %8s %10s %8s\n","N","dofs","L2(rho)","order","nuMax/cap","CGit");
  double prev=0; i32 ok=1;
  // T and the error window are chosen so the window (r<=3 around the moved
  // center) stays CAUSALLY clean of the far-field boxes: the Rusanov edges see
  // the vortex tail (~4e-6) as free-stream error, and at N=128 that floor was
  // the biggest term in a whole-domain L2 (measured 8e-7 stall).
  const double beta=1.0, u0=1.0, v0=0.5, T=0.5;
  for (i32 N : {32, 64, 128}) {
    Solver2d S; S.init(p,N,N,0.0,0.0,10.0/N);
    S.C_DC=CDC; S.C_MAX=CMAX; S.C_SUPG=g_csupg;
    for (i32 k=0;k<NF;k++) S.Uinf[k]=0;
    { double Ui[NF]; vortexExact(-100,-100,0,beta,u0,v0,Ui);
      for (i32 k=0;k<NF;k++) S.Uinf[k]=Ui[k]; }
    S.buildMass();
    // L2 projection of the IC through the Kronecker mass (grid is uncut)
    std::vector<double> U((size_t)NF*S.n,0.0), b((size_t)NF*S.n,0.0), X;
    { real Nvx[BS_NMAX],Nvy[BS_NMAX];
      for (i32 cy=0;cy<N;cy++) for (i32 cx=0;cx<N;cx++)
        for (i32 qx=0;qx<S.gv.n;qx++) for (i32 qy=0;qy<S.gv.n;qy++) {
          double x=(cx+(double)S.gv.x[qx])*S.h, y=(cy+(double)S.gv.x[qy])*S.h;
          double Ue[NF]; vortexExact(x,y,0,beta,u0,v0,Ue);
          S.Sx.val(S.gv.x[qx],Nvx); S.Sy.val(S.gv.x[qy],Nvy);
          double w=(double)S.gv.w[qx]*(double)S.gv.w[qy]*S.h*S.h;
          for (i32 a=0;a<=p;a++) for (i32 b2=0;b2<=p;b2++) {
            double P=(double)Nvx[a]*(double)Nvy[b2];
            for (i32 k=0;k<NF;k++)
              b[(size_t)NF*S.aidx(cx+a,cy+b2)+k]+=w*P*Ue[k];
          } }
      S.massSolve(b,U);
    }
    Stepper st(S);
    double t=0, numax=0, itacc=0, nst=0; i32 guard=0;
    while (t<T-1e-12) {
      double lm=st.wavespeedMax(U);
      if (!(lm>0) || lm>50 || nst>20000) { guard=1; break; }   // blowup guard:
        // a collapsing dt turns a diverging run into an apparent hang
      double dt=fmin(CFL*S.h/lm, T-t);
      st.step(U,dt); t+=dt;
      for (i32 cc=0;cc<N*N;cc++) numax=fmax(numax,S.nuCell[cc]);
      itacc+=st.lastIt; nst++;
      if (((i32)nst)%200==0) fprintf(stderr,"  [vortex N=%d] t=%.3f lam=%.2f\n",N,t,lm);
    }
    if (guard) { printf("%6d  BLOWUP (lam=%.3e after %d steps)\n", N,
                        st.wavespeedMax(U), (i32)nst); ok=0; prev=0; continue; }
    double e2=0, area=0;
    { double Uq[NF],Ux[NF],Uy[NF],Ut[NF];
      for (i32 cy=0;cy<N;cy++) for (i32 cx=0;cx<N;cx++)
        for (i32 qx=0;qx<S.gv.n;qx++) for (i32 qy=0;qy<S.gv.n;qy++) {
          double x=(cx+(double)S.gv.x[qx])*S.h, y=(cy+(double)S.gv.x[qy])*S.h;
          S.evalCell(U,cx,cy,S.gv.x[qx],S.gv.x[qy],Uq,Ux,Uy,Ut);
          double dxc=x-(5.0+u0*T), dyc=y-(5.0+v0*T);
          if (dxc*dxc+dyc*dyc > 9.0) continue;
          double Ue[NF]; vortexExact(x,y,T,beta,u0,v0,Ue);
          double w=(double)S.gv.w[qx]*(double)S.gv.w[qy]*S.h*S.h;
          e2+=w*(Uq[0]-Ue[0])*(Uq[0]-Ue[0]); area+=w;
        } }
    e2=sqrt(e2/area);
    double cap=CMAX*S.h*2.0;
    printf("%6d %10d %12.4e %8.2f %10.2e %8.1f\n", N, NF*S.n, e2,
           prev>0? log2(prev/e2):0.0, numax/cap, itacc/fmax(nst,1.0));
    if (prev>0 && log2(prev/e2) < p+0.4) ok=0;
    prev=e2;
  }
  printf("[vortex] %s\n", ok? "PASS":"FAIL (order below design)");
  return ok;
}

static i32 gateFsp(i32 p) {
  // Free-stream CONSISTENCY is a STATIC statement here: the weak residual of
  // the exact uniform state is the discrete divergence theorem per basis
  // function (volume rule vs wall rule).  The fluid part of a cut cell is not
  // a polynomial subdomain (sqrt bounds), so composite Gauss converges
  // exponentially rather than terminating -- the ladder below measures that.
  // Time-marching with a TRANSPARENT wall is NOT a valid gate: on the inflow
  // half of the arc a one-sided flux has no exterior data (ill-posed), and
  // with C_DC=0 the continuous-Galerkin interior carries zero dissipation, so
  // any closure seed grows without bound (measured: 3e-9 -> 27, dt-independent).
  printf("\n[fsp] free-stream closure through the CUT mesh (static) + real-wall march\n");
  const double M=0.3, aoa=25.0*M_PI/180.0;
  double r=1.0, c=1.0, pr=r*c*c/GAM, u=M*c*cos(aoa), v=M*c*sin(aoa);
  double closure=1e300; i32 ok=1;
  for (i32 ng : {4, 6, 8}) {
    Solver2d S; S.init(p,64,64,-4.0,-4.0,8.0/64);
    S.fsp=1; S.C_DC=0; S.C_MAX=0; S.C_SUPG=0;   // closure probe stays pure Galerkin
    S.Uinf[0]=r; S.Uinf[1]=r*u; S.Uinf[2]=r*v; S.Uinf[3]=pr/(GAM-1)+0.5*r*(u*u+v*v);
    Circle G{0.0,0.0,0.5};
    S.classify(G, ng); S.buildMass();
    if (ng==4) {
      i32 ncut=0, nsolid=0; double amin=1e300;
      for (i32 cc=0;cc<64*64;cc++){ if(S.cls[cc]==1){ncut++;
          amin=fmin(amin,S.cq[S.cutIdx[cc]].area/(S.h*S.h));} if(S.cls[cc]==2)nsolid++; }
      double warea=0; for (auto &Q:S.cq) for (double w:Q.ww) warea+=w;
      printf("  cut cells %d, solid %d, min rel area %.2e, wall length %.12f (exact %.12f)\n",
             ncut, nsolid, amin, warea, 2*M_PI*G.R);
    }
    std::vector<double> U((size_t)NF*S.n), Rp;
    for (i32 a=0;a<S.n;a++) for (i32 k=0;k<NF;k++) U[(size_t)NF*a+k]=S.Uinf[k];
    S.rhs(U,Rp);
    double rmax=0;
    for (i32 a=0;a<S.n;a++){ if(!S.act[a])continue;
      for (i32 k=0;k<NF;k++) rmax=fmax(rmax,fabs(Rp[(size_t)NF*a+k])); }
    printf("  ng=%d: raw |R|max (uniform state) = %.3e\n", ng, rmax);
    closure=rmax;
  }
  ok &= closure < 1e-11;
  // real wall, entropy viscosity on, uniform IC: must stay BOUNDED (the flow
  // physically deflects; this is the stability leg, not preservation)
  { Solver2d S; S.init(p,64,64,-4.0,-4.0,8.0/64);
    S.C_SUPG=g_csupg; S.epsM=g_epsm;
    S.Uinf[0]=r; S.Uinf[1]=r*u; S.Uinf[2]=r*v; S.Uinf[3]=pr/(GAM-1)+0.5*r*(u*u+v*v);
    Circle G{0.0,0.0,0.5};
    S.classify(G); S.buildMass();
    std::vector<double> U((size_t)NF*S.n);
    for (i32 a=0;a<S.n;a++) for (i32 k=0;k<NF;k++) U[(size_t)NF*a+k]=S.Uinf[k];
    Stepper st(S);
    const double cfl = getenv("IGA2_CFL")? atof(getenv("IGA2_CFL")) : 0.3;
    double drift=0; i32 bad=0;
    for (i32 it=0;it<100;it++) {
      double lm=st.wavespeedMax(U);
      if (!(lm>0) || lm>100) { bad=1; break; }
      st.step(U, cfl*S.h/lm);
    }
    for (i32 a=0;a<S.n;a++){ if(!S.act[a])continue;
      for (i32 k=0;k<NF;k++){ double d=fabs(U[(size_t)NF*a+k]-S.Uinf[k]);
        if (!std::isfinite(d)) bad=1; else drift=fmax(drift,d); } }
    printf("  real wall, 100 steps: %s, max deflection %.3e, CG %d\n",
           bad?"BLOWUP":"bounded", drift, st.lastIt);
    ok &= !bad && drift < 1.0;
  }
  printf("[fsp] %s\n", ok? "PASS":"FAIL");
  return ok;
}

static void gateSteady(i32 p, double CDC, double CMAX) {
  const double M = getenv("IGA2_MACH")? atof(getenv("IGA2_MACH")) : 0.3;
  const i32 N = getenv("IGA2_N")? atoi(getenv("IGA2_N")) : 160;
  const i32 maxIt = getenv("IGA2_PITS")? atoi(getenv("IGA2_PITS")) : 120;
  printf("\n[steady] PTC-JFNK, M=%.2f cylinder D=1, N=%d, wallBeta=%.0f, epsM=%.2f\n",
         M, N, getenv("IGA2_WBETA")? atof(getenv("IGA2_WBETA")):16.0,
         g_epsm>0? g_epsm:0.1);
  Solver2d S; S.init(p,N,N,-8.0,-8.0,16.0/N);
  S.C_DC=CDC; S.C_MAX=CMAX; S.C_SUPG=g_csupg;
  S.gpM  = getenv("IGA2_GPM")? atof(getenv("IGA2_GPM")) : 0.0;
  S.epsM = g_epsm>0? g_epsm : (S.gpM>0? 0.0 : 0.1);
  S.wallBeta = getenv("IGA2_WBETA")? atof(getenv("IGA2_WBETA")) : 16.0;
  S.bandEVscale = getenv("IGA2_BEV")? atof(getenv("IGA2_BEV")) : (getenv("IGA2_NOBEV")? 0.0 : 1.0);
  double r=1.0, c=1.0, pr=r*c*c/GAM, u=M*c;
  S.Uinf[0]=r; S.Uinf[1]=r*u; S.Uinf[2]=0; S.Uinf[3]=pr/(GAM-1)+0.5*r*u*u;
  Circle G{0.0,0.0,0.5};
  S.classify(G); S.buildMass();
  std::vector<double> U((size_t)NF*S.n);
  for (i32 a=0;a<S.n;a++) for (i32 k=0;k<NF;k++) U[(size_t)NF*a+k]=S.Uinf[k];
  // Udot stays 0: the sensor becomes the SPATIAL entropy residual, which is
  // exactly the quantity PTC drives to zero -- self-limiting viscosity.
  std::fill(S.Udot.begin(), S.Udot.end(), 0.0);
  Ptc ptc(S);
  std::vector<double> delta, y, pz, scr, Rtrial, Utrial;
  double lam0 = u+c;
  double dtau = 10.0*S.h/lam0;                  // pseudo-CFL 10 start
  S.nuFrozen=0; S.rhs(U, ptc.Rbase);
  double R0=ptc.normv(ptc.Rbase), Rn=R0;
  double qd=0.5*r*u*u;
  printf("%4s %10s %10s %8s %6s %9s %9s %9s\n",
         "it","||R||/R0","dtau","gmres","rej","Cd","max u.n","nuMax");
  i32 rejects=0;
  for (i32 it=1; it<=maxIt; it++) {
    S.nuFrozen=1;                               // lagged viscosity in the solve
    const i32 gm=60;
    i32 gi = ptc.gmres(U, ptc.Rbase, dtau, y, gm, 1e-3);
    // right-prec closure: delta = P^-1 y
    S.precApply(y.data(), (delta.resize(y.size()), delta.data()), scr);
    for (size_t i=0;i<delta.size();i++) delta[i]*=dtau;
    Utrial=U;
    for (size_t i=0;i<U.size();i++) Utrial[i]+=delta[i];
    // positivity + sanity guard on control coefficients
    bool ok=true;
    for (i32 a=0;a<S.n && ok;a++){ if(!S.act[a])continue;
      const double *Ua=&Utrial[(size_t)NF*a];
      double rr=Ua[0], pp=(GAM-1)*(Ua[3]-0.5*(Ua[1]*Ua[1]+Ua[2]*Ua[2])/fmax(rr,1e-12));
      if (!(rr>1e-8) || !(pp>1e-10) || !std::isfinite(Ua[3])) ok=false; }
    double Rt=1e300;
    if (ok) { S.nuFrozen=0; S.rhs(Utrial, Rtrial); Rt=ptc.normv(Rtrial); ok = (Rt < 3.0*Rn); }
    if (!ok) {
      dtau*=0.3; rejects++;
      S.nuFrozen=0; S.rhs(U, ptc.Rbase);        // restore nu state for U
      if (dtau < 1e-6*S.h/lam0) { printf("  STALL: dtau collapsed\n"); break; }
      continue;
    }
    U=Utrial; ptc.Rbase=Rtrial;
    // SER growth ONLY when the linear solve actually converged inside its
    // budget -- growing dtau on an unconverged solve compounds: J dominates,
    // the mass preconditioner fades, and the nonlinear rate stalls at the
    // linear-solve accuracy (measured 0.93/iter with a hard 120 cap).
    if (gi < 3*gm-2) {
      double grow = Rn/fmax(Rt,1e-300);
      dtau *= fmin(fmax(grow,0.5),2.0);
      dtau = fmin(dtau, 1e5*S.h/lam0);
    }
    Rn=Rt;
    // wall diagnostics
    double fx=0, unmax=0, numax=0;
    for (i32 cc=0;cc<N*N;cc++){
      if (S.cls[cc]!=1) continue;
      const CutCellQ &Q=S.cq[S.cutIdx[cc]];
      i32 cx=cc%N, cy=cc/N;
      for (size_t q=0;q<Q.ww.size();q++){
        double xi=(Q.wx[q]-S.x0d)/S.h-cx, yi=(Q.wy[q]-S.y0d)/S.h-cy;
        double Uq[NF],Ux[NF],Uy[NF],Ut[NF];
        S.evalCell(U,cx,cy,xi,yi,Uq,Ux,Uy,Ut);
        double rr,uu,vv,pp,ccs; primEval(Uq,rr,uu,vv,pp,ccs);
        fx+=Q.ww[q]*pp*(-Q.wnx[q]);
        unmax=fmax(unmax, fabs(uu*Q.wnx[q]+vv*Q.wny[q])/u);
      } }
    for (i32 cc=0;cc<N*N;cc++) numax=fmax(numax,S.nuCell[cc]);
    printf("%4d %10.3e %10.3e %8d %6d %9.4f %9.2e %9.2e\n",
           it, Rn/R0, dtau, gi, rejects, fx/qd, unmax, numax);
    fflush(stdout);
    if (Rn/R0 < 1e-8) { printf("  CONVERGED\n"); break; }
  }
  if (getenv("IGA2_DUMP")) {
    FILE *fp=fopen(getenv("IGA2_DUMP"),"w");
    fprintf(fp,"theta,cp,un\n");
    for (i32 cc=0;cc<N*N;cc++){
      if (S.cls[cc]!=1) continue;
      const CutCellQ &Q=S.cq[S.cutIdx[cc]];
      i32 cx=cc%N, cy=cc/N;
      for (size_t q=0;q<Q.ww.size();q++){
        double xi=(Q.wx[q]-S.x0d)/S.h-cx, yi=(Q.wy[q]-S.y0d)/S.h-cy;
        double Uq[NF],Ux[NF],Uy[NF],Ut[NF];
        S.evalCell(U,cx,cy,xi,yi,Uq,Ux,Uy,Ut);
        double rr,uu,vv,pp,ccs; primEval(Uq,rr,uu,vv,pp,ccs);
        double th=atan2(Q.wy[q],Q.wx[q]);
        fprintf(fp,"%.6f,%.6e,%.6e\n",th,(pp-pr)/qd,(uu*Q.wnx[q]+vv*Q.wny[q])/u);
      } }
    fclose(fp);
    printf("  wrote %s\n", getenv("IGA2_DUMP"));
  }
}

static void gateCyl(i32 p, double CFL, double CDC, double CMAX) {
  const double M = getenv("IGA2_MACH")? atof(getenv("IGA2_MACH")) : 0.3;
  const double T = getenv("IGA2_TEND")? atof(getenv("IGA2_TEND")) : 30.0;
  const i32 N = getenv("IGA2_N")? atoi(getenv("IGA2_N")) : 160;
  printf("\n[cyl] M=%.2f cylinder D=1, domain 16D, N=%d (h=D/%.0f), t->%.1f\n",
         M, N, 0.1*N*1.0/1.6, T);
  Solver2d S; S.init(p,N,N,-8.0,-8.0,16.0/N);
  S.C_DC=CDC; S.C_MAX=CMAX; S.C_SUPG=g_csupg; S.epsM=g_epsm;
  S.wallBeta = getenv("IGA2_WBETA")? atof(getenv("IGA2_WBETA")) : 1.0;
  double r=1.0, c=1.0, pr=r*c*c/GAM, u=M*c;
  S.Uinf[0]=r; S.Uinf[1]=r*u; S.Uinf[2]=0; S.Uinf[3]=pr/(GAM-1)+0.5*r*u*u;
  Circle G{0.0,0.0,0.5};
  S.classify(G); S.buildMass();
  std::vector<double> U((size_t)NF*S.n);
  for (i32 a=0;a<S.n;a++) for (i32 k=0;k<NF;k++) U[(size_t)NF*a+k]=S.Uinf[k];
  Stepper st(S);
  double t=0, M0=S.integrate(U,0);
  double qd=0.5*r*u*u;
  printf("%8s %10s %9s %9s %9s %9s %7s\n",
         "t","dM/M0","Cd","Cl","max u.n","nuMax","CGit");
  i32 it=0, nextrep=1;
  while (t<T-1e-12) {
    double lm=st.wavespeedMax(U);
    if (!(lm>0) || lm>50) { printf("  BLOWUP at t=%.3f (lam=%.3e)\n",t,lm); fflush(stdout); break; }
    double dt=fmin(CFL*S.h/lm, T-t);
    st.step(U,dt); t+=dt; it++;
    if (t >= 0.999*nextrep*T/10.0 || t>=T-1e-12) {
      nextrep++;
      double fx=0, fy=0, unmax=0, numax=0;
      for (i32 cc=0;cc<N*N;cc++){
        if (S.cls[cc]!=1) continue;
        const CutCellQ &Q=S.cq[S.cutIdx[cc]];
        i32 cx=cc%N, cy=cc/N;
        for (size_t q=0;q<Q.ww.size();q++){
          double xi=(Q.wx[q]-S.x0d)/S.h-cx, yi=(Q.wy[q]-S.y0d)/S.h-cy;
          double Uq[NF],Ux[NF],Uy[NF],Ut[NF];
          S.evalCell(U,cx,cy,xi,yi,Uq,Ux,Uy,Ut);
          double rr,uu,vv,pp,ccs; primEval(Uq,rr,uu,vv,pp,ccs);
          fx+=Q.ww[q]*pp*(-Q.wnx[q]); fy+=Q.ww[q]*pp*(-Q.wny[q]);
          unmax=fmax(unmax, fabs(uu*Q.wnx[q]+vv*Q.wny[q])/u);
        } }
      for (i32 cc=0;cc<N*N;cc++) numax=fmax(numax,S.nuCell[cc]);
      // force ON the body = -integral of p n_fluid  ->  fx,fy as accumulated
      printf("%8.2f %10.2e %9.4f %9.4f %9.2e %9.2e %7d\n",
             t, (S.integrate(U,0)-M0)/M0, fx/(qd*1.0), fy/(qd*1.0),
             unmax, numax, st.lastIt);
      fflush(stdout);
    }
  }
  // Cp(theta) dump
  if (getenv("IGA2_DUMP")) {
    FILE *fp=fopen(getenv("IGA2_DUMP"),"w");
    fprintf(fp,"theta,cp,un\n");
    for (i32 cc=0;cc<N*N;cc++){
      if (S.cls[cc]!=1) continue;
      const CutCellQ &Q=S.cq[S.cutIdx[cc]];
      i32 cx=cc%N, cy=cc/N;
      for (size_t q=0;q<Q.ww.size();q++){
        double xi=(Q.wx[q]-S.x0d)/S.h-cx, yi=(Q.wy[q]-S.y0d)/S.h-cy;
        double Uq[NF],Ux[NF],Uy[NF],Ut[NF];
        S.evalCell(U,cx,cy,xi,yi,Uq,Ux,Uy,Ut);
        double rr,uu,vv,pp,ccs; primEval(Uq,rr,uu,vv,pp,ccs);
        double th=atan2(Q.wy[q],Q.wx[q]);
        fprintf(fp,"%.6f,%.6e,%.6e\n",th,(pp-pr)/qd,(uu*Q.wnx[q]+vv*Q.wny[q])/u);
      } }
    fclose(fp);
    printf("  wrote %s\n", getenv("IGA2_DUMP"));
  }
}

int main(int argc, char **argv) {
  const char *mode = (argc>1)? argv[1] : "all";
  const i32 p = (argc>2)? atoi(argv[2]) : 2;
  const double CFL  = getenv("IGA2_CFL") ? atof(getenv("IGA2_CFL")) : 0.3;
  const double CDC  = getenv("IGA2_CDC") ? atof(getenv("IGA2_CDC")) : 1.0;
  const double CMAX = getenv("IGA2_CMAX")? atof(getenv("IGA2_CMAX")): 0.5;
  const double CSUP = getenv("IGA2_CSUPG")? atof(getenv("IGA2_CSUPG")): 0.0;
  g_epsm = getenv("IGA2_EPSM")? atof(getenv("IGA2_EPSM")) : 0.0;
  printf("IGA 2-D Euler, C^%d tensor B-splines (p=%d), entropy viscosity "
         "(C=%.2f cap=%.2f), cut-cell cylinder\n", p-1, p, CDC, CMAX);
  i32 ok=1;
  g_csupg=CSUP;
  if (!strcmp(mode,"vortex")||!strcmp(mode,"all")) ok &= gateVortex(p,CFL,CDC,CMAX);
  if (!strcmp(mode,"fsp")   ||!strcmp(mode,"all")) ok &= gateFsp(p);
  if (!strcmp(mode,"cyl")) gateCyl(p,CFL,CDC,CMAX);
  if (!strcmp(mode,"steady")) gateSteady(p,CDC,CMAX);
  printf("\n%s\n", ok? "ALL GATES PASS":"GATE FAILURE");
  return ok?0:1;
}
