//
// Unit tests for the Qp reference-cell basis (LagrangeBasis.h) -- the M1 foundation.
// Decisive checks, each must pass to machine precision for p=1..3:
//   (1) differentiation matrix reproduces d/dx of a degree-p polynomial at nodes
//   (2) barycentric eval/grad reproduce a degree-p poly + gradient at ARBITRARY
//       points (this is what the Saye cut points exercise)
//   (3) GLL tensor quadrature integrates a degree-(2p-1) polynomial exactly
//
//   build: nvcc -O2 -DUSE_DOUBLE -I src/common -I src/fem src/fem/QpTest.cu -o qp_test
//

#include <cstdio>
#include <cmath>
#include "LagrangeBasis.h"

// a reference-cell test field: product of 1-D degree-p polynomials, so it is in
// Q_p exactly.  f(x) = gx(x0) gy(x1) gz(x2)
static double coefX[4], coefY[4], coefZ[4];
static int    P;
static double poly1(const double c[4], double x) {
  double s=0, xp=1; for (int i=0;i<=P;i++){ s+=c[i]*xp; xp*=x; } return s;
}
static double dpoly1(const double c[4], double x) {
  double s=0, xp=1; for (int i=1;i<=P;i++){ s+=i*c[i]*xp; xp*=x; } return s;
}
static double fval(double x,double y,double z){ return poly1(coefX,x)*poly1(coefY,y)*poly1(coefZ,z); }
static void   fgrad(double x,double y,double z,double g[3]){
  g[0]=dpoly1(coefX,x)*poly1(coefY,y)*poly1(coefZ,z);
  g[1]=poly1(coefX,x)*dpoly1(coefY,y)*poly1(coefZ,z);
  g[2]=poly1(coefX,x)*poly1(coefY,y)*dpoly1(coefZ,z);
}

int main() {
  double worst = 0;
  for (P = 1; P <= 3; P++) {
    coefX[0]=0.7; coefX[1]=-1.3; coefX[2]=0.9; coefX[3]=-0.4;
    coefY[0]=-0.2; coefY[1]=1.1; coefY[2]=-0.6; coefY[3]=0.8;
    coefZ[0]=1.0; coefZ[1]=0.5; coefZ[2]=-0.7; coefZ[3]=0.3;
    LagrangeBasis B; B.init(P);
    int n = B.n;
    // nodal values
    static real u[QN_MAX*QN_MAX*QN_MAX];
    for (int k=0;k<n;k++) for (int j=0;j<n;j++) for (int i=0;i<n;i++)
      u[i+n*(j+n*k)] = (real)fval(B.t[i],B.t[j],B.t[k]);

    // (1) differentiation matrix at nodes
    double e1 = 0;
    for (int k=0;k<n;k++) for (int j=0;j<n;j++) for (int i=0;i<n;i++) {
      real x[3]={B.t[i],B.t[j],B.t[k]}, g[3]; B.gradRef(x,u,g);
      double ge[3]; fgrad(B.t[i],B.t[j],B.t[k],ge);
      for (int d=0;d<3;d++) e1 = fmax(e1, fabs(g[d]-ge[d]));
    }

    // (2) eval + grad at arbitrary (non-node) points
    double e2v=0, e2g=0;
    double sample[5]={0.137,0.331,0.5,0.789,0.913};
    for (double sx : sample) for (double sy : sample) for (double sz : sample) {
      real x[3]={(real)sx,(real)sy,(real)sz}, g[3];
      real v = B.eval(x,u); B.gradRef(x,u,g);
      double ge[3]; fgrad(sx,sy,sz,ge);
      e2v = fmax(e2v, fabs((double)v - fval(sx,sy,sz)));
      for (int d=0;d<3;d++) e2g = fmax(e2g, fabs(g[d]-ge[d]));
    }

    // (3) GLL tensor quadrature of a degree-(2p-1) polynomial (exact)
    //     integrand q(x)=x^(2p-1); int_0^1 = 1/(2p).  test each axis 1-D.
    double e3 = 0;
    {
      int deg = 2*P-1;
      double I=0;
      for (int i=0;i<n;i++) { double xp=1; for(int e=0;e<deg;e++) xp*=B.t[i]; I += B.wq[i]*xp; }
      e3 = fabs(I - 1.0/(2*P));
    }

    printf("p=%d:  D-matrix %.2e   eval %.2e  grad@pts %.2e   GLL-quad(deg %d) %.2e\n",
           P, e1, e2v, e2g, 2*P-1, e3);
    worst = fmax(worst, fmax(fmax(e1,e2v), fmax(e2g,e3)));
  }
  printf("\nworst error over all tests, p=1..3:  %.2e   %s\n",
         worst, worst < 1e-10 ? "PASS" : "FAIL");
  return 0;
}
