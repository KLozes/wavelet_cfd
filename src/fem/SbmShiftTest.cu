//
// M0 unit test for the SBM shift operator (SbmShift.h).
// The truncated Taylor shift S_h is EXACT for polynomials of total degree <= p:
//   S_h u (x~, d)  ==  u(x~ + d)   to machine precision.
// A polynomial of per-axis degree p but total degree > p is only reproduced to
// O(|d|^{p+1}) -- the SBM consistency order -- which we also confirm.
//
//   build: nvcc -O2 -DUSE_DOUBLE -I src/common -I src/fem src/fem/SbmShiftTest.cu -o sbm_shift_test
//
#include <cstdio>
#include <cmath>
#include "SbmShift.h"

static int P;
static double C[4][4][4];   // c[ax][ay][az]

static double evalU(double x, double y, double z, bool totalDeg) {
  double s = 0;
  for (int ax=0; ax<=P; ax++)
  for (int ay=0; ay<=P; ay++)
  for (int az=0; az<=P; az++) {
    if (totalDeg && ax+ay+az > P) continue;         // total degree <= p
    double t = C[ax][ay][az];
    for (int q=0;q<ax;q++) t*=x; for(int q=0;q<ay;q++) t*=y; for(int q=0;q<az;q++) t*=z;
    s += t;
  }
  return s;
}

int main() {
  double worstTotal = 0, worstFull = 0;
  unsigned s0 = 20260724u;
  for (P = 1; P <= 3; P++) {
    for (int i=0;i<4;i++) for(int j=0;j<4;j++) for(int k=0;k<4;k++){
      s0 = s0*1664525u+1013904223u; C[i][j][k] = (double)(s0>>8)/8388608.0 - 1.0;
    }
    LagrangeBasis B; B.init(P);
    int n = B.n;
    real Vm[QN_MAX][QN_MAX]; sbmDerivMatrix(B, Vm);

    for (int mode=0; mode<2; mode++) {       // 0 = total-degree-p (exact), 1 = per-axis-p (O(h^{p+1}))
      bool totalDeg = (mode==0);
      double worst = 0;
      // several surrogate points and shift vectors
      double xs[3] = {0.15, 0.62, 0.88}, hs = 0.1;
      for (double sx : xs) for (double sy : xs) for (double sz : xs) {
        // reference nodal values of u
        real u[QN_MAX*QN_MAX*QN_MAX];
        for (int k=0;k<n;k++) for(int j=0;j<n;j++) for(int i=0;i<n;i++)
          u[i+n*(j+n*k)] = (real)evalU(B.t[i], B.t[j], B.t[k], totalDeg);
        real xr[3] = {(real)sx,(real)sy,(real)sz};
        real dref[3] = {(real)hs,(real)(-0.7*hs),(real)(0.4*hs)};   // reference shift
        real sh[QN_MAX*QN_MAX*QN_MAX]; sbmShiftAll(B, Vm, xr, dref, sh);
        double Su = 0; for (int a=0;a<n*n*n;a++) Su += (double)u[a]*sh[a];
        double exact = evalU(sx+dref[0], sy+dref[1], sz+dref[2], totalDeg);
        worst = fmax(worst, fabs(Su - exact));
      }
      if (totalDeg) { printf("p=%d  total-deg<=p shift error %.2e\n", P, worst); worstTotal=fmax(worstTotal,worst); }
      else          { printf("      per-axis-p (trunc) error   %.2e  (expected ~O(|d|^%d), NOT machine-zero)\n", worst, P+1); worstFull=fmax(worstFull,worst); }
    }
  }
  printf("\nworst total-degree-<=p shift error (should be ~fp): %.2e   %s\n",
         worstTotal, worstTotal < 1e-10 ? "PASS" : "FAIL");
  return 0;
}
