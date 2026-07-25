//
// Patch test for the Qp element elasticity operator (QpElem.h) -- the M1 gate.
// For p=1..3, uncut (tensor GLL) and cut (Saye), verify:
//   (A) symmetry:  K = K^T                         (self-adjoint bilinear form)
//   (B) rigid-body null space: K u = 0 for the 3 translations + 3 rotations
//   (C) constant strain: for u = A.x (A const sym), 1/2 u^T K u == strain energy
//       density * |Omega_e|  (exact reproduction of a linear displacement)
//
//   build: nvcc -O2 -DUSE_DOUBLE -I src/common -I src/fem src/fem/QpElemTest.cu -o qpe_test
//

#include <cstdio>
#include <cmath>
#include "QpElem.h"
#include "PolyFit.h"

static const real MU = (real)0.8, LAM = (real)1.7, H = (real)0.37;

// build the dense element matrix by applying the operator to each unit vector
template<class Apply>
static void buildK(int ndof3, Apply apply, double *K) {
  static real u[3*QN_MAX*QN_MAX*QN_MAX], y[3*QN_MAX*QN_MAX*QN_MAX];
  for (int c = 0; c < ndof3; c++) {
    for (int a = 0; a < ndof3; a++) u[a] = (a==c)?1:0;
    apply(u, y);
    for (int r = 0; r < ndof3; r++) K[r*ndof3+c] = y[r];
  }
}

static double symErr(const double *K, int m) {
  double e=0, s=0;
  for (int i=0;i<m;i++) for (int j=0;j<m;j++){ e=fmax(e,fabs(K[i*m+j]-K[j*m+i])); s=fmax(s,fabs(K[i*m+j])); }
  return e/(s>0?s:1);
}

static double matVecNorm(const double *K, int m, const real *u) {
  double e=0;
  for (int i=0;i<m;i++){ double s=0; for(int j=0;j<m;j++) s+=K[i*m+j]*u[j]; e=fmax(e,fabs(s)); }
  return e;
}

int main() {
  static SayeNode arenaBuf[1<<18], outBuf[1<<16];
  double worst = 0;

  for (int p = 1; p <= 3; p++) {
    QpBasis B; B.init(p);
    int n = B.n, ndof = n*n*n, m = 3*ndof;

    for (int cutCase = 0; cutCase < 2; cutCase++) {
      // uncut: whole cube.  cut: a half-space phi = xi_0 - 0.5 (< 0 keeps left
      // half) -> exact half volume, a clean check that the Saye path integrates
      // the same bulk form correctly on a partial cell.
      const char *tag = cutCase? "cut(half)" : "uncut";
      PolyND phi; // half-space level set for the cut case (degree 1)
      {
        real v[PNC*PNC*PNC]; real t[PNC]; gllNodes(p, t);
        for (int k=0;k<n;k++) for (int j=0;j<n;j++) for (int i=0;i<n;i++)
          v[i+n*(j+n*k)] = t[i] - (real)0.5;
        phi = fitPoly3(p, v);
      }
      auto apply = [&](const real *u, real *y) {
        if (!cutCase) qpElemUncut(B, MU, LAM, H, u, y);
        else qpElemCut(B, MU, LAM, H, phi, u, y, arenaBuf, 1<<18, outBuf, 1<<16);
      };

      static double K[ (3*QN_MAX*QN_MAX*QN_MAX)*(3*QN_MAX*QN_MAX*QN_MAX) ];
      buildK(m, apply, K);

      // (A) symmetry
      double eSym = symErr(K, m);

      // (B) rigid body: 3 translations + 3 rotations (linearized)
      static real u[3*QN_MAX*QN_MAX*QN_MAX];
      double eRig = 0;
      // node coords
      auto nodeX = [&](int a, real X[3]){
        int i=a%n, j=(a/n)%n, k=a/(n*n);
        X[0]=B.t[i]; X[1]=B.t[j]; X[2]=B.t[k];
      };
      for (int mode = 0; mode < 6; mode++) {
        for (int a=0;a<ndof;a++){
          real X[3]; nodeX(a,X);
          real d[3]={0,0,0};
          if (mode<3) d[mode]=1;                       // translation
          else { int ax=mode-3; // rotation about axis ax
            int i1=(ax+1)%3, i2=(ax+2)%3;
            d[i1]=-X[i2]; d[i2]=X[i1];
          }
          for (int i=0;i<3;i++) u[3*a+i]=d[i];
        }
        eRig = fmax(eRig, matVecNorm(K, m, u));
      }

      // (C) constant strain u = A.x, A symmetric const.  strain energy =
      //     1/2 u^T K u ; exact = (mu A:A + lam/2 tr(A)^2) * |Omega_e|
      double Am[3][3] = {{0.3,0.1,-0.2},{0.1,-0.15,0.05},{-0.2,0.05,0.25}};
      for (int a=0;a<ndof;a++){
        real X[3]; nodeX(a,X);
        for (int i=0;i<3;i++) u[3*a+i]=(real)(Am[i][0]*(X[0]*H)+Am[i][1]*(X[1]*H)+Am[i][2]*(X[2]*H));
      }
      // NOTE physical coords = X*H (cube of size H at origin). u = A.x_phys.
      double uKu=0;
      for (int i=0;i<m;i++){ double s=0; for(int j=0;j<m;j++) s+=K[i*m+j]*u[j]; uKu+=u[i]*s; }
      double energy = 0.5*uKu;
      double AA=0, trA=Am[0][0]+Am[1][1]+Am[2][2];
      for (int i=0;i<3;i++) for(int j=0;j<3;j++) AA+=Am[i][j]*Am[i][j];
      double vol = (cutCase? 0.5:1.0)*H*H*H;
      double exact = (MU*AA + 0.5*LAM*trA*trA)*vol;
      double eCS = fabs(energy-exact)/fabs(exact);

      printf("p=%d %-9s  sym %.2e   rigid %.2e   const-strain %.2e\n",
             p, tag, eSym, eRig, eCS);
      worst = fmax(worst, fmax(eSym, fmax(eRig, eCS)));
    }
  }
  printf("\nworst relative error, all p / uncut+cut:  %.2e   %s\n",
         worst, worst < 1e-9 ? "PASS" : "FAIL");
  return 0;
}
