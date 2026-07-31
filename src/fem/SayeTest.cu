//
// Standalone order test for the Saye cut quadrature (SayeQuad.h).
//
// Tiles a domain into N^3 cells, fits a degree-p level-set polynomial per cell
// from GLL-node samples of a KNOWN implicit surface, runs sayeVolume/sayeSurface
// on each cut/interior cell, and compares the summed volume of {phi<0} and area
// of {phi=0} against the analytic values.  The error must fall like O(h^{p+1}).
//
//   build: nvcc -O2 -DUSE_DOUBLE -I src/common -I src/fem src/fem/SayeTest.cu -o saye_test
//   run:   ./saye_test
//

#include <cstdio>
#include <cmath>
#include "SayeQuad.h"
#include "PolyFit.h"

// ---- test level sets (physical coordinates) -------------------------------
// each returns phi<0 inside; volume/area are the analytic reference.

struct Sphere {           // SDF of a sphere: transcendental -> true O(h^{p+1})
  double cx, cy, cz, R;
  __host__ double phi(double x, double y, double z) const {
    double dx=x-cx, dy=y-cy, dz=z-cz;
    return sqrt(dx*dx+dy*dy+dz*dz) - R;
  }
  double vol() const { return 4.0/3.0*M_PI*R*R*R; }
  double area() const { return 4.0*M_PI*R*R; }
};

struct Torus {            // SDF of a torus: non-convex, has a hole -> the
  double cx, cy, cz, Rmaj, Rmin;   // interface exits side faces near the ring,
  __host__ double phi(double x, double y, double z) const {  // exercising the
    double dx=x-cx, dy=y-cy, dz=z-cz;                         // base splitters
    double q = sqrt(dx*dx+dy*dy) - Rmaj;
    return sqrt(q*q + dz*dz) - Rmin;
  }
  double vol() const { return 2.0*M_PI*M_PI*Rmaj*Rmin*Rmin; }
  double area() const { return 4.0*M_PI*M_PI*Rmaj*Rmin; }
};

// ---------------------------------------------------------------------------
template<class LS>
void runVolumeArea(const char *name, const LS &ls, i32 p,
                   double dlo, double dhi, double refVol, double refArea) {
  printf("\n== %s, p=%d (deg-%d level set), domain [%.2f,%.2f]^3 ==\n",
         name, p, p, dlo, dhi);
  printf("  %4s  %14s  %8s  %14s  %8s\n", "N", "volErr", "ord", "areaErr", "ord");

  static SayeNode arenaBuf[1<<18];
  static SayeNode outBuf[1<<16];
  double prevV = 0, prevA = 0;
  for (i32 N : {8, 16, 32, 64}) {
    double h = (dhi - dlo) / N;
    double vol = 0, area = 0;
    i32 nOvf = 0;
    for (i32 cz = 0; cz < N; cz++)
    for (i32 cy = 0; cy < N; cy++)
    for (i32 cx = 0; cx < N; cx++) {
      double x0 = dlo + cx*h, y0 = dlo + cy*h, z0 = dlo + cz*h;
      // sample phi at the (p+1)^3 GLL nodes in this cell
      real t[PNC]; gllNodes(p, t);
      real v[PNC*PNC*PNC];
      i32 n = p+1;
      bool anyNeg=false, anyPos=false;
      for (i32 k = 0; k < n; k++)
      for (i32 j = 0; j < n; j++)
      for (i32 i = 0; i < n; i++) {
        double xx = x0 + t[i]*h, yy = y0 + t[j]*h, zz = z0 + t[k]*h;
        double f = ls.phi(xx, yy, zz);
        v[i + n*(j + n*k)] = (real)f;
        if (f < 0) anyNeg = true; else anyPos = true;
      }
      if (!anyNeg) continue;                       // cell entirely outside
      PolyND poly = fitPoly3(p, v);
      SayeArena ar; ar.buf = arenaBuf; ar.cap = 1<<18; ar.top = 0;
      SayeSet out; out.p = outBuf; out.n = 0; out.cap = 1<<16; out.ovf = false;

      if (!anyPos) {                               // interior cell: full volume
        vol += h*h*h;
      } else {
        sayeVolume(poly, &out, &ar, SayeCfg::def());
        if (out.ovf) nOvf++;
        for (i32 q = 0; q < out.n; q++) vol += out.p[q].w * h*h*h;
      }
      // surface (cut cells only)
      if (anyPos && anyNeg) {
        SayeArena ar2; ar2.buf = arenaBuf; ar2.cap = 1<<18; ar2.top = 0;
        SayeSet sout; sout.p = outBuf; sout.n = 0; sout.cap = 1<<16; sout.ovf = false;
        sayeSurface(poly, &sout, &ar2, SayeCfg::def());
        for (i32 q = 0; q < sout.n; q++) area += sout.p[q].w * h*h;  // dS scales as h^2
      }
    }
    double ve = fabs(vol - refVol), ae = (refArea>0)? fabs(area - refArea) : 0;
    double ov = (prevV>0)? log(prevV/ve)/log(2.0) : 0;
    double oa = (prevA>0)? log(prevA/ae)/log(2.0) : 0;
    if (refArea > 0)
      printf("  %4d  %14.3e  %8.2f  %14.3e  %8.2f%s\n", N, ve, ov, ae, oa,
             nOvf? "  [ovf]":"");
    else
      printf("  %4d  %14.3e  %8.2f  %14s  %8s%s\n", N, ve, ov, "-", "-",
             nOvf? "  [ovf]":"");
    prevV = ve; prevA = ae;
  }
}

// ---------------------------------------------------------------------------
//  CSG crease test: a spherical cap = sphere INTERSECT half-space {z > dcut}.
//  The boundary is a spherical cap + a flat disk meeting at a CREASE circle,
//  where phi = max(phi_sphere, phi_plane) has a kink.  Compare:
//    SINGLE : fit ONE degree-p polynomial to max(phi_s,phi_p) (oscillates)
//    MULTI  : fit phi_s and phi_p SEPARATELY, integrate {phi_s<0 & phi_p<0}
//  Both branches are algebraic (deg<=2), so MULTI's geometry is exact; any
//  MULTI error is pure quadrature.  SINGLE's error is the crease.
// ---------------------------------------------------------------------------
void runCreaseTest(i32 p) {
  const double R = 0.8, cz = 0.0, dcut = 0.3;
  const double d = dcut - cz, hc = R - d;
  const double volEx  = M_PI*hc*hc*(3*R - hc)/3.0;
  const double areaEx = 2*M_PI*R*hc + M_PI*(R*R - d*d);   // cap + disk
  auto phiS = [&](double x,double y,double z){ return x*x+y*y+(z-cz)*(z-cz)-R*R; };
  auto phiP = [&](double x,double y,double z){ return dcut - z; };   // <0 => z>dcut

  printf("\n== spherical-cap CREASE test, p=%d (sphere INT {z>%.2f}) ==\n", p, dcut);
  printf("   exact  vol %.8f  area %.8f\n", volEx, areaEx);
  printf("   %4s | %13s %8s | %13s %8s   (single = max fit, multi = CSG-aware)\n",
         "N", "vol err(1)", "vol(M)", "area err(1)", "area(M)");

  static SayeNode arena[1<<18], obuf[1<<16];
  i32 n = p+1; real t[PNC]; gllNodes(p, t);
  for (i32 N : {8, 16, 32, 64}) {
    double h = 2.0/N;
    double vS=0, vM=0, aS=0, aM=0;
    for (i32 cz2=0; cz2<N; cz2++) for (i32 cy=0; cy<N; cy++) for (i32 cx=0; cx<N; cx++) {
      double x0=-1+cx*h, y0=-1+cy*h, z0=-1+cz2*h;
      real vc[PNC*PNC*PNC], vs[PNC*PNC*PNC], vp[PNC*PNC*PNC];
      bool anyNeg=false, anyPos=false;
      for (i32 k=0;k<n;k++) for(i32 j=0;j<n;j++) for(i32 i=0;i<n;i++){
        double X=x0+t[i]*h, Y=y0+t[j]*h, Z=z0+t[k]*h;
        double a=phiS(X,Y,Z), b=phiP(X,Y,Z), m=fmax(a,b);
        vs[i+n*(j+n*k)]=(real)a; vp[i+n*(j+n*k)]=(real)b; vc[i+n*(j+n*k)]=(real)m;
        if(m<0)anyNeg=true; else anyPos=true;
      }
      if(!anyNeg) continue;
      if(!anyPos){ vS+=h*h*h; vM+=h*h*h; continue; }      // interior
      // SINGLE: one poly on max
      { PolyND pc=fitPoly3(p,vc);
        SayeArena ar; ar.buf=arena; ar.cap=1<<18; ar.top=0;
        SayeSet o; o.p=obuf;o.n=0;o.cap=1<<16;o.ovf=false;
        sayeVolume(pc,&o,&ar); for(i32 q=0;q<o.n;q++) vS+=o.p[q].w*h*h*h;
        SayeArena ar2; ar2.buf=arena; ar2.cap=1<<18; ar2.top=0;
        SayeSet os; os.p=obuf;os.n=0;os.cap=1<<16;os.ovf=false;
        sayeSurface(pc,&os,&ar2); for(i32 q=0;q<os.n;q++) aS+=os.p[q].w*h*h; }
      // MULTI: two branch polys on the intersection
      { PolyND ps=fitPoly3(p,vs), pp=fitPoly3(p,vp), br[2]={ps,pp};
        SayeArena ar; ar.buf=arena; ar.cap=1<<18; ar.top=0;
        SayeSet o; o.p=obuf;o.n=0;o.cap=1<<16;o.ovf=false;
        sayeVolumeMulti(br,2,&o,&ar); for(i32 q=0;q<o.n;q++) vM+=o.p[q].w*h*h*h;
        SayeArena ar2; ar2.buf=arena; ar2.cap=1<<18; ar2.top=0;
        SayeSet os; os.p=obuf;os.n=0;os.cap=1<<16;os.ovf=false;
        sayeSurfaceMulti(br,2,&os,&ar2); for(i32 q=0;q<os.n;q++) aM+=os.p[q].w*h*h; }
    }
    printf("   %4d | %13.3e %8.1e | %13.3e %8.1e\n",
           N, fabs(vS-volEx), fabs(vM-volEx), fabs(aS-areaEx), fabs(aM-areaEx));
  }
}

int main() {
  Sphere s{0.03125, 0.015625, 0.0234375, 0.75};   // off-grid center, R=0.75
  for (i32 p : {1, 2, 3, 4})
    runVolumeArea("sphere SDF", s, p, -1.0, 1.0, s.vol(), s.area());

  for (i32 p : {1, 2, 3}) runCreaseTest(p);

  Torus tor{0.017, -0.021, 0.013, 0.6, 0.25};      // off-grid, hole along z
  for (i32 p : {1, 2, 3})
    runVolumeArea("torus SDF", tor, p, -1.0, 1.0, tor.vol(), tor.area());

  return 0;
}
