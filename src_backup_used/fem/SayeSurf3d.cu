// Gallery: 3-D reconstructed boundaries of several CSG INTERSECTION geometries,
// single-polynomial (fit max) vs multi-polynomial CSG-aware Saye quadrature.
// Every geometry is evaluated in a ROTATED frame so no crease lands on a grid
// face (an axis-aligned crease is captured exactly even by single-poly -> a
// degenerate tie).  Per cell only the LOCALLY-ACTIVE branches are passed to the
// multi rule (keeps npsi small; avoids the bpsi[16] base-splitter cap).
//
//   nvcc -O2 -DUSE_DOUBLE -I src/common -I src/fem src/fem/SayeSurf3d.cu -o /tmp/surf3d && /tmp/surf3d
#include <cstdio>
#include <cmath>
#include "SayeQuad.h"
#include "PolyFit.h"

static const char* NAMES[6] = {"lens","band","wedge","corner","roundcube","trisphere"};
static const int NGEOM = 6;

// generic rotation (no axis alignment) so creases fall off the grid
static inline void rotc(double X,double Y,double Z,double&x,double&y,double&z){
  double ca=cos(0.4),sa=sin(0.4), x1=ca*X-sa*Y, y1=sa*X+ca*Y, z1=Z;
  double cb=cos(0.3),sb=sin(0.3); x=cb*x1+sb*z1; y=y1; z=-sb*x1+cb*z1;
}
static int branches(int g,double X,double Y,double Z,double* b){
  double x,y,z; rotc(X,Y,Z,x,y,z);
  auto S=[&](double cx,double cy,double cz,double R){ return (x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz)-R*R; };
  switch(g){
    case 0: b[0]=S(0.5,0,0,0.72); b[1]=S(-0.5,0,0,0.72); return 2;                 // lens
    case 1: b[0]=S(0,0,0,0.8); b[1]=-0.32-z; b[2]=z-0.32; return 3;                 // band
    case 2: b[0]=S(0,0,0,0.8); b[1]=0.12-x; b[2]=0.12-(0.5*x+0.866*y); return 3;    // wedge (dihedral)
    case 3: b[0]=S(0,0,0,0.85); b[1]=-0.12-x; b[2]=-0.12-y; b[3]=-0.12-z; return 4; // corner
    case 4: { double a=0.5; b[0]=x-a; b[1]=-a-x; b[2]=y-a; b[3]=-a-y; b[4]=z-a; b[5]=-a-z;
              b[6]=S(0,0,0,0.72); return 7; }                                       // rounded cube
    case 5: b[0]=S(0.5,0,0,0.8); b[1]=S(-0.25,0.4330,0,0.8); b[2]=S(-0.25,-0.4330,0,0.8); return 3; // 3 spheres
  }
  return 0;
}
static double composite(int g,double X,double Y,double Z){
  double b[8]; int nb=branches(g,X,Y,Z,b); double m=b[0];
  for(int k=1;k<nb;k++) m=fmax(m,b[k]); return m;
}
static double devOf(int g,double X,double Y,double Z){          // dist from true boundary
  const double e=1e-4; double c=composite(g,X,Y,Z);
  double gx=(composite(g,X+e,Y,Z)-composite(g,X-e,Y,Z))/(2*e);
  double gy=(composite(g,X,Y+e,Z)-composite(g,X,Y-e,Z))/(2*e);
  double gz=(composite(g,X,Y,Z+e)-composite(g,X,Y,Z-e))/(2*e);
  double gm=sqrt(gx*gx+gy*gy+gz*gz); return fabs(c)/fmax(gm,1e-9);
}

int main(){
  const int N=26, p=2, n=p+1; const double lo=-1.1, h=2.2/N;
  real t[PNC]; gllNodes(p,t);
  static SayeNode arena[1<<18], obuf[1<<16];

  for(int g=0; g<NGEOM; g++){
    char fs[128], fm[128];
    snprintf(fs,128,"/tmp/surf_%s_single.txt",NAMES[g]);
    snprintf(fm,128,"/tmp/surf_%s_multi.txt",NAMES[g]);
    FILE *FS=fopen(fs,"w"), *FM=fopen(fm,"w"); long nS=0,nM=0;

    for(int cz=0;cz<N;cz++) for(int cy=0;cy<N;cy++) for(int cx=0;cx<N;cx++){
      double x0=lo+cx*h, y0=lo+cy*h, z0=lo+cz*h;
      double bmax[8]={-1e30,-1e30,-1e30,-1e30,-1e30,-1e30,-1e30,-1e30}; int nb=0;
      real vc[PNC*PNC*PNC], vb[8][PNC*PNC*PNC];
      bool anyNeg=false, anyPos=false;
      for(int k=0;k<n;k++) for(int j=0;j<n;j++) for(int i=0;i<n;i++){
        double X=x0+t[i]*h,Y=y0+t[j]*h,Z=z0+t[k]*h;
        double bb[8]; nb=branches(g,X,Y,Z,bb); double m=bb[0];
        for(int q=1;q<nb;q++) m=fmax(m,bb[q]);
        int id=i+n*(j+n*k); vc[id]=(real)m;
        for(int q=0;q<nb;q++){ vb[q][id]=(real)bb[q]; if(bb[q]>bmax[q]) bmax[q]=bb[q]; }
        if(m<0)anyNeg=true; else anyPos=true;
      }
      if(!anyNeg||!anyPos) continue;

      // SINGLE: one poly on the composite max
      { PolyND pc=fitPoly3(p,vc);
        SayeArena ar; ar.buf=arena;ar.cap=1<<18;ar.top=0;
        SayeSet o; o.p=obuf;o.n=0;o.cap=1<<16;o.ovf=false;
        sayeSurface(pc,&o,&ar);
        for(int q=0;q<o.n;q++){ double X=x0+o.p[q].x[0]*h,Y=y0+o.p[q].x[1]*h,Z=z0+o.p[q].x[2]*h;
          fprintf(FS,"%.5f %.5f %.5f %.5f\n",X,Y,Z,devOf(g,X,Y,Z)); nS++; } }
      // MULTI: fit only the LOCALLY ACTIVE branches (bmax>0), integrate the intersection
      { PolyND br[8]; int act[8], na=0;
        for(int q=0;q<nb;q++) if(bmax[q]>0){ br[na]=fitPoly3(p,vb[q]); act[na]=q; na++; }
        SayeArena ar; ar.buf=arena;ar.cap=1<<18;ar.top=0;
        SayeSet o; o.p=obuf;o.n=0;o.cap=1<<16;o.ovf=false;
        sayeSurfaceMulti(br,na,&o,&ar);
        for(int q=0;q<o.n;q++){ double X=x0+o.p[q].x[0]*h,Y=y0+o.p[q].x[1]*h,Z=z0+o.p[q].x[2]*h;
          double bb[8]; branches(g,X,Y,Z,bb); int tg=act[0]; double best=1e30;
          for(int a=0;a<na;a++){ double d=fabs(bb[act[a]]); if(d<best){best=d;tg=act[a];} }
          fprintf(FM,"%.5f %.5f %.5f %.5f %d\n",X,Y,Z,devOf(g,X,Y,Z),tg); nM++; } }
    }
    fclose(FS); fclose(FM);
    printf("%-10s single %6ld  multi %6ld\n", NAMES[g], nS, nM);
  }
  return 0;
}
