#include "CutClip.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace cutclip;
static bool pip(const std::vector<double>&P, double x, double y){ // point in polygon (crossing)
  int n=P.size()/2; bool in=false;
  for(int e=0;e<n;e++){int f=(e+1==n)?0:e+1; double ax=P[2*e],ay=P[2*e+1],bx=P[2*f],by=P[2*f+1];
    if((ay>y)!=(by>y)){double xi=ax+(y-ay)/(by-ay)*(bx-ax); if(x<xi) in=!in;}}
  return in;}
static double rasterArea(const Box&B,const std::vector<double>&P,bool solidInside,int N=1500){
  long cnt=0; for(int j=0;j<N;j++)for(int i=0;i<N;i++){double x=B.x0+(i+0.5)/N*B.dx,y=B.y0+(j+0.5)/N*B.dy;
    bool in=pip(P,x,y); if(in!=solidInside) cnt++;} return (double)cnt/((double)N*N)*B.dx*B.dy;}
static double shoelace(const std::vector<double>&P){int n=P.size()/2;double a=0;for(int e=0;e<n;e++){int f=(e+1==n)?0:e+1;a+=P[2*e]*P[2*f+1]-P[2*f]*P[2*e+1];}return 0.5*a;}
static int fails=0;
static void run(const char*name,const Box&B,const std::vector<double>&P,bool solidInside,
                double expArea=-1,double expAX=-1,double expAXhi=-1,double expAY=-1,double expAYhi=-1){
  bool ccw=shoelace(P)>0; bool fwd=(ccw==!solidInside);
  ClipResult R; clipCell(B,P.data(),P.size()/2,fwd,R);
  double area=0,fl[4]={0,0,0,0},wvx=0,wvy=0; double gcl=0;
  for(int l=0;l<R.nLoop;l++){const ClipLoop&L=R.loop[l]; area+=L.area; for(int f=0;f<4;f++) fl[f]+=L.faceLen[f];
    wvx+=L.wallVx; wvy+=L.wallVy;
    // per-loop GCL: wall + faces = 0 : faces outward normals: low-x (-1,0)*len, high-x (+1,0), low-y (0,-1), high-y (0,1)
    double gx=L.wallVx - L.faceLen[0] + L.faceLen[1], gy=L.wallVy - L.faceLen[2] + L.faceLen[3];
    gcl=fmax(gcl,fabs(gx)+fabs(gy));}
  if(R.nLoop==0){ area = R.nHole ? B.dx*B.dy-R.holeArea : -1; }
  double ref=rasterArea(B,P,solidInside);
  bool ok=!R.bad&&!R.overflow&&gcl<1e-12&&(R.nLoop==0||fabs(area-ref)<2e-3*B.dx*B.dy);
  if(expArea>=0) ok&=fabs(area-expArea)<1e-12;
  if(expAX>=0)   ok&=fabs(fl[0]/B.dy-expAX)<1e-12;
  if(expAXhi>=0) ok&=fabs(fl[1]/B.dy-expAXhi)<1e-12;
  if(expAY>=0)   ok&=fabs(fl[2]/B.dx-expAY)<1e-12;
  if(expAYhi>=0) ok&=fabs(fl[3]/B.dx-expAYhi)<1e-12;
  if(!ok) fails++;
  printf("%-34s %s loops=%d holes=%d area=%.12f raster=%.5f ap(lo-x,hi-x,lo-y,hi-y)=(%.6f %.6f %.6f %.6f) wall=(%+.6f %+.6f) gcl=%.1e%s%s\n",
    name, ok?"OK ":"BAD", R.nLoop,R.nHole,area,ref,fl[0]/B.dy,fl[1]/B.dy,fl[2]/B.dx,fl[3]/B.dx,wvx,wvy,gcl,R.bad?" BAD-TOPO":"",R.overflow?" OVF":"");
}
int main(){
  Box U; U.x0=0;U.y0=0;U.x1=1;U.y1=1;U.dx=1;U.dy=1;U.P=4;
  // a: half-plane x>0.5 solid (CCW square)
  run("half cut x>0.5 solid",U,{0.5,-1, 2,-1, 2,2, 0.5,2},true, 0.5, 1,0,0.5,0.5);
  // same body CW
  run("half cut, CW polyline",U,{0.5,-1, 0.5,2, 2,2, 2,-1},true, 0.5, 1,0,0.5,0.5);
  // b: thin plate y in [0.5,0.5+1e-9]
  run("thin plate 1e-9",U,{-1,0.5, 2,0.5, 2,0.5+1e-9, -1,0.5+1e-9},true, 1-1e-9, 1-1e-9,1-1e-9,1,1);
  // c: plate thickness 1e-15
  run("thin plate 1e-15",U,{-1,0.5, 2,0.5, 2,0.5+1e-15, -1,0.5+1e-15},true);
  // d: vertex exactly on the top face (apex at (0.5,1)), body enters through the right face
  run("apex on top face",U,{0.5,1.0, 3,-2, 3,1.5},true);
  // d2: vertex exactly at a corner
  run("vertex at corner (1,1)",U,{1.0,1.0, 3,-2, 3,1.5},true);
  // d3: vertex exactly at corner, body touching only at the corner (should be uncut)
  run("touch corner only",U,{1.0,1.0, 3,1.0, 3,3},true);
  // e: hole
  run("hole 0.2x0.2",U,{0.4,0.4, 0.6,0.4, 0.6,0.6, 0.4,0.6},true, 0.96, 1,1,1,1);
  // f: segment along the low-x face: solid x<0 with the wall exactly on x=0
  run("wall on low-x face, solid x<0",U,{0,-1, 0,2, -2,2, -2,-1},true, 1.0, 0,1,1,1);
  run("wall on low-x face, solid x>0",U,{0,-1, 2,-1, 2,2, 0,2},true, 0.0);
  // g: diagonal cut through two corners exactly
  run("diagonal through corners",U,{0,0, 1,1, 1,3, -3,3},true, 0.5);
  // h: fluid inside a loop (duct): CCW square [0.25,0.75]^2 bounds the fluid
  run("fluid inside loop",U,{0.25,0.25, 0.75,0.25, 0.75,0.75, 0.25,0.75},false, 0.25,0,0,0,0);
  // i: a body crossing the cell twice (U-shape) -> split cell
  run("U body: split into 2 loops",U,{0.3,-1, 0.4,-1, 0.4,0.8, 0.6,0.8, 0.6,-1, 0.7,-1, 0.7,2, 0.3,2},true);
  // j: circle through cell, cell corner exactly on the circle
  { std::vector<double> C; int N=4096; double R=0.5, cx=1.0, cy=0.5; for(int k=0;k<N;k++){double t=2*M_PI*k/N; C.push_back(cx+R*cos(t)); C.push_back(cy+R*sin(t));}
    run("circle 4096, cx on the right face",U,C,true); }
  // k: random sweep: random convex polygons (CCW) vs unit cell, raster check + GCL
  srand(7); int nb=0;
  for(int t=0;t<300;t++){ int nv=3+rand()%6; std::vector<double> ang(nv); for(auto&a:ang)a=2*M_PI*rand()/RAND_MAX; std::sort(ang.begin(),ang.end());
    double cx=-0.5+2.0*rand()/RAND_MAX, cy=-0.5+2.0*rand()/RAND_MAX, r=0.1+1.0*rand()/RAND_MAX; std::vector<double> P;
    for(double a:ang){P.push_back(cx+r*cos(a));P.push_back(cy+r*sin(a));}
    bool ccw=shoelace(P)>0; bool fwd=(ccw==false); ClipResult R; clipCell(U,P.data(),nv,fwd,R);
    double area=0,gcl=0; for(int l=0;l<R.nLoop;l++){const ClipLoop&L=R.loop[l];area+=L.area; gcl=fmax(gcl,fabs(L.wallVx-L.faceLen[0]+L.faceLen[1])+fabs(L.wallVy-L.faceLen[2]+L.faceLen[3]));}
    if(R.nLoop==0) area = R.nHole? 1-R.holeArea : (pip(P,0.5,0.5)?0:1);
    double ref=rasterArea(U,P,true,700);
    if(R.bad||R.overflow||gcl>1e-12||fabs(area-ref)>3e-3){nb++; if(nb<=5)printf("  random %d: bad=%d ovf=%d gcl=%.1e area=%.6f ref=%.6f nloop=%d\n",t,R.bad,R.overflow,gcl,area,ref,R.nLoop);}
  }
  printf("random sweep: %d/300 bad\n",nb); fails+=nb>0;
  printf("FAILS = %d\n",fails); return fails;
}
// (appended) zero-thickness slit cases are exercised by cliptest0 below
