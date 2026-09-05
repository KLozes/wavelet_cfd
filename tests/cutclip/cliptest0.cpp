#include "CutClip.h"
#include <cstdio>
#include <vector>
using namespace cutclip;
int main(){
  Box U; U.x0=0;U.y0=0;U.x1=1;U.y1=1;U.dx=1;U.dy=1;U.P=4; int fails=0;
  auto shoelace=[](const std::vector<double>&P){int n=P.size()/2;double a=0;for(int e=0;e<n;e++){int f=(e+1==n)?0:e+1;a+=P[2*e]*P[2*f+1]-P[2*f]*P[2*e+1];}return 0.5*a;};
  auto show=[&](const char*n,const std::vector<double>&P,double expA,int expL){ ClipResult R; const bool ccw=shoelace(P)>0; clipCell(U,P.data(),P.size()/2,!ccw,R);   // solid inside: fluid-left forward iff CW
    double a=0; for(int l=0;l<R.nLoop;l++)a+=R.loop[l].area; bool ok=!R.bad&&!R.overflow&&R.nLoop==expL&&fabs(a-expA)<1e-14;
    double il=0; for(int l=0;l<R.nLoop;l++) il+=R.loop[l].intLen;
    printf("%-36s %s loops=%d area=%.15f wall0=(%+.3f,%+.3f) len0=%.3f int=%.3f\n",n,ok?"OK ":"BAD",R.nLoop,a,R.nLoop?R.loop[0].wallVx:0,R.nLoop?R.loop[0].wallVy:0,R.nLoop?R.loop[0].wallLen:0,il); if(!ok)fails++; };
  show("slit through the cell, y=0.3",{-1,0.3, 2,0.3, 2,0.3, -1,0.3},1.0,2);
  show("slit tip inside the cell",{-1,0.3, 0.6,0.3, 0.6,0.3, -1,0.3},1.0,2);   // tip extended to x=1: two pieces, internal face 0.4 (x2)
  show("inclined slit through",{-1,0.1, 2,0.9, 2,0.9, -1,0.1},1.0,2);
  show("inclined slit tip inside",{-1,0.1, 0.5,0.5, 0.5,0.5, -1,0.1},1.0,2);
  show("slit along the low-y face",{-1,0.0, 2,0.0, 2,0.0, -1,0.0},1.0,2);   // the cell + an empty loop below the face
  show("slit through a corner diag",{-1,-1, 2,2, 2,2, -1,-1},1.0,2);
  show("slit TIP exactly on the low-x face",{0.0,0.4, 2,0.4, 2,0.4, 0.0,0.4},1.0,2);      // the plate LE on a grid line: two pieces
  show("slit tip on the face, inclined",{0.0,0.4, 2,0.7, 2,0.7, 0.0,0.4},1.0,2);
  show("vertex on the top face, wedge inside",{0.5,1.0, 2,0.2, 2,3, 0.4,3},1.0-0.5*0.5*(0.8/3.0),1);       // body vertex on a face, chain passes through
  printf("FAILS=%d\n",fails); return fails; }
