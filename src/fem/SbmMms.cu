//
// M1 gate driver: SBM MMS convergence on the analytic sphere.  All of the
// method lives in SbmSolve.h (shared with the blade run in FemMain --sbm), so
// this file is just the sweep + reporting.  See SbmSolve.h for the weak form,
// the GSBM Eq.(35) parameters, the Neumann/gap options and the scaling notes.
//
//   build: make sbm_mms
//   run:   ./sbm_mms <p> <N1> <N2> ...   env: CHI KAP TOL GM DBG NEU NOGAP JAC
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "SbmSolve.h"

int main(int argc,char**argv){
  setbuf(stdout,NULL);
  int p = argc>1?atoi(argv[1]):2;
  std::vector<int> Ns; for(int a=2;a<argc;a++) Ns.push_back(atoi(argv[a]));
  if(Ns.empty()) Ns={8,16,32};

  printf("SBM/GSBM Eq.(35) MMS (sphere)  p=%d  mu=%.2f lam=%.2f\n",p,MU,LAM);
  printf("  %4s  %10s  %12s  %6s  %8s  %8s\n","N","nDof","L2err","ord","cgIt","nFaceBC");

  const double lo3[3]={-1.0,-1.0,-1.0}, L=2.0;
  double prevE=0, prevH=0;
  for(int N:Ns){
    SbmOut o = sbmSolveOne(p,N,lo3,L);
    double ord=(prevE>0)?log(prevE/o.l2abs)/log(prevH/o.h):0.0;
    printf("  %4d  %10ld  %12.4e  %6.2f  %8d  %8d\n",N,o.nd3,o.l2rel,ord,o.iters,o.nBF);
    prevE=o.l2abs; prevH=o.h;
  }
  return 0;
}
