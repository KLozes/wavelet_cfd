#!/usr/bin/env python3
"""RAE 2822 wall-modelled RANS solution (case 15, --rans 1, --a7tol 0.3), t = 0.3.
NOT converged and NOT validated -- this is a snapshot of a run that is still
degrading later in time; see the Cp panel."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.path import Path
S="/tmp/claude-1000/-home-kennyl-Documents-wavelet-cfd/858562fb-c18e-47a7-ace5-829067796ea9/scratchpad/"
su=np.loadtxt(S+"sol_surf.dat"); fd=np.loadtxt(S+"sol_field.dat")
ch=float([l for l in open(S+"sol_field.dat") if "chord=" in l][0].split("chord=")[1])
xs,ys=su[:,3],su[:,4]; cx,cy=.5*(xs.min()+xs.max()),.5*(ys.min()+ys.max())
P=np.column_stack([(xs-cx)/ch,(ys-cy)/ch])
xc,cp,side=su[:,0],su[:,2],su[:,5]
fig,ax=plt.subplots(1,3,figsize=(16,4.8))
# --- Cp
for s,lab,c in ((1,"upper","C0"),(-1,"lower","C3")):
    m=side==s; o=np.argsort(xc[m]); ax[0].plot(xc[m][o],cp[m][o],c,lw=1.5,marker='.',ms=3,label=lab)
ax[0].axhline(1.06,color='k',ls=':',lw=1,label="stagnation $C_p$ (M=0.5)")
ax[0].axhline(0,color='k',lw=.4); ax[0].invert_yaxis()
ax[0].set_xlabel("x/c"); ax[0].set_ylabel("$C_p$"); ax[0].legend(fontsize=8)
ax[0].set_title("surface $C_p$ — spikes are NOT physical"); ax[0].grid(alpha=.3)
# --- Mach + pressure fields
x,y=fd[:,0],fd[:,1]
w=(np.abs(x)<0.9)&(np.abs(y)<0.5)
tri=Triangulation(x[w],y[w])
cxs=x[w][tri.triangles].mean(1); cys=y[w][tri.triangles].mean(1)
tri.set_mask(Path(P).contains_points(np.column_stack([cxs,cys])))
for k,(v,lab,cm,lv) in enumerate(((fd[:,6],"Mach","turbo",np.linspace(0,1.2,25)),
                                  (fd[:,7],"$C_p$","RdBu_r",np.linspace(-2,1.2,25)))):
    a=ax[k+1]
    cf=a.tricontourf(tri,v[w],levels=lv,cmap=cm,extend="both")
    a.plot(P[:,0],P[:,1],"k-",lw=1.0); plt.colorbar(cf,ax=a,label=lab)
    a.set_aspect("equal"); a.set_xlim(-0.7,0.9); a.set_ylim(-0.35,0.35)
    a.set_xlabel("x/c"); a.set_title(f"{lab} field (finest-level cells)")
fig.suptitle("RAE 2822, M=0.5, $\\alpha$=2.31$^\\circ$, Re=6.5e6, k~-tau~ SST wall model, t=0.3 (UNCONVERGED)",
             fontsize=10)
fig.tight_layout(); fig.savefig("output/rae_rans_solution.png",dpi=130)
print("wrote output/rae_rans_solution.png")
u=side>0; l=side<0
print(f"  Cp upper: [{cp[u].min():.2f}, {cp[u].max():.2f}]   lower: [{cp[l].min():.2f}, {cp[l].max():.2f}]")
print(f"  |Cp|>1.5 on {(np.abs(cp)>1.5).sum()}/{len(cp)} surface points")
