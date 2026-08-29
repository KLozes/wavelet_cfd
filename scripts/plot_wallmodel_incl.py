# Rotation-invariance gate for the route-3 (filtered-traction) wall model:
# the 30-deg inclined immersed plate vs the grid-aligned Brinkman plate vs the
# grid-aligned case-13 reference.  All extraction heights measured from the
# TRUE wall (aligned: ibPlane = 6.592e-3; inclined: the exact level-set line),
# wall-function Cf evaluated at the in-solver matching height Lp = 3 h_fine.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
S="/tmp/claude-1000/-home-kennyl-Documents-wavelet-cfd/858562fb-c18e-47a7-ace5-829067796ea9/scratchpad"
h=(1.5/256)/4; Lp=3*h; YW=4.5*(1.5/256)/4; NU=1e-6; KAP=0.41; B=5.2
A=np.deg2rad(30.); T=np.array([np.cos(A),np.sin(A)]); N=np.array([-np.sin(A),np.cos(A)])
P0=np.array([0.,0.1643-0.25*np.tan(A)])
def utau_wf(u,d,nu):
    ut=max(np.sqrt(nu*abs(u)/d),1e-12)
    for _ in range(60):
        yp=d*ut/nu
        up=yp if yp<11.0 else np.log(max(yp,1e-12))/KAP+B
        g=up+(0 if yp<11.0 else 1.0/KAP)
        ut-=(ut*up-abs(u))/max(g,1e-12); ut=max(ut,1e-12)
    return ut
ref=np.loadtxt(f"{S}/ga.dat"); ali=np.loadtxt(f"{S}/wm/final3.dat"); inc=np.loadtxt(f"{S}/wm/incl30_dp.dat")
def col_prof(d,xs,yw):
    x,y,u=d[:,0],d[:,1],d[:,2]
    m=np.abs(x-xs)<0.004; yb=np.unique(np.round(y[m],9))
    uu=np.array([u[m][np.abs(y[m]-t)<1e-9].mean() for t in yb])
    k=yb>yw+1e-9; return yb[k]-yw, uu[k]
xg,yg,ug,vg=inc[:,0],inc[:,1],inc[:,2],inc[:,3]
def incl_prof(s):
    W=P0+s*T; dg=np.linspace(0.35*h,0.05,300)
    pts=W[None,:]+dg[:,None]*N[None,:]
    box=(np.abs(xg-W[0])<0.08)&(np.abs(yg-W[1])<0.08)
    ui=griddata(np.c_[xg[box],yg[box]],ug[box],pts,method='linear')
    vi=griddata(np.c_[xg[box],yg[box]],vg[box],pts,method='linear')
    ut=ui*T[0]+vi*T[1]; ok=np.isfinite(ut); return dg[ok],ut[ok]
fig,axs=plt.subplots(2,2,figsize=(12.5,9.5)); (a,b),(c,dd)=axs
# (a) profiles
cols=plt.cm.viridis(np.linspace(0.15,0.85,4))
for i,s in enumerate((0.4,0.7,1.0,1.3)):
    dr,ur=col_prof(ref,s,0.0); da,ua=col_prof(ali,s,YW); di,ui=incl_prof(s)
    a.plot(ur,dr,'-',color='0.6',lw=3,alpha=0.6,label='case 13 (grid-aligned ref)' if i==0 else None)
    a.plot(ua,da,'-',color=cols[i],lw=1.6,label=f'aligned Brinkman' if i==0 else None)
    a.plot(ui,di,'--',color=cols[i],lw=1.8,label='inclined 30$^\\circ$' if i==0 else None)
    a.annotate(f'x={s}',xy=(0.99,0.0015+0.006*i),fontsize=8,color=cols[i])
a.set_xlim(0.3,1.12); a.set_ylim(0,0.03); a.set_xlabel('$u_t/U_\\infty$'); a.set_ylabel('wall distance d')
a.set_title('(a) profiles at four stations (route 3, $t=4$)'); a.legend(fontsize=8,loc='upper left')
# (b) Cf(s)
def cf_curve(prof_fn,ss):
    out=[]
    for s in ss:
        dv,uv=prof_fn(s); out.append(2*utau_wf(np.interp(Lp,dv,uv),Lp,NU)**2)
    return np.array(out)
ss=np.arange(0.3,1.45,0.05)
b.plot(ss,cf_curve(lambda s: col_prof(ref,s,0.0),ss),'-',color='0.5',lw=3,alpha=0.7,label='case 13 ref')
b.plot(ss,cf_curve(lambda s: col_prof(ali,s,YW),ss),'-',color='tab:blue',lw=1.8,label='aligned Brinkman')
b.plot(ss,cf_curve(incl_prof,ss),'--',color='tab:green',lw=2,label='inclined 30$^\\circ$')
b.plot(ss,0.0592*(1e6*ss)**-0.2,':',color='0.3',lw=1,label='$0.0592\\,Re_x^{-1/5}$')
b.set_xlabel('arc length s from inflow'); b.set_ylabel('$C_f$ (wall fn at $3h$)')
b.set_title('(b) skin friction: rotation invariance to ~5%'); b.legend(fontsize=8); b.set_ylim(0.002,0.007)
# (c) d95
def d95_curve(prof_fn,ss):
    out=[]
    for s in ss:
        dv,uv=prof_fn(s); out.append(np.interp(0.95,uv,dv) if uv.max()>=0.95 else np.nan)
    return np.array(out)
c.plot(ss,d95_curve(lambda s: col_prof(ref,s,0.0),ss),'-',color='0.5',lw=3,alpha=0.7,label='case 13 ref')
c.plot(ss,d95_curve(lambda s: col_prof(ali,s,YW),ss),'-',color='tab:blue',lw=1.8,label='aligned')
c.plot(ss,d95_curve(incl_prof,ss),'--',color='tab:green',lw=2,label='inclined 30$^\\circ$')
c.set_xlabel('arc length s'); c.set_ylabel('$\\delta_{95}$')
c.set_title('(c) layer growth (inclined thinner aft: staircase drag)'); c.legend(fontsize=8)
# (d) near-wall u_t field, inclined
utf=ug*T[0]+vg*T[1]
sNg=(xg-P0[0])*N[0]+(yg-P0[1])*N[1]; sTg=(xg-P0[0])*T[0]+(yg-P0[1])*T[1]
m=(sNg>-0.01)&(sNg<0.04)&(sTg>0.0)&(sTg<1.5)
sc=dd.scatter(sTg[m],sNg[m],c=np.clip(utf[m],0,1.05),s=1.2,cmap='RdYlBu_r',rasterized=True)
dd.axhline(0,color='k',lw=0.8); plt.colorbar(sc,ax=dd,label='$u_t/U_\\infty$')
dd.set_xlabel('arc length s'); dd.set_ylabel('wall-normal distance'); dd.set_ylim(-0.01,0.04)
dd.set_title('(d) inclined run in wall coordinates: the layer follows the plate')
fig.suptitle('Route-3 traction wall model: 30$^\\circ$ inclined immersed plate vs grid-aligned (Re$_L$=10$^6$, Ma 0.2)',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig('output/wallmodel_incl.png',dpi=150)
print("wrote output/wallmodel_incl.png")
