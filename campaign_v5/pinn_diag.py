import json, glob, numpy as np
P=["M_O","A_Omax","A_Pmax","M_R","eta"]
LB=np.array([600.0,50_000.0,30_000.0,0.003,0.15])
UB=np.array([3000.0,1_500_000.0,160_000.0,0.080,0.35])
R=[json.loads(l) for f in sorted(glob.glob("camp_v3_*.jsonl")) for l in open(f)]
T=np.array([r["theta_true"] for r in R])
print("population vraie (min / med / max) :")
for k,p in enumerate(P):
    print("  %-7s %12.4g %12.4g %12.4g   | bornes [%.4g , %.4g]"%(p,T[:,k].min(),np.median(T[:,k]),T[:,k].max(),LB[k],UB[k]))
print("\nmarge relative de la vraie valeur a la borne sup, eta : ",
      np.round(np.sort((UB[4]-T[:,4])/(UB[4]-LB[4]))[:6],4).tolist(), "...")

for m in ["LM","DE","PINN"]:
    E=np.array([r["est"][m] for r in R])
    frac=(E-LB)/(UB-LB)
    print("\n== %s : position dans la boite (0=LB, 1=UB) =="%m)
    for k,p in enumerate(P):
        print("   %-7s med=%.3f   %%<0.02=%3.0f   %%>0.98=%3.0f"%(p,np.median(frac[:,k]),
              100*np.mean(frac[:,k]<0.02),100*np.mean(frac[:,k]>0.98)))
# PINN eta : valeurs
E=np.array([r["est"]["PINN"] for r in R])
print("\nPINN eta : min %.4f  med %.4f  max %.4f ; %% > 0.349 : %.0f"%(E[:,4].min(),np.median(E[:,4]),E[:,4].max(),100*np.mean(E[:,4]>0.349)))
print("PINN eta vs eta vrai : correlation r = %.3f"%np.corrcoef(E[:,4],T[:,4])[0,1])
# W' du PINN : decomposition de l'erreur
Wp_t=T[:,2]*T[:,4]; Wp_e=E[:,2]*E[:,4]
print("\nW' PINN : erreur mediane %.1f %% ; erreur mediane A_Pmax %.1f %% ; erreur mediane eta %.1f %%"%(
  np.median(100*abs(Wp_e-Wp_t)/Wp_t), np.median(100*abs(E[:,2]-T[:,2])/T[:,2]), np.median(100*abs(E[:,4]-T[:,4])/T[:,4])))
# contrefactuel : PINN avec eta fixe a la vraie valeur
print("W' PINN si eta etait exact : %.1f %%"%np.median(100*abs(E[:,2]*T[:,4]-Wp_t)/Wp_t))
