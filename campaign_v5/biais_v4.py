import json,glob,numpy as np
P=["M_O","A_Omax","A_Pmax","M_R","eta"]
CP=lambda t:t[0]*(t[3]*t[1]/(t[3]*t[1]+t[0]))*t[4]; Wp=lambda t:t[2]*t[4]
V4=[json.loads(l) for f in sorted(glob.glob("camp_v4_[0-9].jsonl")) for l in open(f)]
V3=[json.loads(l) for f in sorted(glob.glob("camp_v3_*.jsonl")) for l in open(f)]
by={}
for r in V4: by.setdefault(r["i"],{})[r["sigma_P"]]=r
# 1. Dependance au bruit : dispersion inter-sigma de l'estimation (meme athlete) vs CRLB
print("== 1. Dispersion inter-sigma (2,5,10) de l'estimation d'un meme athlete, en % de la vraie valeur ==")
print("   (si l'estimateur suivait le bruit, cette dispersion devrait etre comparable a la CRLB a sigma=10)")
for m in ["PINN_v4","LM"]:
    D=[]
    for i,d in by.items():
        if len(d)<3: continue
        E=np.array([ (d[s]["est"] if m=="PINN_v4" else [x for x in V3 if x["i"]==i and x["sigma_P"]==s][0]["est"]["LM"]) for s in (2.,5.,10.)])
        t=np.array(d[5.]["theta_true"]); D.append(100*(E.max(0)-E.min(0))/t)
    D=np.array(D); print("   %-8s etendue mediane [%%] :"%m, np.round(np.median(D,0),2).tolist())
cr=np.median([r["crlb_pct"] for r in V3 if r["sigma_P"]==10.],0); print("   CRLB mediane a sigma=10 [%] :",np.round(cr,2).tolist())
# 2. Biais : erreur signee mediane et fraction >0
S=np.array([(np.array(r["est"])-np.array(r["theta_true"]))/np.array(r["theta_true"]) for r in V4])
print("\n== 2. Erreur signee, 150 runs : mediane [%] ", np.round(100*np.median(S,0),2).tolist(), " fraction >0 ", np.round(np.mean(S>0,0),2).tolist())
print("   corr(dln A_Omax, dln M_R) = %.2f"%np.corrcoef(np.log(1+S[:,1]),np.log(1+S[:,3]))[0,1])
# 3. Le biais depend-il de la position de l'athlete ? (A_Omax vrai petit -> surestime ?)
T=np.array([r["theta_true"] for r in V4])
print("   corr(A_Omax vrai, erreur signee A_Omax) = %.2f ; corr(M_R vrai, erreur M_R) = %.2f"%(np.corrcoef(T[:,1],S[:,1])[0,1],np.corrcoef(T[:,3],S[:,3])[0,1]))
print("   A_Omax estime : mediane %.0f (IQR %.0f-%.0f) ; vrai : mediane %.0f (IQR %.0f-%.0f)"%(np.median([r["est"][1] for r in V4]),*np.percentile([r["est"][1] for r in V4],[25,75]),np.median(T[:,1]),*np.percentile(T[:,1],[25,75])))
# 4. Queue de RMSE/opt
R=np.array([r["rmse_fit"]/r["rmse_true"] for r in V4]); idx=np.argsort(R)[::-1][:6]
print("\n== 3. RMSE/optimum : %% de runs > 2 : %.0f ; > 3 : %.0f ; pires :"%(100*np.mean(R>2),100*np.mean(R>3)), [(V4[k]["i"],V4[k]["sigma_P"],round(R[k],1)) for k in idx])
# 5. CP : athletes battant la reference pop (13.04 %) 
for s in (2.,5.,10.):
    e=[100*abs(CP(r["est"])-CP(r["theta_true"]))/CP(r["theta_true"]) for r in V4 if r["sigma_P"]==s]
    print("   sigma=%2.0f : CP < 13.04 %% (ref pop) : %d/50 ; CP < 2 %% : %d/50 ; W' < 5.16 %% (pop+APmax) : %d/50"%(s,sum(x<13.04 for x in e),sum(x<2 for x in e),sum(100*abs(Wp(r["est"])-Wp(r["theta_true"]))/Wp(r["theta_true"])<5.16 for r in V4 if r["sigma_P"]==s)))
