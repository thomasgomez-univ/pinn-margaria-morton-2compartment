import json, glob, numpy as np
P=["M_O","A_Omax","A_Pmax","M_R","eta"]
R=[json.loads(l) for f in sorted(glob.glob("camp_v3_*.jsonl")) for l in open(f)]
print("n runs:",len(R),"  sigma:",sorted(set(r["sigma_P"] for r in R)))
MED=0.6744897501960817   # mediane de |N(0,1)|

def xss(t):
    M_O,A_Om,A_Pm,M_R,eta=t
    return M_R*A_Om/(M_R*A_Om+M_O)
def CP(t): return t[0]*xss(t)*t[4]
def Wp(t): return t[2]*t[4]

for s in sorted(set(r["sigma_P"] for r in R)):
    S=[r for r in R if r["sigma_P"]==s]
    print("\n===== sigma_P = %g W   (n=%d) ====="%(s,len(S)))
    for m in ["LM","DE","PINN"]:
        print(" --",m)
        for k,p in enumerate(P):
            rat=[]
            for r in S:
                th=np.array(r["theta_true"]); e=np.array(r["est"][m])
                err=100*abs(e[k]-th[k])/abs(th[k])     # % relatif
                cr =r["crlb_pct"][k]                   # % relatif
                if cr>0: rat.append(err/cr)
            rat=np.array(rat)
            print("    %-7s mediane(|err|/sCR)=%6.3f   efficacite=%6.3f   IQR=[%.2f,%.2f]"
                  %(p,np.median(rat),np.median(rat)/MED,np.percentile(rat,25),np.percentile(rat,75)))
    # coherence SE asymptotiques LM vs CRLB
    ra=[]
    for r in S:
        th=np.array(r["theta_true"]); se=np.array(r["LM_se"]); cr=np.array(r["crlb_pct"])
        sep=100*se/np.abs(th)
        ra.append(sep/cr)
    ra=np.array(ra)
    print("  SE_LM/CRLB (mediane par parametre):", np.round(np.median(ra,axis=0),3).tolist())

# couverture: |err| < 1.96 sCR ?
print("\n===== couverture nominale 95% (|err| < 1.96 sigma_CR) =====")
for s in sorted(set(r["sigma_P"] for r in R)):
    S=[r for r in R if r["sigma_P"]==s]
    for m in ["LM","DE","PINN"]:
        cov=[]
        for k,p in enumerate(P):
            c=np.mean([100*abs(np.array(r["est"][m])[k]-np.array(r["theta_true"])[k])/abs(r["theta_true"][k])
                       < 1.96*r["crlb_pct"][k] for r in S])
            cov.append(round(100*c))
        print("  sigma=%g  %-5s"%(s,m), dict(zip(P,cov)))
