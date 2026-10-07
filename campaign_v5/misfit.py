import json, glob, numpy as np, importlib.util, sys
spec=importlib.util.spec_from_file_location("p3","pipeline_v3.py"); p3=importlib.util.module_from_spec(spec)
sys.modules["p3"]=p3; spec.loader.exec_module(p3)
cfg=p3.Config()
R=[json.loads(l) for f in sorted(glob.glob("camp_v3_*.jsonl")) for l in open(f)]
S=[r for r in R if r["sigma_P"]==5.0]
rows=[]
for r in S:
    i=r["i"]; th=np.array(r["theta_true"]); sg=r["sigma_AP"]
    p3.calibrate_sigma_AP=lambda theta,kind,sigma_P,n_mc=100,seed=0,_s=sg: _s[kind]
    data=p3.gen_data(th,5.0,1000*i+50,cfg)
    # controle : le bruit reproduit-il bien celui de la campagne ?
    o={}
    for name in ["vrai","LM","DE","PINN"]:
        x = th if name=="vrai" else np.array(r["est"][name])
        ss=0.0; n=0; ok=True
        for kind,d in data["protocols"].items():
            tf=float(d["t"][-1])*(1+1e-9)
            try:
                pred,_=p3.simulate(x, d["_P"], tf, np.asarray(d["t"]), switches=d["_sw"])
            except Exception: ok=False; break
            ss+=float(np.sum((pred-np.asarray(d["A_P_obs"]))**2)); n+=len(d["t"])
        o[name]=np.sqrt(ss/n) if ok else np.nan
    o["sAP"]=np.mean(list(sg.values())); rows.append(o)
A=np.array([[o[k] for k in ["vrai","LM","DE","PINN","sAP"]] for o in rows])
print("RMSE d'ajustement sur A_P observe (J), sigma_P = 5 W, n=%d athletes"%len(A))
print("                    %10s %10s %10s %10s"%("theta vrai","LM","DE","PINN"))
print("mediane RMSE (J)    %10.1f %10.1f %10.1f %10.1f"%tuple(np.nanmedian(A[:,:4],axis=0)))
print("RMSE / sigma_AP     %10.3f %10.3f %10.3f %10.3f"%tuple(np.nanmedian(A[:,:4]/A[:,4:5],axis=0)))
print("RMSE / RMSE(vrai)   %10.3f %10.3f %10.3f %10.3f"%tuple(np.nanmedian(A[:,:4]/A[:,0:1],axis=0)))
print("\nnb d'athletes ou RMSE(methode) < RMSE(theta vrai) :")
for j,name in enumerate(["LM","DE","PINN"],start=1):
    print("   %-5s %d / %d"%(name,int(np.nansum(A[:,j]<A[:,0])),len(A)))
