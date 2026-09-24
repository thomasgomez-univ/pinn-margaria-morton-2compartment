"""Spectre FIM complet par athlete -> npz (pour figures)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
import numpy as np
from mm2c import pipeline as P
P.set_tol(1e-6)
cfg = P.Config(noise_mode="observation")
pop = P.gen_population(50, P.SEED_POP)
spec, cond, soft, ratio, crlb = [], [], [], [], []
for i in range(50):
    th = pop[i]
    data = P.gen_data(th, 5.0, 1000*i+50, cfg)
    S = []
    for kind, d in data["protocols"].items():
        g = np.asarray(d["t"]); sA = d["sigma_AP"]; blk = np.zeros((len(g),5))
        for k in range(5):
            tp, tm = th.copy(), th.copy(); h = 1e-6*abs(th[k]); tp[k]+=h; tm[k]-=h
            yp,_ = P.simulate(tp, d["_P"], d["_t_max"], g, switches=d["_sw"])
            ym,_ = P.simulate(tm, d["_P"], d["_t_max"], g, switches=d["_sw"])
            blk[:,k] = (yp-ym)/(2*h)*th[k]/sA
        S.append(blk)
    F = np.vstack(S); F = F.T@F
    w,V = np.linalg.eigh(F); o = np.argsort(w)[::-1]; w,V = w[o],V[:,o]
    spec.append(w/w[0]); cond.append(w[0]/w[-1])
    v = V[:,-1]/np.abs(V[:,-1]).max()
    if v[3] < 0: v = -v            # convention : M_R positif
    soft.append(v); ratio.append(v[3]/v[1])
    crlb.append(100*np.sqrt(np.diag(np.linalg.inv(F))))
np.savez(_os.path.join(_ROOT, "data", "fim_full.npz"), spec=np.array(spec), cond=np.array(cond),
         soft=np.array(soft), ratio=np.array(ratio), crlb=np.array(crlb))
print("ok", np.array(spec).shape)
