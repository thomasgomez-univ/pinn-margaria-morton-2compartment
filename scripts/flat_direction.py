"""Direction la moins contrainte sous differents protocoles : le protocole a
deux intensites brise-t-il la degenerescence M_R * A_Omax ?"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
import sys, os, json
import numpy as np
from protocol_design_lib import P, fim_for, cp_true, NAMES
from run_protocol_design import CATALOGUE

CIBLES = ["manuscrit A+B", "const 140%", "const 105+140", "140 + interm120/60"]
pop = P.gen_population(50, P.SEED_POP)
out = {}
for name in CIBLES:
    soft, ratio = [], []
    for i in range(10):
        F = fim_for(pop[i], CATALOGUE[name], 5.0, 1000*i+50)
        if F is None: continue
        w, V = np.linalg.eigh(F); o = np.argsort(w)[::-1]; V = V[:, o]
        v = V[:, -1] / np.abs(V[:, -1]).max()
        if v[3] < 0: v = -v
        soft.append(v); ratio.append(v[3]/v[1] if abs(v[1]) > 1e-12 else np.nan)
    soft = np.array(soft)
    out[name] = {"soft": np.median(soft, axis=0).tolist(),
                 "ratio_med": float(np.nanmedian(ratio)),
                 "ratio_iqr": [float(np.nanpercentile(ratio,25)), float(np.nanpercentile(ratio,75))]}
    print("%-22s  d ln M_R / d ln A_Omax = %+.2f  [%.2f ; %.2f]"
          % (name, out[name]["ratio_med"], *out[name]["ratio_iqr"]))
    print("%-22s  %s" % ("", "  ".join("%s=%+.2f" % (n, v)
                          for n, v in zip(NAMES, np.median(soft, axis=0)))))
json.dump(out, open(_os.path.join(_ROOT, "data", "flat_dir.json"), "w"), indent=1)
