#!/usr/bin/env python3
"""Rev.3 (R1 m4) : censure a 3600 s dans les predictions hors echantillon (prediction_v4.py, Table 5).
Pour chaque protocole et chaque estimateur : nombre d'athletes dont l'essai vrai, l'essai predit, ou les deux atteignent
l'horizon de 3600 s sans epuisement ; erreur mediane sur T_lim recalculee (a) telle que dans la Table 5, (b) en excluant
les paires ou l'un des deux essais est censure, (c) avec un horizon de 36 000 s. Memes theta et memes protocoles que prediction_v4.py."""
import json, glob, sys, os, numpy as np
from common_v4 import p3, HERE
p3.set_tol(1e-8)
V3 = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "camp_v3_*.jsonl"))) for l in open(f)]
V4 = {(r["i"], r["sigma_P"]): r for f in sorted(glob.glob(os.path.join(HERE, "camp_v4_[0-9].jsonl"))) for l in open(f) for r in [json.loads(l)]}
S = [r for r in V3 if r["sigma_P"] == 5.0]
def est(r, m): return np.array(V4[(r["i"], 5.0)]["est"]) if m == "PINN_v4" else np.array(r["est"][m])
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def P_const(f, c, H): return (lambda t, p=f * c: p), H, ()
def P_interm(hi, lo, on, c, H): return (lambda t, hi=hi * c, lo=lo * c, on=on: hi if (t % (2 * on)) < on else lo), H, tuple(np.arange(on, H, on))
def P_recov(c, H): return (lambda t, c=c: (1.20 if t < 120 else (0.60 if t < 300 else 1.20)) * c), H, (120.0, 300.0)
PROTOS = {"const 105%": lambda c, H: P_const(1.05, c, H), "const 125%": lambda c, H: P_const(1.25, c, H), "const 150%": lambda c, H: P_const(1.50, c, H),
          "interm 120/60 60s": lambda c, H: P_interm(1.20, 0.60, 60, c, H), "recuperation 120/60/120": P_recov}
M = ["LM", "DE", "PINN", "PINN_v4"]; out = {}
for k, mk in PROTOS.items():
    out[k] = {}
    for m in M:
        rows = []
        for r in S:
            th = np.array(r["theta_true"]); c = cp(th); x = est(r, m); row = []
            for H in (3600.0, 36000.0):
                P, tmax, sw = mk(c, H)
                _, Tt = p3.simulate(th, P, tmax, switches=sw)
                try: _, Te = p3.simulate(x, P, tmax, switches=sw)
                except Exception: Te = np.nan
                row += [Tt, Te]
            rows.append(row)
        A = np.array(rows); Tt, Te, Tt2, Te2 = A.T; H = 3600.0 - 1e-6
        ct, ce = Tt >= H, Te >= H; both, either = ct & ce, ct | ce
        e = 100 * np.abs(Te - Tt) / Tt; e2 = 100 * np.abs(Te2 - Tt2) / Tt2
        out[k][m] = dict(n=len(S), cens_true=int(ct.sum()), cens_est=int(ce.sum()), both=int(both.sum()), either=int(either.sum()),
                         med_table5=float(np.nanmedian(e)), med_excl=float(np.nanmedian(e[~either])) if (~either).any() else None,
                         med_36000=float(np.nanmedian(e2)), cens_true_36000=int((Tt2 >= 36000 - 1e-6).sum()), cens_est_36000=int((Te2 >= 36000 - 1e-6).sum()))
        print("%-26s %-8s vrai censure %2d  predit %2d  les deux %2d  l'un %2d | erreur med Table5 %.2f  hors censure %s  horizon 36000 s %.2f (censures %d/%d)"
              % (k, m, ct.sum(), ce.sum(), both.sum(), either.sum(), out[k][m]["med_table5"], "%.2f" % out[k][m]["med_excl"] if out[k][m]["med_excl"] is not None else "-",
                 out[k][m]["med_36000"], out[k][m]["cens_true_36000"], out[k][m]["cens_est_36000"]), flush=True)
json.dump(out, open(os.path.join(HERE, "rev3_censoring.json"), "w"), indent=1)
