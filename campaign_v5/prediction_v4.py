"""Capacite de prediction hors echantillon des quatre estimateurs : on simule,
avec le theta estime, des protocoles NON utilises pour l'ajustement, et on
compare au theta vrai. sigma_P = 5 W, 50 athletes."""
import json, glob, sys, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("p3", "pipeline_v3.py"); p3 = importlib.util.module_from_spec(spec); sys.modules["p3"] = p3; spec.loader.exec_module(p3)
p3.set_tol(1e-8)
V3 = [json.loads(l) for f in sorted(glob.glob("camp_v3_*.jsonl")) for l in open(f)]
V4 = {(r["i"], r["sigma_P"]): r for f in sorted(glob.glob("camp_v4_[0-9].jsonl")) for l in open(f) for r in [json.loads(l)]}
S = [r for r in V3 if r["sigma_P"] == 5.0]
def est(r, m): return np.array(V4[(r["i"], 5.0)]["est"]) if m == "PINN_v4" else np.array(r["est"][m])
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
# protocoles de prediction, intensites en fraction de la CP VRAIE (meme entree pour tous)
def P_const(f, c): return (lambda t, p=f * c: p), 3600.0, ()
def P_interm(hi, lo, on, c): return (lambda t, hi=hi * c, lo=lo * c, on=on: hi if (t % (2 * on)) < on else lo), 3600.0, tuple(np.arange(on, 3600.0, on))
def P_recov(c):   # 120 s a 120 %, 180 s a 60 % (recuperation), puis 120 % jusqu'a epuisement
    f = lambda t, c=c: (1.20 if t < 120 else (0.60 if t < 300 else 1.20)) * c
    return f, 3600.0, (120.0, 300.0)
PROTOS = {"const 105%": lambda c: P_const(1.05, c), "const 125%": lambda c: P_const(1.25, c), "const 150%": lambda c: P_const(1.50, c),
          "interm 120/60 60s": lambda c: P_interm(1.20, 0.60, 60, c), "recuperation 120/60/120": P_recov}
res = {k: {m: {"T": [], "traj": []} for m in ["LM", "DE", "PINN", "PINN_v4"]} for k in PROTOS}
for r in S:
    th = np.array(r["theta_true"]); c = cp(th)
    for k, mk in PROTOS.items():
        P, tmax, sw = mk(c)
        _, Tt = p3.simulate(th, P, tmax, switches=sw)
        g = np.linspace(0, Tt * 0.999, 200); yt, _ = p3.simulate(th, P, tmax, g, switches=sw)
        for m in res[k]:
            x = est(r, m)
            try:
                _, Te = p3.simulate(x, P, tmax, switches=sw); ye, _ = p3.simulate(x, P, tmax, g, switches=sw)
                res[k][m]["T"].append(100 * abs(Te - Tt) / Tt); res[k][m]["traj"].append(100 * np.max(np.abs(ye - yt)) / th[2])
            except Exception:
                res[k][m]["T"].append(np.nan); res[k][m]["traj"].append(np.nan)
print("Erreur mediane [IQR] sur le temps d'epuisement predit (%), puis ecart max de trajectoire A_P (% de A_Pmax), 50 athletes, sigma_P = 5 W")
print("%-26s" % "protocole de prediction" + "".join("%22s" % m for m in ["LM", "DE", "PINN", "PINN_v4"]))
for k in PROTOS:
    print("%-26s" % (k + "  T_lim") + "".join("%22s" % ("%.2f [%.2f-%.2f]" % (np.nanmedian(v["T"]), *np.nanpercentile(v["T"], [25, 75]))) for v in res[k].values()))
    print("%-26s" % ("   traj") + "".join("%22s" % ("%.2f [%.2f-%.2f]" % (np.nanmedian(v["traj"]), *np.nanpercentile(v["traj"], [25, 75]))) for v in res[k].values()))
json.dump({k: {m: {q: [float(x) for x in v[q]] for q in v} for m, v in d.items()} for k, d in res.items()}, open("prediction_v4.json", "w"))
