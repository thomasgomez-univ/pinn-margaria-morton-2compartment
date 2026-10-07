#!/usr/bin/env python3
"""Compare le PINN v4 (H2) aux methodes de la campagne v3, sur les memes jeux de donnees."""
import json, glob, numpy as np
P = ["M_O", "A_Omax", "A_Pmax", "M_R", "eta"]
LB = np.array([600., 5e4, 3e4, 0.003, 0.15]); UB = np.array([3000., 1.5e6, 1.6e5, 0.080, 0.35])
CP = lambda t: t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
Wp = lambda t: t[2] * t[4]
V3 = [json.loads(l) for f in sorted(glob.glob("camp_v3_*.jsonl")) for l in open(f)]
V4 = [json.loads(l) for f in sorted(glob.glob("camp_v4_*.jsonl")) for l in open(f)]
print("v3 : %d runs ; v4 : %d runs" % (len(V3), len(V4)))
k0 = [r for r in V4 if r["seed_k"] == 0]
key = {(r["i"], r["sigma_P"]): r for r in k0}

def err(est, th): return 100 * np.abs(np.array(est) - np.array(th)) / np.abs(np.array(th))

print("\n== Erreur mediane (%%) sur CP et W' vrais ==")
print("%-10s" % "methode" + "".join("%9s" % ("%s s=%g" % (q, s)) for q in ("CP", "W'") for s in (2, 5, 10)))
for m in ["LM", "DE", "PINN", "PINN_v4"]:
    row = []
    for q, f in (("CP", CP), ("W'", Wp)):
        for s in (2., 5., 10.):
            if m == "PINN_v4":
                v = [100 * abs(f(r["est"]) - f(r["theta_true"])) / f(r["theta_true"]) for r in k0 if r["sigma_P"] == s]
            else:
                v = [100 * abs(f(r["est"][m]) - f(r["theta_true"])) / f(r["theta_true"]) for r in V3 if r["sigma_P"] == s]
            row.append(np.median(v) if v else np.nan)
    print("%-10s" % m + "".join("%9.2f" % x for x in row))

print("\n== Erreur mediane par parametre (%%), sigma = 5 W ==")
print("%-10s" % "methode" + "".join("%9s" % p for p in P))
for m in ["LM", "DE", "PINN", "PINN_v4", "CRLB"]:
    if m == "CRLB":
        v = np.array([r["crlb_pct"] for r in V3 if r["sigma_P"] == 5.])
    elif m == "PINN_v4":
        v = np.array([err(r["est"], r["theta_true"]) for r in k0 if r["sigma_P"] == 5.])
    else:
        v = np.array([err(r["est"][m], r["theta_true"]) for r in V3 if r["sigma_P"] == 5.])
    if len(v): print("%-10s" % m + "".join("%9.2f" % x for x in np.median(v, axis=0)))

print("\n== Efficacite par athlete (mediane |err|/sigma_CR / 0.6745) et couverture 1.96 sigma_CR, sigma = 5 W ==")
MED = 0.6744897501960817
for m in ["LM", "PINN", "PINN_v4"]:
    rat, cov = [], []
    src = k0 if m == "PINN_v4" else V3
    for r in src:
        if r["sigma_P"] != 5.: continue
        e = err(r["est"] if m == "PINN_v4" else r["est"][m], r["theta_true"]); c = np.array(r["crlb_pct"])
        rat.append(e / c); cov.append(e < 1.96 * c)
    if rat:
        print("%-10s eff=%s  couv=%s" % (m, np.round(np.median(rat, axis=0) / MED, 2).tolist(), (100 * np.mean(cov, axis=0)).round().tolist()))

print("\n== Saturation des bornes (%% d'estimations a < 2 %% de la boite) ==")
for m in ["PINN", "PINN_v4"]:
    E = np.array([r["est"] if m == "PINN_v4" else r["est"][m] for r in (k0 if m == "PINN_v4" else V3)])
    if len(E):
        fr = (E - LB) / (UB - LB); print("%-10s" % m, dict(zip(P, (100 * np.mean((fr < 0.02) | (fr > 0.98), axis=0)).round().tolist())))

if V4:
    r_ = np.array([r["rmse_fit"] / r["rmse_true"] for r in k0])
    print("\n== PINN v4 : RMSE d'ajustement / RMSE(theta vrai) : mediane %.2f, IQR [%.2f, %.2f], max %.2f ==" % (
        np.median(r_), *np.percentile(r_, [25, 75]), r_.max()))
    print("   temps median %.0f s, max %.0f s" % (np.median([r["wall_s"] for r in k0]), max(r["wall_s"] for r in k0)))
G = [r for r in V4 if r["seed_k"] > 0]
if G:
    print("\n== Variance inter-graines (athletes avec 3 graines, sigma = 5) ==")
    byi = {}
    for r in V4:
        if r["sigma_P"] == 5.: byi.setdefault(r["i"], []).append(r)
    for i, rs in sorted(byi.items()):
        if len(rs) < 3: continue
        cps = [100 * abs(CP(r["est"]) - CP(r["theta_true"])) / CP(r["theta_true"]) for r in rs]
        wps = [100 * abs(Wp(r["est"]) - Wp(r["theta_true"])) / Wp(r["theta_true"]) for r in rs]
        print("  athlete %2d : CP%% %s   W'%% %s   rmse/vrai %s" % (i, np.round(cps, 2).tolist(), np.round(wps, 1).tolist(),
              np.round([r["rmse_fit"] / r["rmse_true"] for r in rs], 2).tolist()))
