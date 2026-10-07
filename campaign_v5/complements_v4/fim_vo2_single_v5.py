#!/usr/bin/env python3
"""Bloc superieur de Table 2 (partie Fisher) : un athlete de la population recalibree (athlete 0), protocoles A et B,
grille de la campagne, sigma_P = 5 W ; FIM 5 x 5 (conditions initiales connues) sous trois observables : A_P (sigma_AP v5),
VO2 comme flux delivre phi1, VO2 comme flux de recharge M_R (A_Omax - A_O) (ces deux derniers normalises par leur moyenne,
le rapport lambda_min/lambda_max et les projections etant invariants par changement d'echelle d'un bloc unique).
Spectre normalise, lambda_min/lambda_max, direction plate, projections de grad ln CP et grad ln W'.
Meme machinerie que vo2_tlim_v5.py. Sortie : fim_vo2_single_v5.json / .log."""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vo2_tlim_v5 as V   # reutilise integrate, lncp_grad, GRAD_WP, p3, H, N_OBS ; (son bloc principal ne s'execute pas : garde ci-dessous)
import glob
I_ATHS = list(range(10))
V3 = {}
for f in sorted(glob.glob(os.path.join(HERE, "..", "camp_v3_*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if r["sigma_P"] == 5.0: V3[r["i"]] = r
def one(I_ATH):
    th = np.array(V3[I_ATH]["theta_true"]); sg = V3[I_ATH]["sigma_AP"]
    def repl_of(t, AO): return t[3] * (t[1] - AO)
    blocks = {"A_P": [], "VO2 phi1": [], "VO2 repl": []}
    for kind in ("A", "B"):
        P, t_max, sw = V.p3.power_profile(kind, th)
        T, _, _ = V.integrate(th, P, t_max, sw); grid = np.linspace(0.0, 0.999 * T, V.N_OBS)
        _, AO, AP = V.integrate(th, P, t_max, sw, grid, event=False)
        S = {k: np.zeros((V.N_OBS, 5)) for k in blocks}
        for k in range(5):
            tp, tm = th.copy(), th.copy(); tp[k] *= 1 + V.H; tm[k] *= 1 - V.H
            _, AOp, APp = V.integrate(tp, P, t_max, sw, grid, event=False); _, AOm, APm = V.integrate(tm, P, t_max, sw, grid, event=False)
            S["A_P"][:, k] = (APp - APm) / (2 * V.H)
            S["VO2 phi1"][:, k] = (V.phi1_of(tp, AOp, APp) - V.phi1_of(tm, AOm, APm)) / (2 * V.H)
            S["VO2 repl"][:, k] = (repl_of(tp, AOp) - repl_of(tm, AOm)) / (2 * V.H)
        var = {"A_P": sg[kind] ** 2, "VO2 phi1": float(np.mean(V.phi1_of(th, AO, AP))) ** 2, "VO2 repl": float(np.mean(repl_of(th, AO))) ** 2}
        for k in blocks: blocks[k].append(S[k].T @ S[k] / var[k])
        print("protocole %s : T_lim %.1f s, sigma_AP %.1f J, phi1 moyen %.0f W, recharge moyenne %.0f W" % (kind, T, sg[kind], np.sqrt(var["VO2 phi1"]), np.sqrt(var["VO2 repl"])))
    out = {}
    g = V.lncp_grad(th)
    for k, Fs in blocks.items():
        F = Fs[0] + Fs[1]; w, Vv = np.linalg.eigh(F); w = np.clip(w, 0, None); lam = w[::-1] / w[-1]; v = Vv[:, 0]
        out[k] = dict(spectrum=lam.tolist(), lam_ratio=float(lam[-1]), proj_CP=float(abs(g @ v) / np.linalg.norm(g)), proj_Wp=float(abs(V.GRAD_WP @ v) / np.linalg.norm(V.GRAD_WP)),
                      vmin=(v / v[np.argmax(np.abs(v))]).tolist(), rank=int(np.sum(lam > 1e-10)))
        print("%-10s rang %d  spectre %s  lam_min/lam_max %.2e  proj CP %.3f  W' %.3f  v_min %s" % (k, out[k]["rank"], " ".join("%.1e" % x for x in lam), out[k]["lam_ratio"], out[k]["proj_CP"], out[k]["proj_Wp"], np.round(out[k]["vmin"], 2).tolist()))
    return out
ALL = {}
for I in I_ATHS:
    print("--- athlete %d" % I); ALL[I] = one(I)
print("\n== Mediane sur %d athletes ==" % len(ALL))
for k in ALL[0]:
    lr = np.median([ALL[i][k]["lam_ratio"] for i in ALL]); pc = np.median([ALL[i][k]["proj_CP"] for i in ALL]); pw = np.median([ALL[i][k]["proj_Wp"] for i in ALL])
    rk = int(np.median([ALL[i][k]["rank"] for i in ALL])); vm = np.round(np.median([ALL[i][k]["vmin"] for i in ALL], axis=0), 2).tolist()
    print("%-10s rang %d  lam_min/lam_max %.2e  proj CP %.3f  W' %.3f  v_min %s" % (k, rk, lr, pc, pw, vm))
json.dump(dict(athletes=I_ATHS, results={str(i): ALL[i] for i in ALL}), open(os.path.join(HERE, "fim_vo2_single_v5.json"), "w"), indent=1)
print("TERMINE fim_vo2_single_v5")
