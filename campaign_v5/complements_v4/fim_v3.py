#!/usr/bin/env python3
"""Identifiabilite pratique sur la population recalibree : spectre, direction
plate, CRLB, projections, et test de deplacement (Figure identifiabilite)."""
import json, os, sys, time, numpy as np
from common_v4 import p3, load_v3, data_like_campaign, sens_matrix, fim_summary, HERE
V3 = load_v3(); S_P = 5.0
out = {}; t0 = time.time()
for i in range(int(os.environ.get("N_ATH","50"))):
    th, data, sg = data_like_campaign(V3, i, S_P)
    S = sens_matrix(th, data); F = S.T @ S
    out[i] = fim_summary(F, th); out[i]["theta"] = th.tolist()
    print("i=%2d cond=%.3g expo=%+.2f crlb=%s (%.0f s)" % (i, out[i]["cond"], out[i]["expo"], np.round(out[i]["crlb"], 2).tolist(), time.time() - t0), flush=True)
json.dump(out, open(os.path.join(HERE, "fim_v3.json"), "w"), indent=1)
spec = np.array([out[i]["spec"] for i in out]); soft = np.array([out[i]["vmin"] for i in out])
cond = np.array([out[i]["cond"] for i in out]); expo = np.array([out[i]["expo"] for i in out])
crlb = np.array([out[i]["crlb"] for i in out])
np.savez(os.path.join(HERE, "fim_v3.npz"), spec=spec, soft=soft, cond=cond, expo=expo, crlb=crlb)
print("\nspectre median :", np.round(np.median(spec, 0), 8).tolist())
print("cond median %.3g IQR [%.3g, %.3g]" % (np.median(cond), *np.percentile(cond, [25, 75])))
print("vmin median :", np.round(np.median(soft, 0), 2).tolist())
print("expo median %.2f IQR [%.2f, %.2f]" % (np.median(expo), *np.percentile(expo, [25, 75])))
print("CRLB mediane :", np.round(np.median(crlb, 0), 2).tolist())
print("CRLB CP %.2f  W' %.2f ; proj CP %.3f  W' %.3f" % (np.median([out[i]["crlb_CP"] for i in out]), np.median([out[i]["crlb_Wp"] for i in out]),
      np.median([out[i]["proj_CP"] for i in out]), np.median([out[i]["proj_Wp"] for i in out])))

# ---- test de deplacement a la mediane de population (fig. c) ----
pop = p3.gen_population(50); thm = np.median(pop, 0)
vfit = np.median(soft, 0)
def traj(th, kind, grid=None):
    P, tmax, sw = p3.power_profile(kind, th)
    _, te = p3.simulate(th, P, tmax, switches=sw)
    return te
def disp(th_ref, th_new):
    """max sur A et B de |A_P_new - A_P_ref| / A_Pmax_ref, sur la grille de ref."""
    m = 0.0; keep = {}
    for kind in ("A", "B"):
        P, tmax, sw = p3.power_profile(kind, th_ref)          # meme entree (celle de la reference)
        _, te = p3.simulate(th_ref, P, tmax, switches=sw); g = np.linspace(0, te * 0.999, 300)
        yr, _ = p3.simulate(th_ref, P, tmax, g, switches=sw); yn, _ = p3.simulate(th_new, P, tmax, g, switches=sw)
        m = max(m, float(np.max(np.abs(yn - yr)) / th_ref[2])); keep[kind] = (g, yr / th_ref[2], yn / th_ref[2])
    return m, keep
p3.set_tol(1e-10)
sym = thm * np.array([2, 2, 1, 1, 0.5]); prod = thm * np.array([1, 2, 1, 0.5, 1])
fit = thm * np.exp(vfit * (np.log(2) / vfit[1]))          # deplacement le long de vmin tel que A_Omax x2
d_sym, k1 = disp(thm, sym); d_prod, k2 = disp(thm, prod); d_fit, k3 = disp(thm, fit)
print("deplacement max (%% A_Pmax) : sym %.1f  prod %.1f  fit %.1f" % (100 * d_sym, 100 * d_prod, 100 * d_fit))
g, ref, _ = k1["B"]
np.savez(os.path.join(HERE, "flat_traj_v3.npz"), t=g, ref=ref, sym=k1["B"][2], prod=k2["B"][2], fit=k3["B"][2],
         d_sym=d_sym, d_prod=d_prod, d_fit=d_fit, theta_med=thm, vfit=vfit)
print("TERMINE fim_v3")
