#!/usr/bin/env python3
"""Reconstruit profile_v5.json a partir des logs profile_{0,1,2}.log (les trois processus paralleles ont ecrit le meme
fichier JSON ; seul le dernier athlete y restait) et trace la figure. Les valeurs sont celles imprimees par profile_v5.py
(grille a 5 chiffres significatifs, delta chi2 a 1e-3)."""
import re, json, os, numpy as np
from common_v4 import load_v3, HERE
V3 = load_v3(); NAMES = {"A_Omax": 1, "M_R": 3}; out = {}
for i in (0, 1, 2):
    rec = V3[(i, 5.0)]; x_lm = np.array(rec["est"]["LM"]); cr = np.array(rec["crlb_pct"]) / 100
    pts = {"A_Omax": [], "M_R": []}
    for l in open(os.path.join(HERE, "profile_%d.log" % i)):
        m = re.match(r"ath (\d+) (A_Omax|M_R) = (\S+) : chi2 - min = (\S+)", l)
        if m: pts[m.group(2)].append((float(m.group(3)), float(m.group(4))))
    out[i] = dict(theta_true=rec["theta_true"], x_lm=x_lm.tolist(), crlb_pct=rec["crlb_pct"], profiles={})
    for name, k in NAMES.items():
        p = sorted(pts[name]); g = np.array([a for a, b in p]); d = np.array([b for a, b in p])
        inside = g[d <= 3.84]
        # interpolation lineaire du seuil 3,84 de part et d'autre du minimum
        jmin = int(np.argmin(d)); lo = hi = None
        for j in range(jmin, 0, -1):
            if d[j - 1] > 3.84 >= d[j]: lo = g[j - 1] + (3.84 - d[j - 1]) * (g[j] - g[j - 1]) / (d[j] - d[j - 1]); break
        for j in range(jmin, len(g) - 1):
            if d[j] <= 3.84 < d[j + 1]: hi = g[j] + (3.84 - d[j]) * (g[j + 1] - g[j]) / (d[j + 1] - d[j]); break
        half_prof = (np.log(hi) - np.log(lo)) / 2 if lo and hi else None
        out[i]["profiles"][name] = dict(grid=g.tolist(), dchi2=d.tolist(), n=len(g), dchi2_min=float(d.min()),
            ci95_interp=[lo, hi], ci95_halfwidth_pct=100 * half_prof if half_prof else None, cr95_halfwidth_pct=196 * cr[k],
            ratio=half_prof / (1.96 * cr[k]) if half_prof else None, asym=((np.log(hi) - np.log(x_lm[k])) / (np.log(x_lm[k]) - np.log(lo))) if lo and hi else None,
            dchi2_edges=[float(d[0]), float(d[-1])], true_inside=bool(lo <= rec["theta_true"][k] <= hi) if lo and hi else None)
        P = out[i]["profiles"][name]
        print("ath %d %-6s n=%d  min dchi2 %.3f  demi-largeur profil %.2f %%  Cramer-Rao %.2f %%  rapport %.2f  asymetrie (droite/gauche) %.2f  bords %.1f / %.1f  vrai dedans %s" % (
            i, name, P["n"], P["dchi2_min"], P["ci95_halfwidth_pct"], P["cr95_halfwidth_pct"], P["ratio"], P["asym"], *P["dchi2_edges"], P["true_inside"]))
json.dump(out, open(os.path.join(HERE, "profile_v5.json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "legend.fontsize": 6.5, "savefig.dpi": 400, "axes.linewidth": 0.6})
fig, ax = plt.subplots(2, 3, figsize=(7.1, 4.0))
LAB = {"A_Omax": r"$A_{O,\max}$", "M_R": r"$M_R$"}
for c, i in enumerate((0, 1, 2)):
    for r_, (name, k) in enumerate(NAMES.items()):
        P = out[i]["profiles"][name]; g = np.array(P["grid"]); x0 = out[i]["x_lm"][k]; s = out[i]["crlb_pct"][k] / 100
        z = np.log(g / x0) / s
        a = ax[r_, c]; a.plot(z, P["dchi2"], "o-", color="#1b4f9c", ms=2.2, lw=1.0, label="profile likelihood")
        zz = np.linspace(z.min(), z.max(), 200); a.plot(zz, zz ** 2, color="#c0562a", ls="--", lw=0.9, label="quadratic (Fisher)")
        a.axhline(3.84, color="0.4", lw=0.5, ls=":"); a.axvline(np.log(out[i]["theta_true"][k] / x0) / s, color="#2e7d4f", lw=0.8, label="true value")
        a.set_ylim(0, 18); a.set_xlim(-4.2, 4.2)
        a.set_xlabel(r"$\ln(%s/%s^{\mathrm{LM}})\,/\,\sigma_{\mathrm{CR}}$" % (LAB[name].strip("$"), LAB[name].strip("$")))
        a.set_ylabel(r"$\Delta\chi^2$"); a.set_title("athlete %d, %s" % (i, LAB[name]), loc="left"); a.tick_params(direction="in", top=True, right=True)
        if r_ == 0 and c == 0: a.legend(frameon=False, loc="upper center")
fig.tight_layout(pad=0.4); FIG = os.path.join(HERE, "..", "..", "figures")
fig.savefig(os.path.join(FIG, "fig_profile_v5.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "fig_profile_v5.png"), bbox_inches="tight"); print("figure ok")
