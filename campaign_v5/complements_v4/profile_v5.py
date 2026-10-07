#!/usr/bin/env python3
"""M-2 : profils de vraisemblance de A_Omax et de M_R (Raue et al., 2009) sur 3 athletes, sigma_P = 5 W, donnees de la campagne.
Pour chaque valeur fixee du parametre (grille logarithmique sur +/- 4 sigma_CR autour de l'estimation LM, 21 points),
les quatre autres parametres sont reoptimises (least_squares, depart chaud + 3 departs aleatoires). Chi2 = somme des residus
normalises au carre ; intervalle de profil a 95 % : delta chi2 <= 3,84 ; comparaison a +/- 1,96 sigma_CR.
Usage : python3 profile_v5.py [i_ath ...] -> profile_v5.json, fig_profile_v5.pdf/png (dans ../../figures)."""
import json, os, sys, numpy as np
from scipy.optimize import least_squares
from common_v4 import p3, load_v3, data_like_campaign, HERE
SIG, NPT, SPAN = 5.0, 21, 4.0
ATH = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
NAMES = ["M_O", "A_Omax", "A_Pmax", "M_R", "eta"]; PROF = [1, 3]
def chi2(theta, data): r = p3.residuals(np.asarray(theta, float), data); return float(np.sum(r ** 2))
def fit_free(data, free, fixed_k, fixed_v, x0s):
    best = None
    for x0 in x0s:
        def res(z):
            th = np.empty(5); th[free] = z; th[fixed_k] = fixed_v; return p3.residuals(th, data)
        try:
            r = least_squares(res, x0[free], bounds=(p3.PARAM_LB[free], p3.PARAM_UB[free]), method="trf", max_nfev=300)
        except Exception: continue
        if best is None or r.cost < best.cost: best = r
    th = np.empty(5); th[free] = best.x; th[fixed_k] = fixed_v; return th, 2 * best.cost
V3 = load_v3(); out = {}
for i in ATH:
    th_true, data, sg = data_like_campaign(V3, i, SIG); rec = V3[(i, SIG)]
    x_lm = np.array(rec["est"]["LM"]); cr = np.array(rec["crlb_pct"]) / 100; chi0 = chi2(x_lm, data)
    out[i] = dict(theta_true=th_true.tolist(), x_lm=x_lm.tolist(), crlb_pct=rec["crlb_pct"], chi2_min=chi0, profiles={})
    rng = np.random.default_rng(100 + i)
    for k in PROF:
        free = np.array([j for j in range(5) if j != k])
        grid = x_lm[k] * np.exp(np.linspace(-SPAN * cr[k], SPAN * cr[k], NPT)); vals = []
        # parcours vers la droite puis vers la gauche a partir de l'optimum, depart chaud
        order = list(range(NPT // 2, NPT)) + list(range(NPT // 2 - 1, -1, -1)); last = {}
        for j in order:
            v = grid[j]; warm = last.get("R" if j >= NPT // 2 else "L", x_lm)
            x0s = [warm] + [rng.uniform(p3.PARAM_LB, p3.PARAM_UB) for _ in range(3)]
            th, c2 = fit_free(data, free, k, v, x0s); vals.append((j, float(v), c2, th.tolist())); last["R" if j >= NPT // 2 else "L"] = th
            print("ath %d %s = %.5g : chi2 - min = %.3f" % (i, NAMES[k], v, c2 - chi0), flush=True)
        vals.sort(); g = np.array([x[1] for x in vals]); d = np.array([x[2] for x in vals]) - chi0
        inside = g[d <= 3.84]
        out[i]["profiles"][NAMES[k]] = dict(grid=g.tolist(), dchi2=d.tolist(), theta=[x[3] for x in vals],
            ci95=[float(inside.min()), float(inside.max())] if len(inside) else None, cr95=[float(x_lm[k] * (1 - 1.96 * cr[k])), float(x_lm[k] * (1 + 1.96 * cr[k]))],
            open_left=bool(d[0] <= 3.84), open_right=bool(d[-1] <= 3.84))
        print("ath %d %s : IC profil 95 %% %s ; IC Cramer-Rao %s ; vrai %.5g" % (i, NAMES[k], out[i]["profiles"][NAMES[k]]["ci95"], out[i]["profiles"][NAMES[k]]["cr95"], th_true[k]))
json.dump(out, open(os.path.join(HERE, "profile_v5.json"), "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "savefig.dpi": 400})
    fig, ax = plt.subplots(2, len(ATH), figsize=(2.4 * len(ATH), 4.0), squeeze=False)
    for c, i in enumerate(ATH):
        for r_, k in enumerate(PROF):
            P = out[i]["profiles"][NAMES[k]]; g = np.array(P["grid"]); x_lm = out[i]["x_lm"][k]
            a = ax[r_, c]; a.plot(g / x_lm, P["dchi2"], color="#1b4f9c", lw=1.1)
            a.plot(g / x_lm, ((g / x_lm - 1) / (out[i]["crlb_pct"][k] / 100)) ** 2, color="#c0562a", ls="--", lw=0.9, label="quadratic (Cramér–Rao)")
            a.axhline(3.84, color="k", lw=0.5, ls=":"); a.axvline(out[i]["theta_true"][k] / x_lm, color="#2e7d4f", lw=0.8, label="true value")
            a.set_ylim(0, 20); a.set_xlabel("%s / %s$_{LM}$" % (("$A_{O,\\max}$" if k == 1 else "$M_R$"),) * 2); a.set_ylabel(r"$\Delta\chi^2$")
            a.set_title("athlete %d, %s" % (i, "$A_{O,\\max}$" if k == 1 else "$M_R$"), loc="left"); a.tick_params(direction="in", top=True, right=True)
            if r_ == 0 and c == 0: a.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.4); FIG = os.path.join(HERE, "..", "..", "figures")
    fig.savefig(os.path.join(FIG, "fig_profile_v5.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "fig_profile_v5.png"), bbox_inches="tight"); print("figure ok")
except Exception as e: print("figure non produite :", e)
print("TERMINE profile_v5")
