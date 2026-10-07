#!/usr/bin/env python3
"""P-2 / P-4 : cinetique de l'oxygene simule (flux delivre phi1) sur la population recalibree (50 athletes).
(1) Palier a 80 % de CP, 360 s : ajustement mono-exponentiel phi1(t) = phi_ss (1 - exp(-t/tau)) ; tau compare a 1/M_R.
(2) Figure : athlete de CP mediane, phi1(t)/M_O a 60 % (modere), 85 % (lourd) et 110 % (severe, jusqu'a l'epuisement) de CP,
    reperes M_O et M_O x_ss (flux a CP), et A_P/A_Pmax. Sorties : vo2_kinetics_v5.json, ../../figures/fig_vo2_kinetics_v5.pdf/png."""
import json, os, sys, numpy as np
from scipy.optimize import curve_fit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vo2_tlim_v5 as V
pop = np.load(os.path.join(HERE, "pop_recal.npy"))[:50]
def cp_of(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def run(th, frac, tmax):
    P = lambda t, _p=frac * cp_of(th): _p
    T, _, _ = V.integrate(th, P, tmax, ())
    grid = np.linspace(0, min(T, tmax) * 0.999, 400); _, AO, AP = V.integrate(th, P, tmax, (), grid, event=True)
    return grid, V.phi1_of(th, AO, AP), AP, T
res = []
for i, th in enumerate(pop):
    t, phi, AP, T = run(th, 0.80, 360.0)
    f = lambda t, a, tau: a * (1 - np.exp(-t / tau))
    (a, tau), _ = curve_fit(f, t, phi, p0=[phi[-1], 60.0], bounds=([0, 1.0], [np.inf, 2000.0]))
    t63 = float(t[np.argmax(phi >= 0.632 * phi[-1])])
    res.append(dict(i=i, tau_fit=float(tau), t63=t63, invMR=float(1 / th[3]), phi_ss_over_MO=float(phi[-1] / th[0]), xss=float(th[3] * th[1] / (th[3] * th[1] + th[0])), reached_T=bool(T < 360.0)))
tau = np.array([r["tau_fit"] for r in res]); im = np.array([r["invMR"] for r in res]); t63 = np.array([r["t63"] for r in res])
print("palier 80 %% CP, 360 s, 50 athletes : tau mono-exp mediane %.1f s (%.0f-%.0f), t63 %.1f s (%.0f-%.0f) ; 1/M_R mediane %.1f s (%.0f-%.0f) ; r(tau, 1/M_R) = %.2f ; rapport tau/(1/M_R) %.2f-%.2f" % (
    np.median(tau), tau.min(), tau.max(), np.median(t63), t63.min(), t63.max(), np.median(im), im.min(), im.max(), np.corrcoef(tau, im)[0, 1], (tau / im).min(), (tau / im).max()))
print("athletes epuises avant 360 s a 80 %% : %d ; phi_ss/M_O mediane %.3f (0,8 x_ss attendu %.3f)" % (sum(r["reached_T"] for r in res), np.median([r["phi_ss_over_MO"] for r in res]), 0.8 * np.median([r["xss"] for r in res])))
summary = dict(tau_med=float(np.median(tau)), tau_range=[float(tau.min()), float(tau.max())], t63_med=float(np.median(t63)), invMR_med=float(np.median(im)), invMR_range=[float(im.min()), float(im.max())], r=float(np.corrcoef(tau, im)[0, 1]), ratio_range=[float((tau / im).min()), float((tau / im).max())])
json.dump(dict(summary=summary, athletes=res), open(os.path.join(HERE, "vo2_kinetics_v5.json"), "w"), indent=1)
# ---------------- figure
i_med = int(np.argsort([cp_of(t) for t in pop])[25]); th = pop[i_med]; xss = th[3] * th[1] / (th[3] * th[1] + th[0])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "savefig.dpi": 400, "axes.linewidth": 0.6, "lines.linewidth": 1.1})
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.3))
for a, (frac, lab, tmax) in zip(ax, ((0.60, "(a) 60% CP, moderate", 600.0), (0.85, "(b) 85% CP, heavy", 600.0), (1.10, "(c) 110% CP, severe", 900.0))):
    t, phi, AP, T = run(th, frac, tmax)
    a.plot(t, phi / th[0], color="#1b4f9c", label=r"$\varphi_1 / M_O$")
    a.plot(t, AP / th[2], color="#c0562a", ls="--", label=r"$A_P / A_{P,\max}$")
    a.axhline(1.0, color="k", lw=0.5, ls=":"); a.axhline(xss, color="#2e7d4f", lw=0.7, ls="-.", label=r"$x_{\mathrm{ss}}$ (flux at CP)")
    a.axhline(frac * xss, color="#8c8c8c", lw=0.7, ls=":", label="steady flux $P/\\eta$ / $M_O$")
    a.set_ylim(0, 1.05); a.set_xlabel("time (s)"); a.set_title(lab + (", $T_{\\lim}$ = %.0f s" % T if T < tmax else ""), loc="left"); a.tick_params(direction="in", top=True, right=True)
ax[0].set_ylabel("fraction"); ax[0].legend(frameon=False, loc="lower right")
fig.tight_layout(pad=0.4); FIG = os.path.join(HERE, "..", "..", "figures")
fig.savefig(os.path.join(FIG, "fig_vo2_kinetics_v5.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "fig_vo2_kinetics_v5.png"), bbox_inches="tight")
print("figure : athlete %d (CP %.0f W, x_ss %.3f, 1/M_R %.0f s)" % (i_med, cp_of(th), xss, 1 / th[3])); print("TERMINE vo2_kinetics_v5")
