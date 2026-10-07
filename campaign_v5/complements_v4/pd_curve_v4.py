#!/usr/bin/env python3
"""P-1 : relation puissance-duree du modele reduit. Pour les 50 athletes, T_lim aux
intensites f*CP_m (CP_m = Eq. 7), W' hyperbolique (P-CP_m)*T_lim, et CP/W' qu'un test
a deux parametres (P = CP + W'/T, regression lineaire de P sur 1/T) renverrait sur
les essais a 105/110/125/150 % de CP_m. Sortie : pd_curve_v4.json + fig_pd_curve.pdf/png."""
import json, os, glob, numpy as np
from common_v4 import p3, HERE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FR = [1.02, 1.05, 1.10, 1.25, 1.50, 2.00]; FIT = [1.05, 1.10, 1.25, 1.50]
def cpm(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def tlim(th, P, tmax=6000.0):
    _, te = p3.simulate(th, (lambda t: P), tmax); return te if te < tmax * 0.999 else np.inf
TH = {}
for f in sorted(glob.glob(os.path.join(HERE, "..", "camp_v3_*.jsonl"))):
    for l in open(f):
        r = json.loads(l); TH[r["i"]] = np.array(r["theta_true"])
out = []
for i in sorted(TH):
    th = TH[i]; CP = cpm(th); Wp = th[2] * th[4]
    T = {f: tlim(th, f * CP) for f in FR}
    Ps = np.array([f * CP for f in FIT]); Ts = np.array([T[f] for f in FIT])
    A = np.vstack([1 / Ts, np.ones(len(Ts))]).T; Wfit, CPfit = np.linalg.lstsq(A, Ps, rcond=None)[0]
    out.append(dict(i=i, CP_m=CP, Wp_m=Wp, Tlim={str(f): T[f] for f in FR},
                    Wp_hyp={str(f): (f - 1) * CP * T[f] for f in FR}, CP_test=CPfit, Wp_test=Wfit,
                    Tlim_cpmodel={str(f): Wp / ((f - 1) * CP) for f in FR}))
    print("i=%2d CP_m %.0f  CP_test %.0f (%.2f)  W'_m %.1f  W'_test %.1f (%.2f)  Tlim102 %.0f Tlim110 %.0f" % (
        i, CP, CPfit, CPfit / CP, Wp / 1e3, Wfit / 1e3, Wfit / Wp, T[1.02], T[1.10]), flush=True)
json.dump(out, open(os.path.join(HERE, "pd_curve_v4.json"), "w"), indent=1)
rc = np.array([o["CP_test"] / o["CP_m"] for o in out]); rw = np.array([o["Wp_test"] / o["Wp_m"] for o in out])
S = dict(n=len(out), CP_ratio=dict(median=float(np.median(rc)), q=[float(x) for x in np.percentile(rc, [25, 75])], range=[float(rc.min()), float(rc.max())]),
         Wp_ratio=dict(median=float(np.median(rw)), q=[float(x) for x in np.percentile(rw, [25, 75])], range=[float(rw.min()), float(rw.max())]),
         Tlim_median={str(f): float(np.median([o["Tlim"][str(f)] for o in out])) for f in FR},
         Tlim_range={str(f): [float(np.min([o["Tlim"][str(f)] for o in out])), float(np.max([o["Tlim"][str(f)] for o in out]))] for f in FR},
         Tlim_cpmodel_median={str(f): float(np.median([o["Tlim_cpmodel"][str(f)] for o in out])) for f in FR},
         Wp_hyp_median_kJ={str(f): float(np.median([o["Wp_hyp"][str(f)] for o in out])) / 1e3 for f in FR})
json.dump(S, open(os.path.join(HERE, "pd_curve_v4_summary.json"), "w"), indent=1); print(json.dumps(S, indent=1))
# ---- figure : (a) P-D du modele vs hyperboles, athlete median ; (b) ratios sur 50 athletes
med = sorted(out, key=lambda o: o["CP_m"])[len(out) // 2]; th = TH[med["i"]]; CP = med["CP_m"]; Wp = med["Wp_m"]
fs = np.linspace(1.01, 2.2, 60); Tm = np.array([tlim(th, f * CP) for f in fs])
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].plot(Tm, fs * CP, "k-", lw=2, label="reduced model, $T_{\\rm lim}$ to $A_P=0$")
Tg = np.linspace(40, 1500, 400)
ax[0].plot(Tg, CP + Wp / Tg, "--", color="#2a528c", lw=1.5, label="hyperbola, CP$_m$ and $W'_m=A_{P,\\max}\\eta$")
ax[0].plot(Tg, med["CP_test"] + med["Wp_test"] / Tg, ":", color="#8c2d37", lw=2, label="hyperbola fitted to model $T_{\\rm lim}$ (105–150%)")
ax[0].axhline(CP, color="#2a528c", lw=0.8, alpha=0.6); ax[0].axhline(med["CP_test"], color="#8c2d37", lw=0.8, alpha=0.6)
ax[0].set_xscale("log"); ax[0].set_xlim(40, 1500); ax[0].set_ylim(0.75 * CP, 2.3 * CP)
ax[0].set_xlabel("time to exhaustion (s)"); ax[0].set_ylabel("power (W)"); ax[0].legend(fontsize=7, loc="upper right"); ax[0].set_title("(a) power–duration relation, median athlete", fontsize=9)
ax[1].scatter(rc, rw, s=18, color="#4b4b4b"); ax[1].axvline(1, color="k", lw=0.6); ax[1].axhline(1, color="k", lw=0.6)
ax[1].set_xlabel("CP$_{\\rm test}$ / CP$_m$"); ax[1].set_ylabel("$W'_{\\rm test}$ / $W'_m$"); ax[1].set_title("(b) two-parameter fit on model trials, 50 athletes", fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(HERE, "..", "..", "figures", "fig_pd_curve.pdf")); fig.savefig(os.path.join(HERE, "..", "..", "figures", "fig_pd_curve.png"), dpi=300)
print("figure ok")
