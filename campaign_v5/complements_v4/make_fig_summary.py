#!/usr/bin/env python3
"""Figure de synthese (rev.3 bis) : borne de Cramer-Rao mediane (%) par grandeur et par observable.
Sources : campagne (50 athletes, bruit blanc : fim_v3 / camp_v3 crlb_pct ; flux et CP_PD/W'_PD : rev3_cp_pd, common_v4),
rev3_sigma_fim (20 athletes, covariance induite, alpha 0.1, nu 0.1) ; VO2 + T_lim (neuf athletes, quatre reglages de
variabilite, etendue min-max des medianes) et CP/W'/flux/CP_PD/W'_PD sous bruit blanc : valeurs du manuscrit, codees en dur ci-dessous. VO2 seule : rang 4, CP et W' non identifiables."""
import json, glob, os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 8, "axes.linewidth": 0.6, "savefig.dpi": 400})
HERE = os.path.dirname(os.path.abspath(__file__)); RUN = os.path.join(HERE, ".."); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
sys.path.insert(0, RUN); sys.path.insert(0, HERE)   # common_v4 et rev3_sigma_fim_*.jsonl : a cote du script (depot) ou dans le dossier parent (session)
from common_v4 import load_v3
V3 = load_v3(); S = [V3[(i, 5.0)] for i in range(50)]
white = np.median([r["crlb_pct"] for r in S], 0)
# CP, W', flux sur les 50 athletes (bruit blanc) : valeurs du manuscrit (recalculees en session, cf. revision_rev3_calculs.md)
white_cp, white_wp, white_flux, white_cppd, white_wppd = 0.085, 0.178, 1.479, 0.081, 0.144
R = [json.loads(l) for f in (glob.glob(os.path.join(HERE, "rev3_sigma_fim_*.jsonl")) or glob.glob(os.path.join(RUN, "rev3_sigma_fim_*.jsonl"))) for l in open(f)]
sig = np.median([r["crlb_sigma"]["a0.1_n0.1"] for r in R], 0); sig_cp = np.median([r["cp_sigma"]["a0.1_n0.1"] for r in R]); sig_wp = np.median([r["wp_sigma"]["a0.1_n0.1"] for r in R])
# VO2 + T_lim : etendue des medianes sur les quatre reglages (neuf athletes), vo2_tlim_v5.json
pair = {"par_lo": [4.64, 40.88, 6.53, 59.07, 1.87], "par_hi": [9.62, 84.21, 13.91, 125.65, 3.96], "CP": (4.89, 10.34), "Wp": (4.70, 10.07)}
rows = [("CP", white_cp, sig_cp, pair["CP"]), ("$W'$", white_wp, sig_wp, pair["Wp"]), ("$\\mathrm{CP}_{\\mathrm{PD}}$", white_cppd, None, None), ("$W'_{\\mathrm{PD}}$", white_wppd, None, None),
        ("$M_R \\cdot A_{O,\\max}$", white_flux, None, None), ("$M_O$", white[0], sig[0], (pair["par_lo"][0], pair["par_hi"][0])), ("$\\eta$", white[4], sig[4], (pair["par_lo"][4], pair["par_hi"][4])),
        ("$A_{P,\\max}$", white[2], None, (pair["par_lo"][2], pair["par_hi"][2])), ("$A_{O,\\max}$", white[1], sig[1], (pair["par_lo"][1], pair["par_hi"][1])), ("$M_R$", white[3], sig[3], (pair["par_lo"][3], pair["par_hi"][3]))]
C = {"white": "#1b4f9c", "sigma": "#c0562a", "pair": "#2e7d4f"}
fig, ax = plt.subplots(figsize=(6.3, 3.9)); y = np.arange(len(rows))[::-1]; h = 0.26
for k, (lab, w, s, p) in enumerate(rows):
    ax.barh(y[k] + h, w, h, color=C["white"], edgecolor="none", label="$A_P$, independent noise (50 athletes)" if k == 0 else None)
    ax.text(w * 1.15, y[k] + h, "%.3g" % w, va="center", fontsize=6.5, color="#333333")
    if s is not None:
        ax.barh(y[k], s, h, color=C["sigma"], edgecolor="none", hatch="////", label="$A_P$, induced noise covariance (20)" if k == 0 else None)
        ax.text(s * 1.15, y[k], "%.3g" % s, va="center", fontsize=6.5, color="#333333")
    elif lab.startswith("$A_{P"):
        ax.text(0.011, y[k], "set by the nugget", va="center", fontsize=6.5, color="#777777", style="italic")
    if p is not None:
        ax.barh(y[k] - h, p[1] - p[0], h, left=p[0], color=C["pair"], edgecolor="none", alpha=0.9, label="$\\dot{V}\\mathrm{O}_2$ + $T_{\\lim}$ (nine athletes, range over noise settings)" if k == 0 else None)
        ax.text(p[1] * 1.15, y[k] - h, "%.3g–%.3g" % p, va="center", fontsize=6.5, color="#333333")
ax.set_xscale("log"); ax.set_xlim(0.01, 400); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("Cramér–Rao bound on the relative standard deviation (%), median over athletes, $\\sigma_P = 5$ W")
ax.axvline(1, color="#bbbbbb", lw=0.6, ls=":"); ax.axvline(10, color="#bbbbbb", lw=0.6, ls=":")
ax.set_ylim(-0.6, len(rows) - 0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
hd, lb = ax.get_legend_handles_labels(); fig.legend(hd, lb, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.6)
fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "fig_summary.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig_summary.png"), bbox_inches="tight"); print("ok")
