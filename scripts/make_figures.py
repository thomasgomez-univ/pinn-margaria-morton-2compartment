"""Figures de la section Results. Toutes les valeurs proviennent des donnees
de campagne ; aucune valeur n'est saisie a la main."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "savefig.dpi": 400, "figure.dpi": 400,
})
OUT = _os.path.join(_ROOT, "figures") + _os.sep
LBL = [r"$M_O$", r"$A_{O,\max}$", r"$A_{P,\max}$", r"$M_R$", r"$\eta$"]
COL = {"pop": "#8c8c8c", "LM": "#1b4f9c", "DE": "#c0562a", "PINN": "#3f8f5a"}

# ----------------------------------------------------------------- donnees
Z = np.load(_os.path.join(_ROOT, "data", "fim_full.npz"))
F = np.load(_os.path.join(_ROOT, "data", "flat_traj.npz"))
rows = [json.loads(l) for l in
        open(_os.path.join(_ROOT, "data", "campaign_main.jsonl"))]

# =========================================================== Figure : FIM
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))

# (a) spectre
sp = Z["spec"]
ax[0].boxplot([sp[:, k] for k in range(5)], widths=0.55, showfliers=False,
              medianprops=dict(color="#1b4f9c", lw=1.2),
              boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
              capprops=dict(lw=0.6))
ax[0].set_yscale("log")
ax[0].set_xlabel("eigenvalue rank")
ax[0].set_ylabel(r"$\lambda_k / \lambda_1$")
ax[0].yaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax[0].text(0.03, 0.06, r"$\kappa = %.1f \times 10^{5}$" % (np.median(Z["cond"]) / 1e5),
           transform=ax[0].transAxes, fontsize=7)
ax[0].set_title("(a) normalized FIM spectrum", loc="left")

# (b) direction plate
so = Z["soft"]
ax[1].axhline(0, color="k", lw=0.5)
ax[1].boxplot([so[:, k] for k in range(5)], widths=0.55, showfliers=False,
              medianprops=dict(color="#c0562a", lw=1.2),
              boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
              capprops=dict(lw=0.6))
ax[1].set_xticklabels(LBL)
ax[1].set_ylabel(r"coordinate on $\ln\theta$")
ax[1].set_title("(b) least constrained direction", loc="left")

# (c) trajectoires
t, APm = F["t"], 100.0
ax[2].plot(t, F["ref"] * APm, color="k", label="reference")
ax[2].plot(t, F["sym"] * APm, color="#c0562a", ls="--",
           label=r"$M_O\!\times\!2,\ \eta/2,\ A_{O,\max}\!\times\!2$")
ax[2].plot(t, F["prod"] * APm, color="#1b4f9c", ls="-.",
           label=r"$A_{O,\max}\!\times\!2,\ M_R/2$")
ax[2].plot(t, F["fit"] * APm, color="#3f8f5a", ls=":",
           label=r"fitted flat direction")
ax[2].set_xlabel("time (s)")
ax[2].set_ylabel("$A_P(t)$ (% of $A_{P,\\max}$)")
ax[2].legend(frameon=False, loc="upper right", handlelength=1.6)
ax[2].set_title("(c) forward simulation, protocol B", loc="left")

for a in ax:
    a.tick_params(direction="in", top=True, right=True)
fig.tight_layout(pad=0.4)
fig.savefig(OUT + "fig_identifiability.pdf", bbox_inches="tight")
fig.savefig(OUT + "fig_identifiability.png", bbox_inches="tight")
plt.close(fig)

# ================================================= Figure : erreur vs CRLB
sig = 5.0
sub = [r for r in rows if r["sigma_P"] == sig]
meth = ["pop", "LM", "DE", "PINN"]
name = {"pop": "Population mean", "LM": "Levenberg--Marquardt".replace("--", "–"),
        "DE": "Differential Evolution", "PINN": r"PINN ($w_r = 0.3$)"}
err = {m: np.array([[100 * abs(r["est"][m][k] - r["theta_true"][k]) / r["theta_true"][k]
                     for k in range(5)] for r in sub]) for m in meth}
crlb = np.median([r["crlb_pct"] for r in sub], axis=0)

fig, ax = plt.subplots(figsize=(4.4, 2.6))
x = np.arange(5)
w = 0.2
for j, m in enumerate(meth):
    ax.bar(x + (j - 1.5) * w, np.median(err[m], axis=0), w, color=COL[m],
           label=name[m], edgecolor="none")
for k in range(5):
    ax.plot([k - 0.5, k + 0.5], [0.674 * crlb[k]] * 2, color="k", lw=1.2, zorder=5)
ax.plot([], [], color="k", lw=1.2, label="efficient estimator ($0.674\\,\\sigma_{CR}$)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(LBL)
ax.set_ylabel("median relative error (%)")
ax.set_ylim(1e-2, 3e2)
ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=6.5)
ax.tick_params(direction="in", top=True, right=True)
fig.tight_layout(pad=0.4)
fig.savefig(OUT + "fig_accuracy.pdf", bbox_inches="tight")
fig.savefig(OUT + "fig_accuracy.png", bbox_inches="tight")
plt.close(fig)

# ========================================= Figure : par athlete vs baseline
SHORT = {"LM": "Levenberg\u2013Marquardt", "DE": "Differential Evolution",
         "PINN": "PINN ($w_r = 0.3$)"}

def cp(th):
    return th[0] * th[4]

fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.5), sharex=True, sharey=True)
for j, m in enumerate(["LM", "DE", "PINN"]):
    eb, em = [], []
    for r in sub:
        c0 = cp(r["theta_true"])
        eb.append(100 * abs(cp(r["est"]["pop"]) - c0) / c0)
        em.append(100 * abs(cp(r["est"][m]) - c0) / c0)
    eb, em = np.array(eb), np.array(em)
    n = int((em < eb).sum())
    ax[j].plot([1e-3, 1e3], [1e-3, 1e3], color="k", lw=0.6)
    ax[j].scatter(eb, em, s=9, color=COL[m], alpha=0.75, edgecolor="none")
    ax[j].set_xscale("log")
    ax[j].set_yscale("log")
    ax[j].set_xlim(1e-2, 2e2)
    ax[j].set_ylim(1e-2, 2e2)
    ax[j].set_xlabel("population-mean CP error (%)")
    ax[j].set_title("%s" % SHORT[m], loc="left", fontsize=7.5)
    ax[j].text(0.04, 0.92, "%d/50 below the line" % n, transform=ax[j].transAxes,
               fontsize=7, va="top")
    ax[j].tick_params(direction="in", top=True, right=True)
ax[0].set_ylabel("method CP error (%)")
fig.tight_layout(pad=0.4)
fig.savefig(OUT + "fig_vs_baseline.pdf", bbox_inches="tight")
fig.savefig(OUT + "fig_vs_baseline.png", bbox_inches="tight")
plt.close(fig)

print("figures ecrites dans", OUT)
for m in meth:
    print("%-6s med err = %s" % (m, np.round(np.median(err[m], axis=0), 2)))
print("CRLB  =", np.round(crlb, 2))
