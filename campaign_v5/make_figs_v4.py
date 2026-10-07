"""Figures des Results (campagne v3 + PINN v4). Toutes les valeurs proviennent
des donnees ; aucune valeur saisie a la main."""
import json, glob, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "savefig.dpi": 400})
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "figures") + "/"; C = os.path.join(HERE, "complements_v4")
LBL = [r"$M_O$", r"$A_{O,\max}$", r"$A_{P,\max}$", r"$M_R$", r"$\eta$"]
COL = {"pop": "#8c8c8c", "LM": "#1b4f9c", "DE": "#c0562a", "PINN": "#9fcf9f", "PINN_v4": "#2e7d4f"}
NAME = {"pop": "Population mean", "LM": "Levenberg–Marquardt", "DE": "Differential Evolution", "PINN": "PINN, as submitted", "PINN_v4": "PINN, corrected"}
V3 = [json.loads(l) for f in sorted(glob.glob(HERE + "/camp_v3_*.jsonl")) for l in open(f)]
V4 = {(r["i"], r["sigma_P"]): r for f in sorted(glob.glob(HERE + "/camp_v4_[0-9].jsonl")) for l in open(f) for r in [json.loads(l)]}
def est(r, m): return np.array(V4[(r["i"], r["sigma_P"])]["est"]) if m == "PINN_v4" else np.array(r["est"][m])
def cp(th): M_O, A_Om, A_Pm, M_R, eta = th; return M_O * (M_R * A_Om / (M_R * A_Om + M_O)) * eta
sub = [r for r in V3 if r["sigma_P"] == 5.0]
# ================================ identifiabilite
Z = np.load(C + "/fim_v3.npz"); F = np.load(C + "/flat_traj_v3.npz")
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))
sp = Z["spec"]
ax[0].boxplot([sp[:, k] for k in range(5)], widths=0.55, showfliers=False, medianprops=dict(color="#1b4f9c", lw=1.2), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
ax[0].set_yscale("log"); ax[0].set_xlabel("eigenvalue rank"); ax[0].set_ylabel(r"$\lambda_k / \lambda_1$"); ax[0].yaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax[0].text(0.03, 0.06, r"$\kappa = %.1f \times 10^{5}$" % (np.median(Z["cond"]) / 1e5), transform=ax[0].transAxes, fontsize=7); ax[0].set_title("(a) normalized FIM spectrum", loc="left")
so = Z["soft"]; ax[1].axhline(0, color="k", lw=0.5)
ax[1].boxplot([so[:, k] for k in range(5)], widths=0.55, showfliers=False, medianprops=dict(color="#c0562a", lw=1.2), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
ax[1].set_xticklabels(LBL); ax[1].set_ylabel(r"coordinate on $\ln\theta$"); ax[1].set_title("(b) least constrained direction", loc="left")
t = F["t"]
ax[2].plot(t, F["ref"] * 100, color="k", label="reference")
ax[2].plot(t, F["sym"] * 100, color="#c0562a", ls="--", label=r"$M_O\!\times\!2,\ \eta/2,\ A_{O,\max}\!\times\!2$")
ax[2].plot(t, F["prod"] * 100, color="#1b4f9c", ls="-.", label=r"$A_{O,\max}\!\times\!2,\ M_R/2$")
ax[2].plot(t, F["fit"] * 100, color="#2e7d4f", ls=":", label=r"$A_{O,\max}\!\times\!2,\ M_R\!\times\!2^{%.2f}$" % float(F["vfit"][3] / F["vfit"][1]))
ax[2].set_xlabel("time (s)"); ax[2].set_ylabel(r"$A_P(t)$ (% of $A_{P,\max}$)"); ax[2].legend(frameon=False, loc="upper right", handlelength=1.6); ax[2].set_title("(c) forward simulation, protocol B", loc="left")
for a in ax: a.tick_params(direction="in", top=True, right=True)
fig.tight_layout(pad=0.4); fig.savefig(OUT + "fig_identifiability.pdf", bbox_inches="tight"); fig.savefig(OUT + "fig_identifiability.png", bbox_inches="tight"); plt.close(fig)
# ================================ erreur vs CRLB
meth = ["pop", "LM", "DE", "PINN", "PINN_v4"]
err = {m: np.array([[100 * abs(est(r, m)[k] - r["theta_true"][k]) / abs(r["theta_true"][k]) for k in range(5)] for r in sub]) for m in meth}
crlb = np.median([r["crlb_pct"] for r in sub], axis=0)
fig, ax = plt.subplots(figsize=(4.6, 2.7)); x = np.arange(5); w = 0.16
for j, m in enumerate(meth):
    ax.bar(x + (j - 2) * w, np.median(err[m], axis=0), w, color=COL[m], label=NAME[m], edgecolor="none")
for k in range(5): ax.plot([k - 0.45, k + 0.45], [0.674 * crlb[k]] * 2, color="k", lw=1.2, zorder=5)
ax.plot([], [], color="k", lw=1.2, label=r"efficient estimator ($0.674\,\sigma_{CR}$)")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(LBL); ax.set_ylabel("median relative error (%)"); ax.set_ylim(1e-2, 5e2)
ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=6); ax.tick_params(direction="in", top=True, right=True)
fig.tight_layout(pad=0.4); fig.savefig(OUT + "fig_accuracy.pdf", bbox_inches="tight"); fig.savefig(OUT + "fig_accuracy.png", bbox_inches="tight"); plt.close(fig)
# ================================ vs baseline
fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.1), sharex=True, sharey=True)
for j, m in enumerate(["LM", "DE", "PINN", "PINN_v4"]):
    eb = np.array([100 * abs(cp(est(r, "pop")) - cp(r["theta_true"])) / cp(r["theta_true"]) for r in sub])
    em = np.array([100 * abs(cp(est(r, m)) - cp(r["theta_true"])) / cp(r["theta_true"]) for r in sub])
    n = int((em < eb).sum()); ax[j].plot([1e-3, 1e3], [1e-3, 1e3], color="k", lw=0.6)
    ax[j].scatter(eb, em, s=9, color=COL[m], alpha=0.8, edgecolor="none"); ax[j].set_xscale("log"); ax[j].set_yscale("log"); ax[j].set_xlim(1e-2, 2e2); ax[j].set_ylim(1e-2, 2e2)
    ax[j].set_title(NAME[m], loc="left", fontsize=7.5)
    ax[j].text(0.04, 0.92, "%d/50 below the line" % n, transform=ax[j].transAxes, fontsize=7, va="top"); ax[j].tick_params(direction="in", top=True, right=True)
ax[0].set_ylabel("method CP error (%)"); fig.supxlabel("population-mean CP error (%)", fontsize=8, y=0.02)
fig.tight_layout(pad=0.4, rect=(0, 0.04, 1, 1)); fig.savefig(OUT + "fig_vs_baseline.pdf", bbox_inches="tight"); fig.savefig(OUT + "fig_vs_baseline.png", bbox_inches="tight"); plt.close(fig)
# ================================ PINN v4 : biais le long de la direction plate + exces de RMSE
V4l = list(V4.values()); T = np.array([r["theta_true"] for r in V4l]); E = np.array([r["est"] for r in V4l])
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.3))
ax[0].plot([1e5, 8e5], [1e5, 8e5], color="k", lw=0.6)
for m, c, lab in (("LM", COL["LM"], NAME["LM"]), ("PINN_v4", COL["PINN_v4"], NAME["PINN_v4"])):
    ee = np.array([est(r, m)[1] for r in V3]); tt = np.array([r["theta_true"][1] for r in V3])
    ax[0].scatter(tt / 1e3, ee / 1e3, s=7, color=c, alpha=0.7, edgecolor="none", label=lab)
ax[0].set_xlabel(r"true $A_{O,\max}$ (kJ)"); ax[0].set_ylabel(r"estimated $A_{O,\max}$ (kJ)"); ax[0].legend(frameon=False, loc="upper left", markerscale=2.2, handletextpad=0.3); ax[0].set_title("(a) shrinkage of $A_{O,\\max}$", loc="left")
ax[0].set_xlim(80, 800); ax[0].set_ylim(80, 900)
dA = np.log(E[:, 1] / T[:, 1]); dM = np.log(E[:, 3] / T[:, 3])
ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(0, color="k", lw=0.5)
ax[1].scatter(dA, dM, s=7, color=COL["PINN_v4"], alpha=0.7, edgecolor="none")
r_ = np.corrcoef(dA, dM)[0, 1]; ax[1].text(0.04, 0.06, "r = %.2f" % r_, transform=ax[1].transAxes, fontsize=7)
ax[1].set_xlabel(r"$\ln(\hat A_{O,\max} / A_{O,\max})$"); ax[1].set_ylabel(r"$\ln(\hat M_R / M_R)$"); ax[1].set_title("(b) error along the flat direction", loc="left")
sig = np.array([r["sigma_P"] for r in V4l]); ex = np.array([r["rmse_fit"] - r["rmse_true"] for r in V4l])
ax[2].boxplot([ex[sig == s] for s in (2., 5., 10.)], widths=0.5, showfliers=False, medianprops=dict(color=COL["PINN_v4"], lw=1.2), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
ax[2].set_xticklabels(["2", "5", "10"]); ax[2].set_xlabel(r"$\sigma_P$ (W)"); ax[2].set_ylabel(r"RMSE$_{\hat\theta}$ $-$ RMSE$_{\theta}$ (J)"); ax[2].set_title("(c) excess misfit, corrected PINN", loc="left")
for a in ax: a.tick_params(direction="in", top=True, right=True)
fig.tight_layout(pad=0.4); fig.savefig(OUT + "fig_pinn_bias.pdf", bbox_inches="tight"); fig.savefig(OUT + "fig_pinn_bias.png", bbox_inches="tight"); plt.close(fig)
print("figures ecrites dans", os.path.abspath(OUT))
for m in meth: print("%-8s med err = %s" % (m, np.round(np.median(err[m], axis=0), 2)))
print("CRLB =", np.round(crlb, 2)); print("corr flat dir PINN v4 =", round(r_, 2))
