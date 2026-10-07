#!/usr/bin/env python3
"""Corps LaTeX des tableaux et chiffres du texte, a partir de camp_v3 + camp_v4
(+ fim_v3, rep_lm, protocoles, inverse_crime, wr_sweep quand presents).
Aucune valeur saisie a la main. Sorties : tables_v4.tex, numbers_v4.json."""
import json, glob, os, numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "complements_v4")
P = ["M_O", "A_Omax", "A_Pmax", "M_R", "eta"]
LB = np.array([600., 5e4, 3e4, 0.003, 0.15]); UB = np.array([3000., 1.5e6, 1.6e5, 0.080, 0.35])
CP = lambda t: t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]; Wp = lambda t: t[2] * t[4]
MED = 0.6744897501960817
V3 = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "camp_v3_*.jsonl"))) for l in open(f)]
V4 = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "camp_v4_[0-9].jsonl"))) for l in open(f)]
assert len(V3) == 150 and len(V4) == 150, (len(V3), len(V4))
k4 = {(r["i"], r["sigma_P"]): r for r in V4}
def est(r, m): return np.array(k4[(r["i"], r["sigma_P"])]["est"]) if m == "PINN_v4" else np.array(r["est"][m])
METH = [("pop", "Population mean"), ("pop+APmax", r"Population mean $+\ A_{P,\max}$"), ("LM", "Levenberg--Marquardt"),
        ("DE", "Differential Evolution"), ("PINN", r"PINN, as submitted"), ("PINN_v4", r"PINN, corrected (Section~\ref{sec:pinn_v4})")]
N = {}; T = []
def fmt(v, iqr=None, bold=False):
    s = "%.2f" % v + ("" if iqr is None else " [%.2f]" % iqr)
    return r"\textbf{%s}" % s if bold else s
def err_q(r, m, f): t = np.array(r["theta_true"]); return 100 * abs(f(est(r, m)) - f(t)) / f(t)
# ---------------- Table cp_wp
rows = []
for m, name in METH:
    cells = []
    for f in (CP, Wp):
        for s in (2., 5., 10.):
            v = [err_q(r, m, f) for r in V3 if r["sigma_P"] == s]; cells.append((np.median(v), np.percentile(v, 75) - np.percentile(v, 25)))
    rows.append((m, name, cells))
best = [min(range(len(rows)), key=lambda j: rows[j][2][c][0]) for c in range(6)]
T.append("%% ---- tab:cp_wp\n" + "\n".join(name + " & " + " & ".join(fmt(c[0], c[1], bold=(best[ci] == j)) for ci, c in enumerate(cells)) + r" \\" for j, (m, name, cells) in enumerate(rows)))
N["cp_wp"] = {m: [c[0] for c in cells] for m, _, cells in rows}
# ---------------- Table per_param (sigma = 5)
S5 = [r for r in V3 if r["sigma_P"] == 5.]
def perr(r, m): t = np.array(r["theta_true"]); return 100 * np.abs(est(r, m) - t) / np.abs(t)
E = {m: np.array([perr(r, m) for r in S5]) for m, _ in METH}
crlb = np.median([r["crlb_pct"] for r in S5], 0)
lines = [r"\multicolumn{6}{l}{\emph{Median relative error} (\%)} \\"]
for m, name in METH:
    med = np.median(E[m], 0); lines.append(name + " & " + " & ".join((r"\textbf{%.2f}" % v if m == "LM" else "%.2f" % v) for v in med) + r" \\")
lines.append(r"Cram\'er--Rao bound (\%) & " + " & ".join("%.2f" % v for v in crlb) + r" \\[2pt]")
lines.append(r"\multicolumn{6}{l}{\emph{Athlete-level efficiency, 50 athletes, one realization each}} \\")
for m, name in (("LM", "Levenberg--Marquardt"), ("PINN_v4", "PINN, corrected")):
    rat = np.array([perr(r, m) / np.array(r["crlb_pct"]) for r in S5]); cov = np.array([perr(r, m) < 1.96 * np.array(r["crlb_pct"]) for r in S5])
    lines.append(name + r", median $|$error$|$ / $(0.674\,\sigma_{\mathrm{CR}})$ & " + " & ".join("%.2f" % v for v in np.median(rat, 0) / MED) + r" \\")
    lines.append(name + r", coverage of $\pm 1.96\,\sigma_{\mathrm{CR}}$ (\%) & " + " & ".join("%.0f" % v for v in 100 * cov.mean(0)) + r" \\")
    N["eff_" + m] = (np.median(rat, 0) / MED).tolist(); N["cov_" + m] = (100 * cov.mean(0)).tolist()
se_ratio = np.median([100 * np.array(r["LM_se"]) / np.abs(r["theta_true"]) / np.array(r["crlb_pct"]) for r in S5], 0)
N["se_ratio_LM"] = se_ratio.tolist()
# lignes realisations repetees si disponibles
REP = [json.loads(l) for f in sorted(glob.glob(os.path.join(C, "rep_lm_v3_*.jsonl"))) for l in open(f)]
if REP:
    byi = {}
    for r in REP: byi.setdefault(r["i"], []).append(r)
    sd, ratio, bias = [], [], []
    for i, rs in byi.items():
        if len(rs) < 20: continue
        t = np.array(rs[0]["theta_true"]); X = np.array([r["est"] for r in rs]); cr = np.array(rs[0]["crlb_pct"])
        s = 100 * X.std(0, ddof=1) / np.abs(t); sd.append(s); ratio.append(s / cr); bias.append(np.abs(100 * (X.mean(0) - t) / t) / s)
    if sd:
        lines.append(r"\multicolumn{6}{l}{\emph{Levenberg--Marquardt, 20 noise realizations $\times$ %d athletes}} \\" % len(sd))
        lines.append(r"Sampling standard deviation (\%) & " + " & ".join("%.2f" % v for v in np.median(sd, 0)) + r" \\")
        lines.append(r"Ratio to Cram\'er--Rao bound & " + " & ".join("%.2f" % v for v in np.median(ratio, 0)) + r" \\")
        lines.append(r"$|$bias$|$ / sampling s.d. & " + " & ".join("%.2f" % v for v in np.median(bias, 0)) + r" \\")
        N["rep_lm"] = dict(n_ath=len(sd), sd=np.median(sd, 0).tolist(), ratio=np.median(ratio, 0).tolist(), bias=np.median(bias, 0).tolist(),
                           ratio_min=np.min(np.median(ratio, 0)), ratio_max=np.max(np.median(ratio, 0)))
T.append("%% ---- tab:per_param\n" + "\n".join(lines))
N["per_param"] = {m: np.median(E[m], 0).tolist() for m, _ in METH}; N["crlb"] = crlb.tolist()
# ---------------- Table vs_baseline
lines = []
for m, name in METH[2:]:
    cells = []
    for s in (2., 5., 10.):
        sub = [r for r in V3 if r["sigma_P"] == s]; eb = np.array([err_q(r, "pop", CP) for r in sub]); em = np.array([err_q(r, m, CP) for r in sub])
        n = int((em < eb).sum()); p = wilcoxon(em, eb, alternative="less").pvalue
        ex = int(np.floor(np.log10(p))); cells.append("%d/50 ($p = %.1f\\times10^{%d}$)" % (n, p / 10 ** ex, ex)); N.setdefault("vs_base", {}).setdefault(m, []).append([n, p])
    lines.append(name + " & " + " & ".join(cells) + r" \\")
T.append("%% ---- tab:vs_baseline\n" + "\n".join(lines))
# ---------------- Population (tab:params) et protocoles
pop = np.load(os.path.join(HERE, "pop_recal.npy")); M_O, A_Om, A_Pm, M_R, eta = pop.T
xss = M_R * A_Om / (M_R * A_Om + M_O); cp = M_O * xss * eta; wp = A_Pm * eta; vo2 = M_O / 20.9 * 60 / 1000
N["pop"] = dict(M_O=[M_O.mean(), M_O.std(ddof=1), M_O.min(), M_O.max()], A_Omax_kJ=[A_Om.mean() / 1e3, A_Om.std(ddof=1) / 1e3, A_Om.min() / 1e3, A_Om.max() / 1e3],
                A_Pmax_kJ=[A_Pm.mean() / 1e3, A_Pm.std(ddof=1) / 1e3], M_R=[M_R.mean(), M_R.std(ddof=1), M_R.min(), M_R.max()], invMR=[np.median(1 / M_R), (1 / M_R).min(), (1 / M_R).max()],
                eta=[eta.mean(), eta.std(ddof=1)], CP=[cp.mean(), cp.std(ddof=1), cp.min(), cp.max()], Wp_kJ=[wp.mean() / 1e3, wp.std(ddof=1) / 1e3], xss=[xss.mean(), xss.std(ddof=1), xss.min(), xss.max()],
                VO2max=[vo2.mean(), vo2.std(ddof=1), vo2.min(), vo2.max()], MOeta=[(M_O * eta).mean(), (M_O * eta).std(ddof=1)])
sA = np.median([r["sigma_AP"]["A"] for r in S5]); sB = np.median([r["sigma_AP"]["B"] for r in S5]); N["sigma_AP_5W"] = [sA, sB]
# ---------------- Budget, temps
N["ode"] = {m: [np.median([r["ode_calls"][m] for r in V3]), max(r["ode_calls"][m] for r in V3)] for m in ("LM", "DE")}
N["wall"] = {m: np.median([r["wall_s"][m] for r in V3]) for m in ("LM", "DE", "PINN")}; N["wall"]["PINN_v4"] = np.median([r["wall_s"] for r in V4])
# ---------------- PINN v4 : biais, exces de RMSE, bornes
Sg = np.array([(np.array(r["est"]) - np.array(r["theta_true"])) / np.array(r["theta_true"]) for r in V4]); Tt = np.array([r["theta_true"] for r in V4])
N["pinn_v4"] = dict(signed_med=(100 * np.median(Sg, 0)).tolist(), frac_pos=Sg.__gt__(0).mean(0).tolist(),
                    corr_AOm=float(np.corrcoef(Tt[:, 1], Sg[:, 1])[0, 1]), corr_MR=float(np.corrcoef(Tt[:, 3], Sg[:, 3])[0, 1]),
                    corr_flat=float(np.corrcoef(np.log(1 + Sg[:, 1]), np.log(1 + Sg[:, 3]))[0, 1]),
                    excess_J={s: float(np.median([r["rmse_fit"] - r["rmse_true"] for r in V4 if r["sigma_P"] == s])) for s in (2., 5., 10.)},
                    ratio={s: float(np.median([r["rmse_fit"] / r["rmse_true"] for r in V4 if r["sigma_P"] == s])) for s in (2., 5., 10.)},
                    AOm_est_med=float(np.median([r["est"][1] for r in V4])), AOm_true_med=float(np.median(Tt[:, 1])),
                    range_sigma={m: np.median([100 * (np.ptp([est(r_, m) for r_ in V3 if r_["i"] == i], 0)) / np.abs(np.array(next(r_ for r_ in V3 if r_["i"] == i)["theta_true"])) for i in range(50)], 0).tolist() for m in ("LM", "PINN_v4")},
                    crlb10=np.median([r["crlb_pct"] for r in V3 if r["sigma_P"] == 10.], 0).tolist())
for m in ("PINN", "PINN_v4"):
    Em = np.array([est(r, m) for r in V3]); fr = (Em - LB) / (UB - LB); N["bounds_" + m] = (100 * ((fr < 0.02) | (fr > 0.98)).mean(0)).tolist()
# ---------------- LM RMSE vs vrai (misfit) : pas stocke en v3 -> voir misfit.py ; PINN v3/v4 :
# ---------------- FIM (fim_v3)
if os.path.exists(os.path.join(C, "fim_v3.npz")):
    Z = np.load(os.path.join(C, "fim_v3.npz")); Fz = np.load(os.path.join(C, "flat_traj_v3.npz"))
    N["fim"] = dict(spec=np.median(Z["spec"], 0).tolist(), cond=[np.median(Z["cond"]), *np.percentile(Z["cond"], [25, 75])], vmin=np.median(Z["soft"], 0).tolist(),
                    expo=[np.median(Z["expo"]), *np.percentile(Z["expo"], [25, 75])], crlb=np.median(Z["crlb"], 0).tolist(),
                    disp=dict(sym=float(Fz["d_sym"]), prod=float(Fz["d_prod"]), fitx=float(Fz["d_fit"]) if "d_fit" in Fz.files else None))
# ---------------- Protocoles
PR = list({(r["proto"], r["i"]): r for r in (json.loads(l) for l in open(os.path.join(C, "protocoles_v3.jsonl")))}.values()) if os.path.exists(os.path.join(C, "protocoles_v3.jsonl")) else []
if PR:
    names = {"manuscrit A+B": "Protocols A and B (as above)", "manuscrit A seul": "Protocol A alone", "const 105%": r"Constant, 105\% CP", "const 140%": r"Constant, 140\% CP",
             "interm 120/60 30s": r"Intermittent 120/60\%, 30~s blocks", "interm 120/60 120s": r"Intermittent 120/60\%, 120~s blocks",
             "interm 120/60 60s": r"Intermittent 120/60\%, 60~s blocks",
             "const 105+140": r"Constant 105\% + constant 140\%", "const 110+140": r"Constant 110\% + constant 140\%",
             "105 + interm120/60": r"Constant 105\% + intermittent 120/60\%, 60~s", "140 + interm120/60": r"Constant 140\% + intermittent 120/60\%, 60~s",
             "105 + 140 + interm": r"105\% + 140\% + intermittent 120/60\%, 60~s"}
    lines = []; N["protocols"] = {}
    for key, name in names.items():
        rs = [r for r in PR if r["proto"] == key and r.get("ok")]
        if not rs: continue
        c = np.median([r["cond"] for r in rs]); b = np.median([r["crlb"] for r in rs], 0); ex = int(np.floor(np.log10(c)))
        lines.append(name + r" & $%.2f \times 10^{%d}$ & " % (c / 10 ** ex, ex) + " & ".join("%.2f" % v for v in b) + r" \\")
        N["protocols"][key] = dict(n=len(rs), cond=c, crlb=b.tolist(), lam_min=np.median([r["lam_min"] for r in rs]), logdet=np.median([r["logdet"] for r in rs]),
                                   expo=np.median([r["expo"] for r in rs]))
    T.append("%% ---- tab:protocols\n" + "\n".join(lines))
    N["protocols_all"] = {k: dict(n=len([r for r in PR if r["proto"] == k and r.get("ok")]), cond=np.median([r["cond"] for r in PR if r["proto"] == k and r.get("ok")] or [np.nan]),
                                  crlb=np.median([r["crlb"] for r in PR if r["proto"] == k and r.get("ok")] or [[np.nan] * 5], 0).tolist()) for k in sorted(set(r["proto"] for r in PR))}
# ---------------- crime inverse, balayage w_r
IC = [json.loads(l) for f in sorted(glob.glob(os.path.join(C, "inverse_crime_v3_*.jsonl"))) for l in open(f)]
if IC:
    N["inverse_crime"] = {}
    for m in ("LM", "DE", "PINN_v4"):
        a = np.median([err_q(dict(theta_true=r["theta_true"], est={m: r["est"][m]}, i=r["i"], sigma_P=5.) if m != "PINN_v4" else r, m if m != "PINN_v4" else "PINN_v4", CP) if m != "PINN_v4" else 100 * abs(CP(np.array(r["est"]["PINN_v4"])) - CP(np.array(r["theta_true"]))) / CP(np.array(r["theta_true"])) for r in IC])
        ref = np.median([err_q(r, m, CP) for r in V3 if r["sigma_P"] == 5. and r["i"] < 15])
        N["inverse_crime"][m] = dict(cp_dop853=float(a), cp_campagne=float(ref), n=len(IC))
WR = [json.loads(l) for f in sorted(glob.glob(os.path.join(C, "wr_sweep_v4_*.jsonl"))) for l in open(f)]
if WR:
    N["wr"] = {}
    for w in sorted(set(r["w_r"] for r in WR)):
        rs = [r for r in WR if r["w_r"] == w]; t = lambda r: np.array(r["theta_true"]); e = lambda r: np.array(r["est"])
        N["wr"][w] = dict(n=len(rs), AOm=np.median([100 * abs(e(r)[1] - t(r)[1]) / t(r)[1] for r in rs]), MR=np.median([100 * abs(e(r)[3] - t(r)[3]) / t(r)[3] for r in rs]),
                          CP=np.median([100 * abs(CP(e(r)) - CP(t(r))) / CP(t(r)) for r in rs]), Wp=np.median([100 * abs(Wp(e(r)) - Wp(t(r))) / Wp(t(r)) for r in rs]))
    ws = sorted(N["wr"]); ids = sorted(set(r["i"] for r in WR if all(any(x["i"] == r["i"] and x["w_r"] == w for x in WR) for w in ws)))
    if len(ids) >= 5 and len(ws) == 5:
        M = [[100 * abs(CP(np.array(next(x for x in WR if x["i"] == i and x["w_r"] == w)["est"])) - CP(np.array(next(x for x in WR if x["i"] == i)["theta_true"]))) / CP(np.array(next(x for x in WR if x["i"] == i)["theta_true"])) for i in ids] for w in ws]
        N["wr"]["friedman_CP"] = list(friedmanchisquare(*M)); N["wr"]["wilcoxon_extremes_CP"] = float(wilcoxon(M[0], M[-1]).pvalue); N["wr"]["n_complete"] = len(ids)
open(os.path.join(HERE, "tables_v4.tex"), "w").write("\n\n".join(T) + "\n")
json.dump(N, open(os.path.join(HERE, "numbers_v4.json"), "w"), indent=1, default=float)
print(open(os.path.join(HERE, "tables_v4.tex")).read()); print(json.dumps({k: N[k] for k in ("ode", "wall", "sigma_AP_5W", "pop", "se_ratio_LM", "eff_LM", "cov_LM", "eff_PINN_v4", "cov_PINN_v4", "bounds_PINN", "bounds_PINN_v4")}, indent=1, default=float))
print("pinn_v4:", json.dumps(N["pinn_v4"], default=float)); print("fim:", json.dumps(N.get("fim"), default=float))
