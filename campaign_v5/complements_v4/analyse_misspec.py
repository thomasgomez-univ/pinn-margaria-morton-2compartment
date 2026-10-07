#!/usr/bin/env python3
"""Synthese du controle de mauvaise specification : misspec_v4_{AR1,DRIFT}_*.jsonl
contre la reference de la campagne (camp_v3 pour LM/DE, camp_v4 pour PINN v4),
memes athletes, sigma_P = 5 W. Ecrit misspec_v4_summary.json et un tableau LaTeX."""
import json, glob, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["M_O", "A_Omax", "A_Pmax", "M_R", "eta"]
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def wp(t): return t[2] * t[4]
def load(pat):
    R = {}
    for f in sorted(glob.glob(os.path.join(HERE, pat)) + glob.glob(os.path.join(HERE, "..", pat))):
        for l in open(f):
            r = json.loads(l)
            if isinstance(r.get("est"), list):            # camp_v4 : une ligne par (i, sigma, seed_k, method)
                if r.get("seed_k", 0) != 0: continue
                r = dict(r, est={r["method"]: r["est"]})
            if abs(r.get("sigma_P", 5.0) - 5.0) < 1e-9: R[r["i"]] = r
    return R
def errs(r, m):
    th, x = np.array(r["theta_true"]), np.array(r["est"][m])
    return dict(CP=100 * abs(cp(x) / cp(th) - 1), Wp=100 * abs(wp(x) / wp(th) - 1),
                par=100 * np.abs(x / th - 1), signed=100 * (x / th - 1))
def stats(R, m, label):
    ids = sorted(R); e = [errs(R[i], m) for i in ids]
    q = lambda v: [float(x) for x in np.percentile(v, [25, 75])]
    out = dict(label=label, n=len(ids), CP=float(np.median([d["CP"] for d in e])), Wp=float(np.median([d["Wp"] for d in e])),
               CP_iqr=q([d["CP"] for d in e]), Wp_iqr=q([d["Wp"] for d in e]),
               par=np.median([d["par"] for d in e], axis=0).tolist(),
               bias=np.median([d["signed"] for d in e], axis=0).tolist())
    cr = np.array([R[i]["crlb_pct"] for i in ids]); pe = np.array([d["par"] for d in e])
    out["eff"] = (np.median(pe / cr, axis=0) / 0.6745).tolist()
    if m == "LM" and all("LM_se" in R[i] for i in ids):
        cov = []
        for i in ids:
            th, x, se = np.array(R[i]["theta_true"]), np.array(R[i]["est"]["LM"]), np.array(R[i]["LM_se"])
            cov.append(np.abs(x - th) <= 1.96 * se)
        out["coverage"] = (100 * np.mean(cov, axis=0)).tolist()
    if "rmse" in R[ids[0]]:
        fl = [np.sqrt(np.mean([v ** 2 for v in R[i]["sigma_AP"].values()])) for i in ids]   # plancher ~ RMS des sigma_AP
        out["misfit_ratio"] = float(np.median([R[i]["rmse"][m] / f for i, f in zip(ids, fl)]))
    return out
S = {}
V3 = load("camp_v3_*.jsonl"); V4 = load("camp_v4_*.jsonl")
for case in ("AR1", "DRIFT"):
    R = load("misspec_v4_%s_*.jsonl" % case)
    if not R: continue
    ids = sorted(R); S[case] = {}
    for m in R[ids[0]]["est"]:
        S[case][m] = stats(R, m, "%s / %s" % (case, m))
        ref = {i: V4[i] for i in ids if i in V4} if m == "PINN_v4" else {i: V3[i] for i in ids if i in V3}
        rm = "PINN_v4" if m == "PINN_v4" else m
        if ref and all(rm in ref[i]["est"] for i in ref):
            S[case][m + "_ref"] = stats(ref, rm, "reference / %s" % m)
json.dump(S, open(os.path.join(HERE, "misspec_v4_summary.json"), "w"), indent=1)
rows = []
for case in S:
    for m in [k for k in S[case] if not k.endswith("_ref")]:
        for key, tag in ((m + "_ref", "exact model"), (m, case)):
            if key not in S[case]: continue
            s = S[case][key]
            rows.append("%s & %s & %.2f & %.2f & %s & %s & %s \\\\" % (
                m.replace("_", " "), tag, s["CP"], s["Wp"], " / ".join("%.1f" % v for v in s["par"]),
                " / ".join("%.1f" % v for v in s["eff"]),
                ("%d/%d/%d/%d/%d" % tuple(round(c) for c in s["coverage"])) if "coverage" in s else "--"))
    print()
open(os.path.join(HERE, "misspec_v4_table.tex"), "w").write("\n".join(rows) + "\n")
for case in S:
    for k, s in S[case].items():
        print("%-28s n=%2d  CP %.2f [%.2f-%.2f]  W' %.2f [%.2f-%.2f]  par %s  eff %s  cov %s  misfit %s" % (
            s["label"], s["n"], s["CP"], *s["CP_iqr"], s["Wp"], *s["Wp_iqr"], np.round(s["par"], 2), np.round(s["eff"], 2),
            np.round(s.get("coverage", [np.nan]), 0), ("%.2f" % s["misfit_ratio"]) if "misfit_ratio" in s else "--"))
