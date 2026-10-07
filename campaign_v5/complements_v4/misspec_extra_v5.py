#!/usr/bin/env python3
"""Complements au controle de mauvaise specification (v5) : misfit (mediane, IQR, fraction > 1.1),
comparaisons appariees entre estimateurs (Wilcoxon), erreur signee sur eta, correlation des erreurs log A_Omax / M_R."""
import json, glob, os, numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def wp(t): return t[2] * t[4]
def load(pat):
    R = {}
    for f in sorted(glob.glob(os.path.join(HERE, pat))):
        for l in open(f):
            r = json.loads(l); R[r["i"]] = r
    return R
for case in ("AR1", "DRIFT"):
    R = load("misspec_v4_%s_*.jsonl" % case); ids = sorted(R); print("=== %s  n=%d" % (case, len(ids)))
    E = {}
    for m in ("LM", "DE", "PINN_v4"):
        th = np.array([R[i]["theta_true"] for i in ids]); x = np.array([R[i]["est"][m] for i in ids])
        E[m] = dict(CP=100 * np.abs([cp(a) / cp(b) - 1 for a, b in zip(x, th)]), Wp=100 * np.abs([wp(a) / wp(b) - 1 for a, b in zip(x, th)]),
                    signed=100 * (x / th - 1), logerr=np.log(x / th))
        fl = np.array([np.sqrt(np.mean([v ** 2 for v in R[i]["sigma_AP"].values()])) for i in ids]); mr = np.array([R[i]["rmse"][m] for i in ids]) / fl
        print("  %-8s misfit/plancher mediane %.2f IQR [%.2f, %.2f]  n>1.1 : %d/%d" % (m, np.median(mr), *np.percentile(mr, [25, 75]), int((mr > 1.1).sum()), len(ids)))
        print("           erreur signee mediane (%%) :", np.round(np.median(E[m]["signed"], 0), 1).tolist(),
              "  corr(ln err A_Omax, ln err M_R) = %.2f" % np.corrcoef(E[m]["logerr"][:, 1], E[m]["logerr"][:, 3])[0, 1])
    for q in ("CP", "Wp"):
        for other in ("PINN_v4", "DE"):
            d = E[other][q] - E["LM"][q]; nb = int((d > 0).sum())
            print("  %s : LM meilleur que %s pour %d/%d athletes, Wilcoxon p = %.3g" % (q, other, nb, len(ids), wilcoxon(E["LM"][q], E[other][q]).pvalue))
