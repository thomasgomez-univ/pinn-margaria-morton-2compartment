#!/usr/bin/env python3
"""M-19 / M-14 : variabilite du reseau corrige entre graines d'entrainement, 50 athletes, sigma_P = 5 W (camp_v4_seeds_*.jsonl,
seed_k 0 = campagne, 1 et 2 = run_rev2). Dedoublonne par (i, seed_k)."""
import json, glob, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def wp(t): return t[2] * t[4]
R = {}
for f in sorted(glob.glob(os.path.join(HERE, "..", "camp_v4_seeds_*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if r["sigma_P"] == 5.0: R[(r["i"], r["seed_k"])] = r
seeds = sorted(set(k for i, k in R)); ids = sorted(set(i for i, k in R if all((i, kk) in R for kk in seeds)))
print("graines %s, athletes complets %d" % (seeds, len(ids)))
E = {k: np.array([[100 * abs(cp(R[(i, k)]["est"]) / cp(R[(i, k)]["theta_true"]) - 1), 100 * abs(wp(R[(i, k)]["est"]) / wp(R[(i, k)]["theta_true"]) - 1)] for i in ids]) for k in seeds}
for k in seeds: print("graine %d : CP mediane %.2f [IQR %.2f-%.2f]  W' %.2f [%.2f-%.2f]" % (k, np.median(E[k][:, 0]), *np.percentile(E[k][:, 0], [25, 75]), np.median(E[k][:, 1]), *np.percentile(E[k][:, 1], [25, 75])))
A = np.stack([E[k] for k in seeds]); sd = np.std(A, axis=0, ddof=1)
print("ecart-type inter-graines par athlete : CP mediane %.2f pt (max %.2f) ; W' %.2f (max %.2f)" % (np.median(sd[:, 0]), sd[:, 0].max(), np.median(sd[:, 1]), sd[:, 1].max()))
med = [np.median(E[k][:, 0]) for k in seeds]; print("medianes CP par graine %s ; ecart-type %.3f pt ; mediane des medianes par athlete %.2f" % (np.round(med, 3).tolist(), np.std(med, ddof=1), np.median(np.median(A[:, :, 0], axis=0))))
n_bim = int(np.sum(sd[:, 0] > 1.0)); print("athletes dont l'ecart-type inter-graines sur CP depasse 1 point : %d / %d" % (n_bim, len(ids)))
