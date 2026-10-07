#!/usr/bin/env python3
"""Complement : rapports par athlete a 'manuscrit A+B' sur kappa, lambda_min, volume d'information par direction
(exp((logdet - logdet_AB)/5)), bornes des 5 parametres, avec Wilcoxon apparie ; plus l'exposant median."""
import json, os, numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); R = {}
for f in ("protocoles_v3.jsonl", "protocoles_extra_v5.jsonl"):
    for l in open(os.path.join(HERE, f)):
        r = json.loads(l)
        if r.get("ok"): R[(r["proto"], r["i"])] = r
ref = {i: r for (n, i), r in R.items() if n == "manuscrit A+B"}
protos = sorted(set(k[0] for k in R), key=lambda n: np.median([R[(n, i)]["cond"] for i in range(50) if (n, i) in R]))
def rp(x): p = wilcoxon(np.log(x)).pvalue; return "%.2f (%s)" % (np.median(x), ("p=%.3f" % p) if p >= 0.001 else "p<0.001")
print("%-42s kappa/AB        lam_min/AB      vol/dir        M_O            A_Omax         A_Pmax         M_R            eta        expo" % "design")
for n in protos:
    if n == "manuscrit A+B": continue
    ids = [i for i in range(50) if (n, i) in R and i in ref]
    k = np.array([R[(n, i)]["cond"] / ref[i]["cond"] for i in ids]); lm = np.array([R[(n, i)]["lam_min"] / ref[i]["lam_min"] for i in ids])
    vol = np.array([np.exp((R[(n, i)]["logdet"] - ref[i]["logdet"]) / 5) for i in ids])
    cr = [np.array([R[(n, i)]["crlb"][q] / ref[i]["crlb"][q] for i in ids]) for q in range(5)]
    print("%-42s %-15s %-15s %-14s %s  %6.2f" % (n, rp(k), rp(lm), rp(vol), "  ".join("%-13s" % rp(c) for c in cr), np.median([R[(n, i)]["expo"] for i in ids])))
