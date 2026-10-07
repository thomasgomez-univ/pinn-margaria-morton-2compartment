#!/usr/bin/env python3
"""Compare chaque design (catalogue + extra) a 'manuscrit A+B' sur les memes athletes : medianes de kappa,
bornes, exposant ; rapport par athlete et Wilcoxon apparie sur les bornes A_Omax et M_R."""
import json, glob, os, numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); R = {}
for f in ("protocoles_v3.jsonl", "protocoles_extra_v5.jsonl"):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r.get("ok"): R[(r["proto"], r["i"])] = r
protos = sorted(set(k[0] for k in R), key=lambda n: (n != "manuscrit A+B", n))
ref = {i: r for (n, i), r in R.items() if n == "manuscrit A+B"}
print("%-42s n  kappa(med)  CRLB M_O/A_Omax/A_Pmax/M_R/eta (med %%)   expo   ratio A_Omax, M_R vs A+B (p)" % "design")
for n in protos:
    rows = [R[(n, i)] for i in range(50) if (n, i) in R]; ids = [i for i in range(50) if (n, i) in R and i in ref]
    cr = np.median([r["crlb"] for r in rows], axis=0); kap = np.median([r["cond"] for r in rows]); ex = np.median([r["expo"] for r in rows])
    line = "%-42s %2d  %9.3g  %5.2f/%5.2f/%5.2f/%5.2f/%5.2f  %6.2f" % (n, len(rows), kap, *cr, ex)
    if n != "manuscrit A+B" and len(ids) >= 5:
        ra = np.array([R[(n, i)]["crlb"][1] / ref[i]["crlb"][1] for i in ids]); rm = np.array([R[(n, i)]["crlb"][3] / ref[i]["crlb"][3] for i in ids])
        pa = wilcoxon(ra - 1).pvalue if len(ids) > 5 else float("nan"); pm = wilcoxon(rm - 1).pvalue if len(ids) > 5 else float("nan")
        line += "   %.2f (p=%.3f), %.2f (p=%.3f)" % (np.median(ra), pa, np.median(rm), pm)
    print(line)
