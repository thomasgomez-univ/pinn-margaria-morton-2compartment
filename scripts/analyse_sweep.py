#!/usr/bin/env python3
"""
Analyse d'un balayage produit par run_sweep.py.

Affiche, pour chaque valeur balayee, l'erreur relative mediane sur les
quantites derivees (CP, W') et sur les cinq parametres, puis teste l'effet
de l'hyperparametre par un Friedman apparie (toutes les conditions) et un
Wilcoxon apparie entre les deux extremes. Seuls les athletes presents dans
TOUTES les conditions entrent dans les tests, pour que l'appariement soit
exact.

    python3 analyse_sweep.py e5_wr_full.jsonl
    python3 analyse_sweep.py e5_wr_full.jsonl --baseline 7.95
"""
import argparse, json
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

NAMES = ["M_O", "A_Omax", "A_Pmax", "M_R", "eta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="fichiers JSONL a cumuler")
    ap.add_argument("--baseline", type=float, default=7.95,
                    help="erreur CP mediane de la baseline de population (%%)")
    a = ap.parse_args()

    rows = []
    for fn in a.files:
        for line in open(fn):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # compatibilite avec l'ancien format (cle "w_r" au lieu de param/value)
    for r in rows:
        if "value" not in r:
            r["param"], r["value"] = "w_r", r["w_r"]

    param = rows[0]["param"]
    values = sorted({r["value"] for r in rows})
    E = {(r["i"], r["value"]): r for r in rows}
    complete = sorted({i for (i, _) in E}
                      - {i for (i, v) in
                         [(i, v) for i in {i for (i, _) in E} for v in values]
                         if (i, v) not in E})

    def err(r):
        th, x = np.array(r["theta_true"]), np.array(r["est"])
        cp = lambda t: t[0] * t[4]
        wp = lambda t: t[2] * t[4]
        return (100 * abs(cp(x) - cp(th)) / cp(th),
                100 * abs(wp(x) - wp(th)) / wp(th),
                100 * np.abs(x - th) / th)

    print("n = %d athletes complets sur %d presents, parametre balaye : %s\n"
          % (len(complete), len({i for (i, _) in E}), param))
    hdr = "%-10s %8s %8s | %s" % (param, "CP (%)", "W' (%)",
                                  "  ".join("%8s" % n for n in NAMES))
    print(hdr); print("-" * len(hdr))
    CP = {}
    for v in values:
        c, w, p = zip(*[err(E[(i, v)]) for i in complete])
        CP[v] = np.array(c)
        print("%-10s %8.2f %8.2f | %s"
              % (v, np.median(c), np.median(w),
                 "  ".join("%8.2f" % x for x in np.median(np.array(p), axis=0))))

    if len(values) > 2 and len(complete) >= 3:
        st, pv = friedmanchisquare(*[CP[v] for v in values])
        print("\nFriedman (erreur CP, %d conditions appariees) : chi2 = %.2f, p = %.4f"
              % (len(values), st, pv))
    if len(complete) >= 6:
        _, p2 = wilcoxon(CP[values[0]], CP[values[-1]])
        print("Wilcoxon %s=%s vs %s=%s : p = %.4f"
              % (param, values[0], param, values[-1], p2))

    med = [np.median(CP[v]) for v in values]
    print("\nmedianes CP : %.2f a %.2f %% sur la plage balayee (rapport %.2f)"
          % (min(med), max(med), max(med) / min(med)))
    print("baseline de population : %.2f %%" % a.baseline)
    for v in values:
        print("  %s=%-6s : bat la baseline pour %d/%d athletes"
              % (param, v, int((CP[v] < a.baseline).sum()), len(complete)))


if __name__ == "__main__":
    main()
