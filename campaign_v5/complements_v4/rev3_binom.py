#!/usr/bin/env python3
"""Rev.3 (R1 m5, R2 R8) : intervalles de confiance exacts (Clopper-Pearson, 95 %) sur les couvertures rapportees.
Couverture : |erreur relative| < 1.96 sigma_CR par parametre (campagne, sigma_P = 5 W, 50 athletes), et IC du jacobien
sous bruit AR(1) (OLS, GLS phi connu, 20 athletes)."""
import json, glob, os, numpy as np
from scipy.stats import beta
HERE = os.path.dirname(os.path.abspath(__file__)); P = ["M_O", "A_Omax", "A_Pmax", "M_R", "eta"]
def cpi(k, n): return (0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(0.975, k + 1, n - k))
def show(name, hits):
    hits = np.asarray(hits); n = len(hits); k = hits.sum(0)
    print("%-28s n=%d  " % (name, n) + "  ".join("%s %d/%d=%.0f%% [%.0f-%.0f]" % (p, kk, n, 100 * kk / n, 100 * cpi(kk, n)[0], 100 * cpi(kk, n)[1]) for p, kk in zip(P, k)))
    return {p: dict(k=int(kk), n=n, lo=cpi(kk, n)[0], hi=cpi(kk, n)[1]) for p, kk in zip(P, k)}
V3 = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "camp_v3_*.jsonl"))) for l in open(f)]
V4 = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "camp_v4_[0-9].jsonl"))) for l in open(f)]
def err(x, th): return 100 * np.abs(np.array(x) / np.array(th) - 1)
out = {}
S = [r for r in V3 if r["sigma_P"] == 5.0]
for m in ["LM", "DE", "PINN"]:
    out[m] = show(m + " (+-1.96 borne)", [err(r["est"][m], r["theta_true"]) < 1.96 * np.array(r["crlb_pct"]) for r in S])
crl = {r["i"]: np.array(r["crlb_pct"]) for r in S}
k0 = [r for r in V4 if r.get("seed_k", 0) == 0 and r["sigma_P"] == 5.0]
out["PINN_v4"] = show("PINN corrige (+-1.96 borne)", [err(r["est"], r["theta_true"]) < 1.96 * crl[r["i"]] for r in k0])
out["LM_jac"] = show("LM IC jacobien (campagne)", [np.abs(np.array(r["est"]["LM"]) - np.array(r["theta_true"])) <= 1.96 * np.array(r["LM_se"]) for r in S if "LM_se" in r])
G = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "gls_ar1_v5_*.jsonl"))) for l in open(f)]
O = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "misspec_v4_AR1_[0-9].jsonl"))) for l in open(f)]
out["OLS_AR1"] = show("OLS sous AR(1) phi=0.8", [np.abs(np.array(r["est"]["LM"]) - np.array(r["theta_true"])) <= 1.96 * np.array(r["LM_se"]) for r in O])
out["GLS_AR1"] = show("GLS phi connu", [np.abs(np.array(r["est"]["GLS"]) - np.array(r["theta_true"])) <= 1.96 * np.array(r["GLS_se"]) for r in G])
D = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "misspec_v4_DRIFT_[0-9].jsonl"))) for l in open(f)]
out["LM_DRIFT"] = show("LM sous derive 5%", [np.abs(np.array(r["est"]["LM"]) - np.array(r["theta_true"])) <= 1.96 * np.array(r["LM_se"]) for r in D])
json.dump(out, open(os.path.join(HERE, "rev3_binom.json"), "w"), indent=1)
