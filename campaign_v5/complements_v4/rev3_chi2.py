#!/usr/bin/env python3
"""Rev.3 bis (R2) : test de detection d'un ecart au modele par le residu. Sous le modele correct, bruit blanc gaussien de
sigma_AP connu et estimation par moindres carres, S = sum (r_t / sigma)^2 ~ chi2(n - p), n = 120 observations, p = 5.
Le misfit m des sorties misspec_v4 est RMS(residu) / RMS(sigma_AP) ; on approche S = n m^2 (les deux sigma_AP d'un athlete
different de moins de 5 %). Taux de rejet a alpha = 0.05 et 0.01 par cas. Reference : campagne (modele correct) via le
misfit des ajustements TRF si disponible, sinon la loi chi2 elle-meme."""
import json, glob, os, numpy as np
from scipy.stats import chi2
HERE = os.path.dirname(os.path.abspath(__file__)); n, p = 120, 5; dof = n - p
def rate(ms, alpha):
    S = n * np.asarray(ms) ** 2; pv = chi2.sf(S, dof); return int(np.sum(pv < alpha)), len(ms), float(np.median(pv))
print("seuil : misfit > %.3f pour alpha = 0.05, > %.3f pour alpha = 0.01" % (np.sqrt(chi2.ppf(0.95, dof) / n), np.sqrt(chi2.ppf(0.99, dof) / n)))
out = {}
for lab, pat in [("AR1 phi=0.3", "misspec_v4_AR1_0_phi3"), ("AR1 phi=0.6", "misspec_v4_AR1_0_phi6"), ("AR1 phi=0.8", "misspec_v4_AR1_[0-9]"), ("AR1 phi=0.95", "misspec_v4_AR1_0_phi95"),
                 ("drift 2.5%", "misspec_v4_DRIFT_0_d0025"), ("drift 5%", "misspec_v4_DRIFT_[0-9]"), ("drift 10%", "misspec_v4_DRIFT_0_d010")]:
    R = [json.loads(l) for f in glob.glob(os.path.join(HERE, pat + ".jsonl")) for l in open(f)]
    ms = [r["rmse"]["LM"] / np.sqrt(np.mean(np.square(list(r["sigma_AP"].values())))) for r in R]
    k5, N, pv = rate(ms, 0.05); k1, _, _ = rate(ms, 0.01)
    out[lab] = dict(n=N, misfit_median=float(np.median(ms)), misfit_min=float(np.min(ms)), reject_05=k5, reject_01=k1, p_median=pv)
    print("%-12s n=%2d misfit med %.3f min %.3f  rejet 5%% : %2d/%d  rejet 1%% : %2d/%d  p median %.2g" % (lab, N, np.median(ms), np.min(ms), k5, N, k1, N, pv))
json.dump(out, open(os.path.join(HERE, "rev3_chi2.json"), "w"), indent=1)
