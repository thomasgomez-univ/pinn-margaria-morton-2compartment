#!/usr/bin/env python3
"""Rev.3 (R1 M6, R2 R7) : moindres carres generalises FAISABLES sous bruit AR(1), phi estime sur les residus.
Donnees : identiques a misspec_v4 AR1 (meme graine 1000 i + 50 + 777, meme sigma_AP), phi vrai = --phi.
Etape 1 : estimation OLS (relue dans misspec_v4_AR1_*<tagols>.jsonl si disponible, sinon LM multi-depart de la campagne).
Etape 2 : phi_hat = autocorrelation de rang 1 des residus normalises, regroupee sur les protocoles (estimateur de Yule-Walker).
Etape 3 : ajustement GLS (residus blanchis avec phi_hat), depart = estimation precedente ; etapes 2-3 iterees deux fois
(Cochrane-Orcutt itere). Intervalles : jacobien blanchi. Bornes : F = S^T Sigma^-1 S avec le phi VRAI (reference).
Usage : python3 rev3_fgls.py [shard nshards] [--phi 0.8] [--n 20] [--tagols ""] [--tag ""] ; synthese : --analyse --tag ..."""
import json, os, sys, glob, time, argparse, numpy as np
from scipy.optimize import least_squares
from common_v4 import p3, load_v3, data_like_campaign, BUDGET, HERE
ap = argparse.ArgumentParser(); ap.add_argument("shard", nargs="?", type=int, default=0); ap.add_argument("nshards", nargs="?", type=int, default=1)
ap.add_argument("--phi", type=float, default=0.8); ap.add_argument("--n", type=int, default=20); ap.add_argument("--tagols", default="")
ap.add_argument("--tag", default=""); ap.add_argument("--analyse", action="store_true"); a = ap.parse_args()
SIG = 5.0
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def wp(t): return t[2] * t[4]

if a.analyse:
    from scipy.stats import beta
    def cpi(k, n): return (0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(0.975, k + 1, n - k))
    R = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "rev3_fgls_*%s.jsonl" % a.tag))) for l in open(f)]
    R = [r for r in R if abs(r["phi"] - a.phi) < 1e-9]; print("athletes", len(R), "phi vrai", a.phi)
    ph = np.array([r["phi_hat"][-1] for r in R]); print("phi_hat mediane %.3f IQR [%.3f, %.3f]" % (np.median(ph), *np.percentile(ph, [25, 75])))
    for key, sek in (("OLS", "OLS_se"), ("FGLS", "FGLS_se")):
        th = np.array([r["theta_true"] for r in R]); x = np.array([r["est"][key] for r in R]); se = np.array([r[sek] for r in R])
        cr = np.array([r["crlb_gls_true_pct"] for r in R]); err = 100 * np.abs(x / th - 1)
        k = np.sum(np.abs(x - th) <= 1.96 * se, axis=0); n = len(R)
        print("%-5s erreur med CP %.2f W' %.2f ; par param %s ; borne GLS %s ; eff %s ; couverture IC jacobien %s"
              % (key, np.median([100 * abs(cp(u) / cp(v) - 1) for u, v in zip(x, th)]), np.median([100 * abs(wp(u) / wp(v) - 1) for u, v in zip(x, th)]),
                 np.round(np.median(err, 0), 2), np.round(np.median(cr, 0), 2), np.round(np.median(err / cr, 0) / 0.6745, 2),
                 ["%d/%d [%.0f-%.0f]" % (kk, n, 100 * cpi(kk, n)[0], 100 * cpi(kk, n)[1]) for kk in k]))
    sys.exit()

def gen(V3, i, phi):
    th, data, sg = data_like_campaign(V3, i, SIG); rng = np.random.default_rng(1000 * i + int(SIG * 10) + 777)
    for kind, d in data["protocols"].items():
        grid = np.asarray(d["t"]); sA = d["sigma_AP"]; n = len(grid); e = np.empty(n); e[0] = rng.normal(0, sA)
        for k in range(1, n): e[k] = phi * e[k - 1] + rng.normal(0, sA * np.sqrt(1 - phi ** 2))
        d["A_P_obs"] = (np.asarray(d["A_P_clean"]) + e).tolist()
    return th, data
def raw_res(theta, data):
    out = {}
    for kind, d in data["protocols"].items():
        tf = float(d["t"][-1]) * (1 + 1e-9)
        pred, _ = p3.simulate(np.asarray(theta, float), d["_P"], tf, np.asarray(d["t"]), switches=d["_sw"])
        out[kind] = (pred - np.asarray(d["A_P_obs"])) / d["sigma_AP"]
    return out
def whiten(r, phi):
    z = np.empty_like(r); z[0] = r[0]; z[1:] = (r[1:] - phi * r[:-1]) / np.sqrt(1 - phi ** 2); return z
def res_gls(theta, data, phi): return np.concatenate([whiten(r, phi) for r in raw_res(theta, data).values()])
def phi_hat(theta, data):
    R = raw_res(theta, data); num = sum(float(np.dot(r[1:], r[:-1])) for r in R.values()); den = sum(float(np.dot(r[:-1], r[:-1])) for r in R.values())
    return float(np.clip(num / den, -0.99, 0.99))
def fit_from(x0, data, phi):
    r = least_squares(res_gls, x0, args=(data, phi), bounds=(p3.PARAM_LB, p3.PARAM_UB), method="trf", max_nfev=200)
    dof = max(len(r.fun) - 5, 1); s2 = float(np.sum(r.fun ** 2) / dof)
    try: se = np.sqrt(np.clip(np.diag(s2 * np.linalg.inv(r.jac.T @ r.jac)), 0, None))
    except np.linalg.LinAlgError: se = np.full(5, np.nan)
    return r.x, se, int(r.nfev)
def crlb_gls(theta, data, phi, rel_step=1e-4):
    keep = p3._TOL[0]; p3.set_tol(1e-10)
    try:
        blocks = []
        for kind, d in data["protocols"].items():
            grid = np.asarray(d["t"]); tf = float(grid[-1]) * (1 + 1e-9); S = np.zeros((len(grid), 5))
            for k in range(5):
                tp, tm = np.array(theta, float), np.array(theta, float); h = rel_step * abs(theta[k]); tp[k] += h; tm[k] -= h
                yp, _ = p3.simulate(tp, d["_P"], tf, grid, switches=d["_sw"]); ym, _ = p3.simulate(tm, d["_P"], tf, grid, switches=d["_sw"])
                S[:, k] = whiten((yp - ym) / (2 * h) * theta[k] / d["sigma_AP"], phi)
            blocks.append(S)
        S = np.vstack(blocks); return (100 * np.sqrt(np.diag(np.linalg.inv(S.T @ S)))).tolist()
    finally: p3.set_tol(keep)

V3 = load_v3(); cfg = p3.Config(budget_ode=BUDGET)
OLS = {}
for f in glob.glob(os.path.join(HERE, "misspec_v4_AR1_[0-9]%s.jsonl" % a.tagols)):
    for l in open(f):
        r = json.loads(l)
        if abs(r["perturb"]["A"]["phi"] - a.phi) < 1e-9 and "LM" in r["est"]: OLS[r["i"]] = r
OUT = os.path.join(HERE, "rev3_fgls_%d%s.jsonl" % (a.shard, a.tag))
done = {(json.loads(l)["i"], json.loads(l)["phi"]) for l in open(OUT)} if os.path.exists(OUT) else set()
t0 = time.time()
for i in [i for i in range(a.n) if i % a.nshards == a.shard and (i, a.phi) not in done]:
    th, data = gen(V3, i, a.phi); seed = 1000 * i + 50
    if i in OLS:
        x_ols, se_ols, src = np.array(OLS[i]["est"]["LM"]), np.array(OLS[i]["LM_se"]), "fichier"
        assert np.allclose(OLS[i]["theta_true"], th)
    else:
        x_ols, se_ols, _ = p3.fit_LM_with_CI(data, cfg, seed); src = "LM"
    x, phs, nf = np.array(x_ols, float), [], 0
    for it in range(2):
        ph = phi_hat(x, data); phs.append(ph); x, se, n_ = fit_from(x, data, ph); nf += n_
    rec = dict(i=i, phi=a.phi, theta_true=th.tolist(), ols_source=src, phi_hat=phs, est={"OLS": np.asarray(x_ols).tolist(), "FGLS": x.tolist()},
               OLS_se=np.asarray(se_ols).tolist(), FGLS_se=se.tolist(), crlb_gls_true_pct=crlb_gls(th, data, a.phi), nfev_fgls=nf)
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print("phi %.2f i=%2d phi_hat %s  CP OLS %.2f%% FGLS %.2f%%  (%.0f s)" % (a.phi, i, np.round(phs, 3), 100 * abs(cp(x_ols) / cp(th) - 1), 100 * abs(cp(x) / cp(th) - 1), time.time() - t0), flush=True)
print("TERMINE fgls phi", a.phi, "shard", a.shard)
