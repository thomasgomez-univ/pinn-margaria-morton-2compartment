#!/usr/bin/env python3
"""M-10 : sous bruit AR(1) (phi = 0,8, memes donnees que misspec_v4 AR1), ajustement par moindres carres generalises
(residus blanchis par le facteur de Cholesky de Sigma^-1 : z_1 = r_1, z_t = (r_t - phi r_{t-1}) / sqrt(1 - phi^2),
par protocole), intervalles asymptotiques du jacobien blanchi, et borne de Cramer-Rao GLS F = S^T Sigma^-1 S.
Compare a l'ajustement OLS deja enregistre (misspec_v4_AR1_*.jsonl). 20 athletes, sigma_P = 5 W.
Usage : python3 gls_ar1_v5.py [shard nshards]   -> gls_ar1_v5_<shard>.jsonl ; synthese : python3 gls_ar1_v5.py --analyse"""
import json, os, sys, time, glob, numpy as np
from scipy.optimize import least_squares
from common_v4 import p3, load_v3, data_like_campaign, BUDGET, HERE
PHI, SIG, N = 0.8, 5.0, 20

def gen_ar1(V3, i, s=SIG):
    th, data, sg = data_like_campaign(V3, i, s); rng = np.random.default_rng(1000 * i + int(s * 10) + 777)
    for kind, d in data["protocols"].items():
        grid = np.asarray(d["t"]); sA = d["sigma_AP"]; n = len(grid); e = np.empty(n); e[0] = rng.normal(0, sA)
        for k in range(1, n): e[k] = PHI * e[k - 1] + rng.normal(0, sA * np.sqrt(1 - PHI ** 2))
        d["A_P_obs"] = (np.asarray(d["A_P_clean"]) + e).tolist()
    return th, data, sg

def whiten(r):
    z = np.empty_like(r); z[0] = r[0]; z[1:] = (r[1:] - PHI * r[:-1]) / np.sqrt(1 - PHI ** 2); return z

def residuals_gls(theta, data):
    out = []
    for kind, d in data["protocols"].items():
        tf = float(d["t"][-1]) * (1 + 1e-9)
        pred, _ = p3.simulate(np.asarray(theta), d["_P"], tf, np.asarray(d["t"]), switches=d["_sw"])
        out.append(whiten((pred - np.asarray(d["A_P_obs"])) / d["sigma_AP"]))
    return np.concatenate(out)

def crlb_gls(theta, data, rel_step=1e-4):
    keep = p3._TOL[0]; p3.set_tol(1e-10)
    try:
        blocks = []
        for kind, d in data["protocols"].items():
            grid = np.asarray(d["t"]); tf = float(grid[-1]) * (1 + 1e-9); S = np.zeros((len(grid), 5))
            for k in range(5):
                tp, tm = np.array(theta, float), np.array(theta, float); h = rel_step * abs(theta[k]); tp[k] += h; tm[k] -= h
                yp, _ = p3.simulate(tp, d["_P"], tf, grid, switches=d["_sw"]); ym, _ = p3.simulate(tm, d["_P"], tf, grid, switches=d["_sw"])
                S[:, k] = whiten((yp - ym) / (2 * h) * theta[k] / d["sigma_AP"])
            blocks.append(S)
        S = np.vstack(blocks); F = S.T @ S
        return 100 * np.sqrt(np.diag(np.linalg.inv(F)))
    finally: p3.set_tol(keep)

if "--analyse" in sys.argv:
    def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
    def wp(t): return t[2] * t[4]
    G = {json.loads(l)["i"]: json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "gls_ar1_v5_*.jsonl"))) for l in open(f)}
    O = {json.loads(l)["i"]: json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "misspec_v4_AR1_*.jsonl"))) for l in open(f)}
    ids = sorted(set(G) & set(O)); print("athletes :", len(ids))
    rows = []
    for name, R, key, se_key, cr_key in (("OLS (misspec AR1)", O, "LM", "LM_se", "crlb_pct"), ("GLS", G, "GLS", "GLS_se", "crlb_gls_pct")):
        th = np.array([R[i]["theta_true"] for i in ids]); x = np.array([R[i]["est"][key] if isinstance(R[i]["est"], dict) else R[i]["est"] for i in ids])
        se = np.array([R[i][se_key] for i in ids]); cr = np.array([R[i][cr_key] for i in ids])
        err = 100 * np.abs(x / th - 1); eff = np.median(err / cr, axis=0) / 0.6745; cov = 100 * np.mean(np.abs(x - th) <= 1.96 * se, axis=0)
        ecp = np.median([100 * abs(cp(a) / cp(b) - 1) for a, b in zip(x, th)]); ewp = np.median([100 * abs(wp(a) / wp(b) - 1) for a, b in zip(x, th)])
        print("%-20s CP %.2f  W' %.2f  par %s  borne %s  eff %s  couverture %s" % (name, ecp, ewp, np.round(np.median(err, 0), 2), np.round(np.median(cr, 0), 2), np.round(eff, 2), np.round(cov, 0)))
    sys.exit(0)

shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0; nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT = os.path.join(HERE, "gls_ar1_v5_%d.jsonl" % shard); done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)["i"])
V3 = load_v3(); cfg = p3.Config(budget_ode=BUDGET); t0 = time.time()
for i in [i for i in range(N) if i % nsh == shard and i not in done]:
    th, data, sg = gen_ar1(V3, i); seed = 1000 * i + 50
    orig = p3.residuals; p3.residuals = residuals_gls
    try:
        x, n = p3.fit_LM(data, cfg, seed)
        r = least_squares(residuals_gls, x, args=(data,), bounds=(p3.PARAM_LB, p3.PARAM_UB), method="trf", max_nfev=200)
        J = r.jac; dof = max(len(r.fun) - 5, 1); s2 = float(np.sum(r.fun ** 2) / dof)
        se = np.sqrt(np.clip(np.diag(s2 * np.linalg.inv(J.T @ J)), 0, None))
    finally: p3.residuals = orig
    rec = dict(i=i, case="AR1_GLS", phi=PHI, sigma_P=SIG, theta_true=th.tolist(), est={"GLS": r.x.tolist()}, GLS_se=se.tolist(),
               crlb_gls_pct=crlb_gls(th, data).tolist(), ode_calls=int(n), wall_s=round(time.time() - t0, 1), s2=s2)
    open(OUT, "a").write(json.dumps(rec) + "\n"); print("i=%2d GLS fait (%.0f s)" % (i, time.time() - t0), flush=True)
print("TERMINE gls_ar1_v5 shard", shard)
