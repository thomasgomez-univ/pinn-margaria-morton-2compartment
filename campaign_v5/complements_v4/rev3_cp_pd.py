#!/usr/bin/env python3
"""Rev.3 (R1 M1, R2 R5) : borne de Cramer-Rao sur la CP et le W' qu'un test puissance-duree renverrait.
Pour chaque athlete (sigma_P = 5 W, information de Fisher de la campagne, bruit blanc) : temps d'epuisement du modele
a 105, 110, 125 et 150 % de la CP vraie (puissances fixees en watts), ajustement lineaire P = CP_PD + W'_PD / T_lim ;
gradient de ln CP_PD et ln W'_PD sur ln theta par differences centrees ; borne = sqrt(g^T F^-1 g).
Usage : python3 rev3_cp_pd.py [shard nshards] ; synthese : python3 rev3_cp_pd.py --analyse"""
import json, os, sys, glob, time, numpy as np
from common_v4 import p3, load_v3, data_like_campaign, sens_matrix, fim_summary, HERE
FR = np.array([1.05, 1.10, 1.25, 1.50])
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def pd_fit(th, Pw):
    T = []
    for p in Pw:
        _, Te = p3.simulate(np.asarray(th, float), (lambda t, p=p: p), 2.0e5); T.append(Te)
    T = np.array(T); A = np.vstack([np.ones_like(T), 1.0 / T]).T
    (c, w), *_ = np.linalg.lstsq(A, Pw, rcond=None); return c, w, T
if "--analyse" in sys.argv:
    R = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "rev3_cp_pd_*.jsonl"))) for l in open(f)]
    print("athletes", len(R))
    for k in ["ratio_CP", "ratio_Wp", "crlb_CP_model", "crlb_CP_PD", "crlb_Wp_model", "crlb_Wp_PD", "proj_CP_PD", "proj_Wp_PD"]:
        v = np.array([r[k] for r in R]); print("%-14s mediane %.4g  IQR [%.4g, %.4g]  min %.4g max %.4g" % (k, np.median(v), *np.percentile(v, [25, 75]), v.min(), v.max()))
    sys.exit()
sh, ns = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
V3 = load_v3(); OUT = os.path.join(HERE, "rev3_cp_pd_%d.jsonl" % sh)
done = {json.loads(l)["i"] for l in open(OUT)} if os.path.exists(OUT) else set()
p3.set_tol(1e-10); t0 = time.time()
for i in [i for i in range(50) if i % ns == sh and i not in done]:
    th, data, sg = data_like_campaign(V3, i, 5.0)
    S = sens_matrix(th, data); F = S.T @ S; Fi = np.linalg.inv(F); summ = fim_summary(F, th)
    p3.set_tol(1e-10)
    Pw = FR * cp(th); c0, w0, T0 = pd_fit(th, Pw)
    g_c, g_w = np.zeros(5), np.zeros(5)
    for k in range(5):
        tp, tm = th.copy(), th.copy(); h = 1e-4 * th[k]; tp[k] += h; tm[k] -= h
        cpp, wpp, _ = pd_fit(tp, Pw); cpm, wpm, _ = pd_fit(tm, Pw)
        g_c[k] = (np.log(cpp) - np.log(cpm)) / (2 * h) * th[k]; g_w[k] = (np.log(wpp) - np.log(wpm)) / (2 * h) * th[k]
    w, V = np.linalg.eigh(F); u = V[:, 0]
    rec = dict(i=i, theta=th.tolist(), T_lim=T0.tolist(), CP=cp(th), Wp=th[2] * th[4], CP_PD=c0, Wp_PD=w0,
               ratio_CP=c0 / cp(th), ratio_Wp=w0 / (th[2] * th[4]), g_CP_PD=g_c.tolist(), g_Wp_PD=g_w.tolist(),
               crlb_CP_model=summ["crlb_CP"], crlb_Wp_model=summ["crlb_Wp"],
               crlb_CP_PD=float(100 * np.sqrt(g_c @ Fi @ g_c)), crlb_Wp_PD=float(100 * np.sqrt(g_w @ Fi @ g_w)),
               proj_CP_PD=float(abs(g_c @ u) / np.linalg.norm(g_c)), proj_Wp_PD=float(abs(g_w @ u) / np.linalg.norm(g_w)))
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print("i=%2d CP_PD/CP %.3f W'_PD/W' %.3f  borne CP %.3f -> %.3f %%  W' %.3f -> %.3f %%  (%.0f s)" % (i, rec["ratio_CP"], rec["ratio_Wp"], rec["crlb_CP_model"], rec["crlb_CP_PD"], rec["crlb_Wp_model"], rec["crlb_Wp_PD"], time.time() - t0), flush=True)
