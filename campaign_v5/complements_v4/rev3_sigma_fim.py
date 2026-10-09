#!/usr/bin/env python3
"""Rev.3 (R1 M4) : information de Fisher avec la covariance du bruit induite par l'erreur de capteur.
Pour chaque athlete et protocole (sigma_P = 5 W) : N realisations Monte-Carlo du bruit de puissance (meme mecanisme que la
calibration de la campagne : bruit blanc a 1 s, interpole, ajoute au profil exact), deviations de A_P sur la grille de la
campagne (points anterieurs a l'epuisement le plus precoce), covariance empirique Sigma avec retrait vers la diagonale
(alpha) et pepite nu^2 * sigma_AP^2 I (bruit de mesure residuel ; sans elle, la variance nulle a t = 0 rend Sigma singuliere),
F = sum_p S_p^T Sigma_p^-1 S_p avec S_ik = theta_k dy_i/dtheta_k. Comparaison avec la borne bruit blanc de la campagne.
Usage : python3 rev3_sigma_fim.py [shard nshards] [--n 15] [--mc 400] ; synthese : --analyse"""
import json, os, sys, glob, time, argparse, numpy as np
from common_v4 import p3, load_v3, data_like_campaign, sens_matrix, fim_summary, HERE
ap = argparse.ArgumentParser(); ap.add_argument("shard", nargs="?", type=int, default=0); ap.add_argument("nshards", nargs="?", type=int, default=1)
ap.add_argument("--n", type=int, default=15); ap.add_argument("--mc", type=int, default=400); ap.add_argument("--analyse", action="store_true"); ap.add_argument("--tag", default="")
a = ap.parse_args()
def cp_grad(th):
    M_O, A_Om, A_Pm, M_R, eta = th; x = M_R * A_Om / (M_R * A_Om + M_O); return np.array([x, 1 - x, 0., 1 - x, 1.])
GW = np.array([0., 0., 1., 0., 1.])
COMBOS = [(0.1, 0.05), (0.1, 0.1), (0.1, 0.3), (0.05, 0.1), (0.2, 0.1), (0.0, 0.1)]
if a.analyse:
    R = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "rev3_sigma_fim_*%s.jsonl" % a.tag))) for l in open(f)]
    print("athletes", len(R))
    for key in R[0]["crlb_sigma"]:
        cw = np.array([r["crlb_white"] for r in R]); cs = np.array([r["crlb_sigma"][key] for r in R])
        cpw = np.array([r["cp_white"] for r in R]); cps = np.array([r["cp_sigma"][key] for r in R])
        wpw = np.array([r["wp_white"] for r in R]); wps = np.array([r["wp_sigma"][key] for r in R])
        ex = np.array([r["expo_sigma"][key] for r in R]); kw = np.array([r["cond_white"] for r in R]); ks = np.array([r["cond_sigma"][key] for r in R])
        print("%s  borne blanc %s | Sigma %s ; CP %.3f -> %.3f [%.3f,%.3f] ; W' %.3f -> %.3f [%.3f,%.3f] ; ratio par param %s ; expo Sigma %.2f [%.2f,%.2f] (blanc %.2f) ; cond %.3g -> %.3g"
              % (key, np.round(np.median(cw, 0), 2), np.round(np.median(cs, 0), 2), np.median(cpw), np.median(cps), *np.percentile(cps, [25, 75]),
                 np.median(wpw), np.median(wps), *np.percentile(wps, [25, 75]), np.round(np.median(cs / cw, 0), 2), np.median(ex), *np.percentile(ex, [25, 75]),
                 np.median([r["expo_white"] for r in R]), np.median(kw), np.median(ks)))
    lag = np.array([r["corr_lag1_A"] for r in R]); print("autocorrelation lag-1 des deviations (protocole A) mediane %.3f, (B) %.3f" % (np.median(lag), np.median([r["corr_lag1_B"] for r in R])))
    sys.exit()
V3 = load_v3(); OUT = os.path.join(HERE, "rev3_sigma_fim_%d%s.jsonl" % (a.shard, a.tag))
done = {json.loads(l)["i"] for l in open(OUT)} if os.path.exists(OUT) else set()
t0 = time.time()
for i in [i for i in range(a.n) if i % a.nshards == a.shard and i not in done]:
    th, data, sg = data_like_campaign(V3, i, 5.0)
    Fw = sens_matrix(th, data); Fw = Fw.T @ Fw; sw = fim_summary(Fw, th)
    blocks = {}; lag = {}
    for kind, d in data["protocols"].items():
        P, t_max, swi = p3.power_profile(kind, th); grid = np.asarray(d["t"])
        clean, _ = p3.simulate(th, P, t_max, grid, switches=swi)
        rng = np.random.default_rng(10000 + 7 * i + (0 if kind == "A" else 1)); tt = np.arange(0.0, float(d["t_end"]) + 1.0, 1.0)
        devs, tends = [], []
        for _ in range(a.mc):
            nz = rng.normal(0, 5.0, len(tt)); Pn = lambda t, nz=nz, P=P: P(t) + float(np.interp(t, tt, nz))
            y, te = p3.simulate(th, Pn, t_max, grid, switches=swi); devs.append(y - clean); tends.append(te)
        m = grid < 0.98 * min(tends); D = np.array(devs)[:, m]
        C = np.cov(D, rowvar=False)
        Sabs = np.zeros((m.sum(), 5)); keep = p3._TOL[0]; p3.set_tol(1e-10)
        for k in range(5):
            tp, tm = th.copy(), th.copy(); h = 1e-4 * th[k]; tp[k] += h; tm[k] -= h
            yp, _ = p3.simulate(tp, P, t_max, grid[m], switches=swi); ym, _ = p3.simulate(tm, P, t_max, grid[m], switches=swi)
            Sabs[:, k] = (yp - ym) / (2 * h) * th[k]
        p3.set_tol(keep)
        sd = np.sqrt(np.diag(C)); ok = sd > 1e-9 * sd.max()
        with np.errstate(all="ignore"): Rm = C / np.outer(sd, sd)
        lag[kind] = float(np.median(np.diag(Rm, 1)[ok[:-1] & ok[1:]]))
        blocks[kind] = (Sabs, C)
    res = dict(i=i, crlb_white=sw["crlb"], cp_white=sw["crlb_CP"], wp_white=sw["crlb_Wp"], cond_white=sw["cond"], expo_white=sw["expo"],
               crlb_sigma={}, cp_sigma={}, wp_sigma={}, expo_sigma={}, cond_sigma={}, corr_lag1_A=lag.get("A"), corr_lag1_B=lag.get("B"))
    for alpha, nu in COMBOS:
        F = np.zeros((5, 5))
        for kind, (Sabs, C) in blocks.items():
            dC = np.diag(C); Cs = (1 - alpha) * C + alpha * np.diag(dC) + nu ** 2 * dC.mean() * np.eye(len(dC))
            F += Sabs.T @ np.linalg.solve(Cs, Sabs)
        s = fim_summary(F, th); key = "a%g_n%g" % (alpha, nu)
        res["crlb_sigma"][key] = s["crlb"]; res["cp_sigma"][key] = s["crlb_CP"]; res["wp_sigma"][key] = s["crlb_Wp"]; res["expo_sigma"][key] = s["expo"]; res["cond_sigma"][key] = s["cond"]
    open(OUT, "a").write(json.dumps(res) + "\n")
    print("i=%2d blanc %s | Sigma(a0.1,n0.1) %s ; CP %.3f -> %.3f ; lag1 A %.2f B %.2f (%.0f s)" % (i, np.round(sw["crlb"], 2), np.round(res["crlb_sigma"]["a0.1_n0.1"], 2), sw["crlb_CP"], res["cp_sigma"]["a0.1_n0.1"], lag["A"], lag["B"], time.time() - t0), flush=True)
