#!/usr/bin/env python3
"""Rev.3 bis (R2, bruit de puissance) : erreurs dans les variables. L'athlete est pilote par la puissance vraie P(t) et A_P(t)
est observe sans bruit sur la grille de la campagne ; l'estimateur recoit la puissance enregistree par le capteur,
P_n(t) = P(t) + bruit blanc N(0, sigma_P) a 1 s interpole (meme mecanisme que la calibration de sigma_AP), comme entree du
modele. TRF (fit_LM_with_CI de la campagne, memes bornes, budget, graines ; pas de difference finie 1e-5 et rtol 1e-10, voir ci-dessous),
residu pondere par le sigma_AP de la campagne.
Comparaison : erreur par parametre contre la borne blanche (crlb_pct de la campagne) et la borne Sigma (rev3_sigma_fim, cle
a0.1_n0.1) ; couverture des IC du jacobien ; couverture par les bornes. 20 athletes, sigma_P = 5 W.
Usage : python3 rev3_eiv.py [shard nshards] [--n 20] ; synthese : --analyse"""
import json, os, sys, glob, time, argparse, functools, numpy as np
from common_v4 import p3, load_v3, data_like_campaign, BUDGET, HERE
ap = argparse.ArgumentParser(); ap.add_argument("shard", nargs="?", type=int, default=0); ap.add_argument("nshards", nargs="?", type=int, default=1)
ap.add_argument("--n", type=int, default=20); ap.add_argument("--analyse", action="store_true"); a = ap.parse_args()
SIG = 5.0
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def wp(t): return t[2] * t[4]
if a.analyse:
    from scipy.stats import beta
    def cpi(k, n): return (0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(0.975, k + 1, n - k))
    R = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "rev3_eiv_*.jsonl"))) for l in open(f)]; n = len(R); print("athletes", n)
    S = {json.loads(l)["i"]: json.loads(l) for f in glob.glob(os.path.join(HERE, "rev3_sigma_fim_*.jsonl")) for l in open(f)}
    th = np.array([r["theta_true"] for r in R]); x = np.array([r["est"] for r in R]); se = np.array([r["se"] for r in R])
    cw = np.array([r["crlb_white"] for r in R]); err = 100 * np.abs(x / th - 1)
    ids = [r["i"] for r in R]; cs = np.array([S[i]["crlb_sigma"]["a0.1_n0.1"] for i in ids]) if all(i in S for i in ids) else None
    print("erreur mediane par parametre (%%) : %s ; CP %.3f ; W' %.3f" % (np.round(np.median(err, 0), 3), np.median([100 * abs(cp(u) / cp(v) - 1) for u, v in zip(x, th)]), np.median([100 * abs(wp(u) / wp(v) - 1) for u, v in zip(x, th)])))
    print("borne blanche mediane %s ; efficacite vs blanche %s" % (np.round(np.median(cw, 0), 3), np.round(np.median(err / cw, 0) / 0.6745, 2)))
    if cs is not None: print("borne Sigma mediane %s ; efficacite vs Sigma %s" % (np.round(np.median(cs, 0), 3), np.round(np.median(err / cs, 0) / 0.6745, 2)))
    k1 = np.sum(err < 1.96 * cw, 0); print("dans +-1.96 borne blanche :", ["%d/%d [%.0f-%.0f]" % (k, n, 100 * cpi(k, n)[0], 100 * cpi(k, n)[1]) for k in k1])
    if cs is not None: k2 = np.sum(err < 1.96 * cs, 0); print("dans +-1.96 borne Sigma  :", ["%d/%d [%.0f-%.0f]" % (k, n, 100 * cpi(k, n)[0], 100 * cpi(k, n)[1]) for k in k2])
    k3 = np.sum(np.abs(x - th) <= 1.96 * se, 0); print("IC du jacobien (bruit blanc suppose) :", ["%d/%d" % (k, n) for k in k3])
    print("misfit / sigma_AP mediane %.3f" % np.median([r["misfit"] for r in R]))
    sys.exit()
V3 = load_v3(); cfg = p3.Config(budget_ode=BUDGET)
# L'entree bruitee interpolee rend la trajectoire non lisse en theta a l'echelle du pas de difference finie par defaut
# (1.5e-8) : le jacobien numerique est alors du bruit et l'algorithme s'arrete (xtol) loin du minimum. Pas porte a 1e-5
# et tolerance d'integration a 1e-10 (verifie sur l'athlete 0 : convergence depuis un depart perturbe de 30 %).
p3.least_squares = functools.partial(p3.least_squares, diff_step=1e-5)
OUT = os.path.join(HERE, "rev3_eiv_%d.jsonl" % a.shard); done = {json.loads(l)["i"] for l in open(OUT)} if os.path.exists(OUT) else set()
t0 = time.time()
for i in [i for i in range(a.n) if i % a.nshards == a.shard and i not in done]:
    th, data, sg = data_like_campaign(V3, i, SIG); seed = 1000 * i + int(SIG * 10)
    rng = np.random.default_rng(seed + 4242)
    for kind, d in data["protocols"].items():
        P, t_max, sw = p3.power_profile(kind, th); tt = np.arange(0.0, float(d["t_end"]) + 1.0, 1.0); nz = rng.normal(0, SIG, len(tt))
        d["_P"] = (lambda t, nz=nz, P=P, tt=tt: P(t) + float(np.interp(t, tt, nz)))      # entree bruitee donnee a l'estimateur
        d["A_P_obs"] = list(d["A_P_clean"])                                                  # observation exacte sous la puissance vraie
    p3.set_tol(1e-10); x, se, nf = p3.fit_LM_with_CI(data, cfg, seed)
    ss, nn = 0.0, 0
    for kind, d in data["protocols"].items():
        tf = float(d["t"][-1]) * (1 + 1e-9); pred, _ = p3.simulate(np.asarray(x), d["_P"], tf, np.asarray(d["t"]), switches=d["_sw"])
        ss += float(np.sum(((pred - np.asarray(d["A_P_obs"])) / d["sigma_AP"]) ** 2)); nn += len(d["t"])
    rec = dict(i=i, sigma_P=SIG, theta_true=th.tolist(), est=np.asarray(x, float).tolist(), se=np.asarray(se, float).tolist(),
               crlb_white=V3[(i, SIG)]["crlb_pct"], misfit=float(np.sqrt(ss / nn)), ode_calls=int(nf), wall_s=round(time.time() - t0, 1))
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print("i=%2d CP %.3f%% W' %.3f%% par %s misfit %.2f (%.0f s)" % (i, 100 * abs(cp(x) / cp(th) - 1), 100 * abs(wp(x) / wp(th) - 1), np.round(100 * np.abs(np.array(x) / th - 1), 2), rec["misfit"], time.time() - t0), flush=True)
print("TERMINE eiv shard", a.shard)
