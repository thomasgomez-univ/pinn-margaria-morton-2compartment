#!/usr/bin/env python3
"""Controle de mauvaise specification : les donnees sont generees avec un ecart
que le modele ajuste ignore, puis LM, DE et PINN v4 sont ajustes avec le modele
exact de la campagne (RK45 1e-8, memes bornes, meme budget, memes graines).

Deux ecarts, sigma_P = 5 W, athletes 0..N-1 (defaut 20) :
  AR1   : bruit d'observation autocorrele AR(1), phi = 0.8 par pas de grille,
          meme ecart-type marginal sigma_AP que la campagne (le modele et la
          CRLB supposent un bruit blanc) ;
  DRIFT : efficacite eta(t) = eta0 * (1 - delta * t / T0), delta = 0.05, ou T0
          est le temps d'epuisement du modele non perturbe (eta tombe de 5 % a
          l'horizon nominal de chaque essai) ; bruit blanc comme la campagne.
Pour DRIFT, theta_true est le vecteur a t = 0 (eta0) ; les erreurs sur eta,
CP et W' sont relatives a cette valeur initiale.

Sortie : misspec_v4_<CASE>_<shard>.jsonl, une ligne par athlete : est (LM, DE,
PINN_v4), LM_se, crlb_pct (bruit blanc, theta_true), rmse (ajuste et a
theta_true) et sigma_AP, pour le test d'adequation.

Usage : python3 misspec_v4.py CASE [shard nshards] [--n 20] [--methods LM,DE,PINN_v4]
"""
import json, os, sys, time, argparse, numpy as np
from common_v4 import p3, load_v3, data_like_campaign, BUDGET, HERE, _load

ap = argparse.ArgumentParser()
ap.add_argument("case", choices=["AR1", "DRIFT"])
ap.add_argument("shard", nargs="?", type=int, default=0); ap.add_argument("nshards", nargs="?", type=int, default=1)
ap.add_argument("--n", type=int, default=20); ap.add_argument("--methods", default="LM,DE,PINN_v4")
ap.add_argument("--phi", type=float, default=0.8); ap.add_argument("--delta", type=float, default=0.05)
ap.add_argument("--sigma", type=float, default=5.0); ap.add_argument("--tag", default="")
a = ap.parse_args()
METHODS = a.methods.split(",")
p4 = _load("p4", "pipeline_v4.py") if "PINN_v4" in METHODS else None
V3 = load_v3()
OUT = os.path.join(HERE, "misspec_v4_%s_%d%s.jsonl" % (a.case, a.shard, a.tag))
done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)["i"])
jobs = [i for i in range(a.n) if i % a.nshards == a.shard and i not in done]
cfg = p3.Config(budget_ode=BUDGET)


def gen_perturbed(i, s):
    """Jeu de donnees de la campagne (grille, sigma_AP, graine), avec l'ecart demande."""
    th, data, sg = data_like_campaign(V3, i, s)          # reference non perturbee
    seed = 1000 * i + int(s * 10)
    rng = np.random.default_rng(seed + 777)
    for kind, d in data["protocols"].items():
        grid = np.asarray(d["t"]); sA = d["sigma_AP"]; T0 = float(d["t_end"])
        if a.case == "AR1":
            n = len(grid); e = np.empty(n)
            e[0] = rng.normal(0, sA)
            for k in range(1, n):
                e[k] = a.phi * e[k - 1] + rng.normal(0, sA * np.sqrt(1 - a.phi ** 2))
            d["A_P_obs"] = (np.asarray(d["A_P_clean"]) + e).tolist()
            d["perturb"] = dict(case="AR1", phi=a.phi)
        else:  # DRIFT : regenerer la trajectoire avec eta(t)
            rate = a.delta / T0
            def rhs_drift(t, state, theta, P, _rate=rate):
                M_O, A_Om, A_Pm, M_R, eta = theta
                A_O, A_P = state
                phi1 = M_O * (A_O / A_Om) * (1.0 - A_P / A_Pm)
                return [M_R * (A_Om - A_O) - phi1, phi1 - P(t) / (eta * (1.0 - _rate * t))]
            orig = p3.mm_rhs; p3.mm_rhs = rhs_drift
            try:
                _, t_end = p3.simulate(th, d["_P"], d["_t_max"], switches=d["_sw"])
                grid = np.linspace(0.0, t_end * 0.999, p3.N_OBS)
                clean, _ = p3.simulate(th, d["_P"], d["_t_max"], grid, switches=d["_sw"])
            finally:
                p3.mm_rhs = orig
            d["t"] = grid.tolist(); d["A_P_clean"] = np.asarray(clean).tolist(); d["t_end"] = float(t_end)
            d["A_P_obs"] = (clean + rng.normal(0.0, sA, len(grid))).tolist()
            d["perturb"] = dict(case="DRIFT", delta=a.delta, T0=T0, t_end_drift=float(t_end))
    return th, data, sg


def rmse(theta, data):
    ss, n = 0.0, 0
    for kind, d in data["protocols"].items():
        tf = float(d["t"][-1]) * (1 + 1e-9)
        pred, _ = p3.simulate(np.asarray(theta, float), d["_P"], tf, np.asarray(d["t"]), switches=d["_sw"])
        ss += float(np.sum((pred - np.asarray(d["A_P_obs"])) ** 2)); n += len(d["t"])
    return float(np.sqrt(ss / n))


t0 = time.time()
for i in jobs:
    th, data, sg = gen_perturbed(i, a.sigma); seed = 1000 * i + int(a.sigma * 10)
    rec = dict(i=i, case=a.case, sigma_P=a.sigma, theta_true=th.tolist(),
               sigma_AP={k: v["sigma_AP"] for k, v in data["protocols"].items()},
               perturb={k: v["perturb"] for k, v in data["protocols"].items()},
               crlb_pct=p3.crlb(th, data).tolist(), rmse_true=rmse(th, data),
               est={}, rmse={}, ode_calls={}, wall_s={})
    for m in METHODS:
        t1 = time.time()
        if m == "LM":
            x, se, n = p3.fit_LM_with_CI(data, cfg, seed); rec["LM_se"] = np.asarray(se, float).tolist()
        elif m == "DE":
            x, n = p3.fit_DE(data, cfg, seed)
        elif m == "PINN_v4":
            x, n = p4.fit_PINN_v4(data, seed=seed)[0], 0
        else:
            raise ValueError(m)
        rec["est"][m] = np.asarray(x, float).tolist(); rec["rmse"][m] = rmse(x, data)
        rec["ode_calls"][m] = int(n); rec["wall_s"][m] = round(time.time() - t1, 1)
    open(OUT, "a").write(json.dumps(rec) + "\n")
    cp = lambda t: t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
    msg = " ".join("%s CP %.2f%%" % (m, 100 * abs(cp(rec["est"][m]) / cp(th) - 1)) for m in METHODS)
    print("%s i=%2d  %s  rmse_true %.0f  (%.0f s)" % (a.case, i, msg, rec["rmse_true"], time.time() - t0), flush=True)
print("TERMINE misspec", a.case, "shard", a.shard)
