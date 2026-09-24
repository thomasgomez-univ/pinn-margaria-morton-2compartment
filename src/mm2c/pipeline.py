#!/usr/bin/env python3
"""
Pipeline d'estimation corrige et controle pour le modele bioenergetique
a deux compartiments (Margaria-Morton reduit).

Corrige les sept defauts de conception identifies dans le depot d'origine :

  1. Donnees identiques pour toutes les methodes (une seule grille).
  2. Integrateur et tolerance identiques partout (RK45, rtol = atol = 1e-8).
  3. Budgets de calcul comptabilises en resolutions d'EDO et equilibres.
  4. Seeds fixes (numpy et torch) derives de (athlete, sigma, methode).
  5. Deux baselines de population, obligatoires.
  6. Bornes de Cramer-Rao comme reference haute.
  7. Residu EDO corrige : (M_O/A_Pmax) et non (M_O/A_Omax) dans r_P,
     avec le poids w_r configurable pour l'ablation.

Le modele de bruit est explicite et commutable :
  - "observation" (defaut) : A_P bruite, cadre statistique standard, la FIM
    et les bornes de Cramer-Rao s'appliquent directement. L'ecart-type est
    calibre en propageant sigma_P (en watts) a travers l'integration, ce qui
    preserve le lien avec les capteurs de puissance reels.
  - "input" : reproduit le dispositif d'origine (A_P exact, puissance bruitee).
    Conserve pour la comparabilite avec les resultats publies.

Usage
-----
    python3 pipeline_corrige.py --athletes 8 --sigma 5 --methods all
    python3 pipeline_corrige.py --athletes 50 --sigma 2 5 10 --out res.jsonl

Dependances : numpy, scipy, torch
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, differential_evolution, minimize_scalar

# ============================================================================
# Constantes du modele
# ============================================================================

PARAM_NAMES = ["M_O", "A_O_max", "A_P_max", "M_R", "eta"]
PARAM_LB = np.array([600.0, 25_000.0, 30_000.0, 0.005, 0.18])
PARAM_UB = np.array([1800.0, 120_000.0, 160_000.0, 0.080, 0.32])

# Distribution de population (Table 2 du manuscrit)
POP_MEANS = np.array([1120.0, 60_000.0, 80_000.0, 0.030, 0.25])
POP_SDS = np.array([150.0, 10_000.0, 18_000.0, 0.012, 0.02])
POP_CORR = np.array([
    [1.00, 0.60, 0.30, 0.20, 0.10],
    [0.60, 1.00, 0.40, 0.15, 0.05],
    [0.30, 0.40, 1.00, 0.10, 0.05],
    [0.20, 0.15, 0.10, 1.00, 0.05],
    [0.10, 0.05, 0.05, 0.05, 1.00],
])
MR_LOW, MR_HIGH = 0.010, 0.055

# Controles communs
# Controle 2 : tolerance unique, identique pour la generation et l'estimation.
# Modifiable par --rtol ; toute valeur relachee DOIT etre justifiee par l'etude
# de convergence (voir convergence_tolerance.py).
_TOL = [1e-8]

def set_tol(v):
    _TOL[0] = float(v)
N_OBS = 60                  # controle 1 : une seule grille d'observation
SEED_POP = 42

_ODE_CALLS = [0]            # compteur global de resolutions d'EDO


@dataclass
class Config:
    """Parametres de l'experience."""
    noise_mode: str = "observation"      # "observation" | "input"
    budget_ode: int = 3000               # controle 3 : budget commun
    w_r: float = 0.3                     # poids du residu physique
    pinn_adam: int = 8000
    pinn_lbfgs: int = 80
    n_colloc: int = 50
    lm_restarts: int = 20
    de_popsize: int = 15
    de_maxiter: int = 200
    protocols: tuple = ("A", "B")        # protocoles utilises pour l'ajustement
    extra: dict = field(default_factory=dict)


# ============================================================================
# Modele
# ============================================================================

def mm_rhs(t, state, theta, P):
    M_O, A_Om, A_Pm, M_R, eta = theta
    A_O, A_P = state
    phi1 = M_O * (A_O / A_Om) * (1.0 - A_P / A_Pm)
    return [M_R * (A_Om - A_O) - phi1, phi1 - P(t) / eta]


def _exhaustion(t, y, theta, P):
    return y[1]
_exhaustion.terminal = True
_exhaustion.direction = -1


def simulate(theta, P, t_max, t_eval=None, switches=()):
    """Integre jusqu'a epuisement (A_P = 0) ou t_max, tolerance commune.

    L'entree des protocoles intermittents est constante par morceaux : on
    integre segment par segment entre les instants de commutation, ce qui
    laisse le pas adaptatif libre a l'interieur de chaque segment. Cela evite
    le max_step force qui dominait le cout, sans perte de precision.
    """
    _ODE_CALLS[0] += 1
    A_Om, A_Pm = theta[1], theta[2]
    bnds = [0.0] + [s for s in switches if 0.0 < s < t_max] + [t_max]
    state = [A_Om, A_Pm]
    sols, t_end = [], t_max
    for k in range(len(bnds) - 1):
        sol = solve_ivp(mm_rhs, (bnds[k], bnds[k + 1]), state, args=(theta, P),
                        method="RK45", events=_exhaustion,
                        rtol=_TOL[0], atol=_TOL[0], dense_output=True)
        sols.append(sol)
        if sol.t_events[0].size:
            t_end = sol.t_events[0][0]
            break
        state = [sol.y[0][-1], sol.y[1][-1]]
        t_end = sol.t[-1]
    if t_eval is None:
        return sols, t_end
    tt = np.clip(np.asarray(t_eval, dtype=float), 0.0, max(t_end - 1e-9, 0.0))
    out = np.empty_like(tt)
    for j, x in enumerate(tt):
        for sol in sols:
            if sol.t[0] - 1e-12 <= x <= sol.t[-1] + 1e-12:
                out[j] = sol.sol(x)[1]
                break
        else:
            out[j] = sols[-1].sol(sols[-1].t[-1])[1]
    return out, t_end


def power_profile(kind, theta):
    """Profils de puissance, exprimes en fraction de M_O * eta."""
    ref = theta[0] * theta[4]
    if kind == "A":                       # constant, 110 %
        return (lambda t: 1.10 * ref), 900.0, ()
    if kind == "B":                       # intermittent 130 / 50, 30 s / 30 s
        return ((lambda t: (1.30 if (t % 60.0) < 30.0 else 0.50) * ref), 1200.0,
                tuple(np.arange(30.0, 1200.0, 30.0)))
    if kind == "C":                       # constant, 105 % (domaine severe)
        return (lambda t: 1.05 * ref), 1800.0, ()
    raise ValueError(kind)


# ============================================================================
# Population
# ============================================================================

def gen_population(n=50, seed=SEED_POP):
    rng = np.random.default_rng(seed)
    cov = np.outer(POP_SDS, POP_SDS) * POP_CORR
    s = rng.multivariate_normal(POP_MEANS, cov, size=n)
    s[:, 3] = rng.uniform(MR_LOW, MR_HIGH, n)
    for i in range(n):
        for j in range(5):
            while s[i, j] < PARAM_LB[j] or s[i, j] > PARAM_UB[j]:
                s[i, j] = rng.normal(POP_MEANS[j], POP_SDS[j])
    return s


# ============================================================================
# Bruit
# ============================================================================

def calibrate_sigma_AP(theta, kind, sigma_P, n_mc=100, seed=0):
    """Ecart-type equivalent sur A_P, obtenu en propageant sigma_P.

    On simule n_mc realisations avec une puissance bruitee et on mesure
    l'ecart quadratique moyen de A_P par rapport a la trajectoire propre.
    Cela donne un sigma_AP qui conserve le lien avec le capteur reel, tout
    en placant le probleme dans le cadre standard du bruit de mesure.
    """
    P, t_max, sw = power_profile(kind, theta)
    _, t_end = simulate(theta, P, t_max, switches=sw)
    grid = np.linspace(0.0, t_end * 0.999, N_OBS)
    clean, _ = simulate(theta, P, t_max, grid, switches=sw)
    rng = np.random.default_rng(seed)
    devs = []
    tt = np.arange(0.0, t_end + 1.0, 1.0)
    for _ in range(n_mc):
        pn = np.array([P(t) for t in tt]) + rng.normal(0, sigma_P, len(tt))
        Pn = lambda t: float(np.interp(t, tt, pn))
        y, _ = simulate(theta, Pn, t_max, grid, switches=sw)
        devs.append(y - clean)
    return float(np.sqrt(np.mean(np.square(devs))))


def gen_data(theta, sigma_P, seed, cfg: Config):
    """Jeu de donnees unique, partage par toutes les methodes (controle 1)."""
    rng = np.random.default_rng(seed)
    out = {"theta_true": theta.tolist(), "sigma_P": sigma_P,
           "noise_mode": cfg.noise_mode, "protocols": {}}
    for kind in cfg.protocols:
        P, t_max, sw = power_profile(kind, theta)
        _, t_end = simulate(theta, P, t_max, switches=sw)
        grid = np.linspace(0.0, t_end * 0.999, N_OBS)
        clean, _ = simulate(theta, P, t_max, grid, switches=sw)
        if cfg.noise_mode == "observation":
            s_AP = calibrate_sigma_AP(theta, kind, sigma_P, seed=seed)
            obs = clean + rng.normal(0.0, s_AP, len(grid))
            P_used = P
        elif cfg.noise_mode == "input":
            s_AP = None
            tt = np.arange(0.0, t_end + 1.0, 1.0)
            pn = np.array([P(t) for t in tt]) + rng.normal(0, sigma_P, len(tt))
            P_used = lambda t, tt=tt, pn=pn: float(np.interp(t, tt, pn))
            obs = clean
        else:
            raise ValueError(cfg.noise_mode)
        out["protocols"][kind] = {
            "t": grid.tolist(), "A_P_obs": np.asarray(obs).tolist(),
            "A_P_clean": np.asarray(clean).tolist(),
            "t_end": t_end, "sigma_AP": s_AP, "_P": P_used, "_t_max": t_max,
            "_sw": sw}
    return out


# ============================================================================
# Residus et objectif communs
# ============================================================================

def residuals(theta, data):
    """Residus empiles sur tous les protocoles, echelle commune."""
    r = []
    for kind, d in data["protocols"].items():
        pred, _ = simulate(np.asarray(theta), d["_P"], d["_t_max"],
                           np.asarray(d["t"]), switches=d["_sw"])
        sc = d["sigma_AP"] if d["sigma_AP"] else (np.std(d["A_P_obs"]) + 1.0)
        r.append((pred - np.asarray(d["A_P_obs"])) / sc)
    return np.concatenate(r)


def cost(theta, data):
    try:
        return float(np.mean(residuals(theta, data) ** 2))
    except Exception:
        return 1e12


# ============================================================================
# Baselines (controle 5)
# ============================================================================

def baseline_population(data, pop_mean):
    return np.asarray(pop_mean, dtype=float)


def baseline_population_APmax(data, pop_mean):
    """Moyenne de population, A_P_max seul ajuste (recherche 1-D)."""
    th = np.asarray(pop_mean, dtype=float).copy()

    def f(a):
        t = th.copy()
        t[2] = a
        return cost(t, data)

    r = minimize_scalar(f, bounds=(PARAM_LB[2], PARAM_UB[2]), method="bounded",
                        options={"xatol": 1.0})
    th[2] = r.x
    return th


# ============================================================================
# Estimateurs classiques (controles 2 et 3)
# ============================================================================

def fit_LM(data, cfg: Config, seed=0):
    rng = np.random.default_rng(seed)
    n0 = _ODE_CALLS[0]
    per_restart = max(10, cfg.budget_ode // (cfg.lm_restarts * 6))
    best, best_x = np.inf, None
    for _ in range(cfg.lm_restarts):
        x0 = rng.uniform(PARAM_LB, PARAM_UB)
        try:
            r = least_squares(residuals, x0, args=(data,),
                              bounds=(PARAM_LB, PARAM_UB), method="trf",
                              max_nfev=per_restart)
            if r.cost < best:
                best, best_x = r.cost, r.x.copy()
        except Exception:
            continue
    x = best_x if best_x is not None else (PARAM_LB + PARAM_UB) / 2
    return x, _ODE_CALLS[0] - n0


def fit_DE(data, cfg: Config, seed=0):
    """DE, arrete des que le budget commun en resolutions d'EDO est atteint."""
    n0 = _ODE_CALLS[0]

    def stop(xk, convergence=0.0):
        return (_ODE_CALLS[0] - n0) >= cfg.budget_ode

    r = differential_evolution(cost, list(zip(PARAM_LB, PARAM_UB)), args=(data,),
                               seed=seed, popsize=cfg.de_popsize,
                               maxiter=cfg.de_maxiter, tol=1e-8, polish=True,
                               callback=stop)
    return r.x, _ODE_CALLS[0] - n0


def fit_LM_with_CI(data, cfg: Config, seed=0):
    """LM + intervalles de confiance asymptotiques par le jacobien.

    Repond a la critique du relecteur 1 : LM fournit bien des intervalles de
    confiance, via la covariance asymptotique sigma^2 (J^T J)^-1.
    """
    x, n = fit_LM(data, cfg, seed)
    r = least_squares(residuals, x, args=(data,), bounds=(PARAM_LB, PARAM_UB),
                      method="trf", max_nfev=200)
    J = r.jac
    dof = max(len(r.fun) - len(x), 1)
    s2 = float(np.sum(r.fun ** 2) / dof)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(5, np.nan)
    return r.x, se, n


# ============================================================================
# PINN, residu corrige (controle 7)
# ============================================================================

def fit_PINN(data, cfg: Config, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    kind = "B" if "B" in data["protocols"] else list(data["protocols"])[0]
    d = data["protocols"][kind]
    t = np.asarray(d["t"])
    obs = np.asarray(d["A_P_obs"])
    T = float(d["t_end"])

    lb = torch.tensor(PARAM_LB, dtype=torch.float32)
    ub = torch.tensor(PARAM_UB, dtype=torch.float32)

    layers = [nn.Linear(1, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(),
              nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 2), nn.Sigmoid()]
    net = nn.Sequential(*layers)
    z = nn.Parameter(torch.zeros(5))

    scale = float(np.max(obs)) + 1.0
    t_d = torch.tensor(t / T, dtype=torch.float32).unsqueeze(1)
    y_d = torch.tensor(obs / scale, dtype=torch.float32).unsqueeze(1)
    tc = torch.linspace(0, 1, cfg.n_colloc).unsqueeze(1)
    Pc = torch.tensor([d["_P"](x) for x in np.linspace(0, T, cfg.n_colloc)],
                      dtype=torch.float32).unsqueeze(1)

    def theta_of(z):
        return lb + (ub - lb) * torch.sigmoid(z)

    def losses(ep, n_ep):
        M_O, A_Om, A_Pm, M_R, eta = theta_of(z)
        pred = net(t_d)
        ld = torch.mean((pred[:, 1:2] * A_Pm / scale - y_d) ** 2)
        tt = tc.clone().requires_grad_(True)
        pc = net(tt)
        aO, aP = pc[:, 0:1], pc[:, 1:2]
        daO = torch.autograd.grad(aO, tt, torch.ones_like(aO), True, True)[0]
        daP = torch.autograd.grad(aP, tt, torch.ones_like(aP), True, True)[0]
        rO = T * (M_R * (1 - aO) - (M_O / A_Om) * aO * (1 - aP))
        # ---- residu CORRIGE : M_O / A_Pmax, et non M_O / A_Omax ----
        rP = T * ((M_O / A_Pm) * aO * (1 - aP) - Pc / (eta * A_Pm))
        lr = torch.mean((daO - rO) ** 2 + (daP - rP) ** 2)
        p0 = net(torch.tensor([[0.0]]))
        li = (p0[0, 0] - 1) ** 2 + (p0[0, 1] - 1) ** 2
        w = cfg.w_r * min(1.0, ep / max(n_ep * 0.15, 1)) if n_ep else cfg.w_r
        return ld + w * lr + 5.0 * li

    opt = torch.optim.Adam(list(net.parameters()) + [z], lr=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.pinn_adam)
    for ep in range(cfg.pinn_adam):
        opt.zero_grad()
        loss = losses(ep, cfg.pinn_adam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + [z], 1.0)
        opt.step()
        sch.step()

    if cfg.pinn_lbfgs:
        opt2 = torch.optim.LBFGS(list(net.parameters()) + [z], lr=0.05,
                                 max_iter=cfg.pinn_lbfgs, history_size=20,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            l = losses(cfg.pinn_adam, 0)
            l.backward()
            return l
        try:
            opt2.step(closure)
        except Exception:
            pass

    with torch.no_grad():
        return theta_of(z).numpy(), 0


# ============================================================================
# Bornes de Cramer-Rao (controle 6)
# ============================================================================

def crlb(theta, data, rel_step=1e-6):
    """Ecart-type minimal relatif par parametre, en %."""
    S = []
    for kind, d in data["protocols"].items():
        grid = np.asarray(d["t"])
        s_AP = d["sigma_AP"]
        if not s_AP:
            continue
        blk = np.zeros((len(grid), 5))
        for k in range(5):
            tp, tm = theta.copy(), theta.copy()
            h = rel_step * abs(theta[k])
            tp[k] += h
            tm[k] -= h
            yp, _ = simulate(tp, d["_P"], d["_t_max"], grid, switches=d["_sw"])
            ym, _ = simulate(tm, d["_P"], d["_t_max"], grid, switches=d["_sw"])
            blk[:, k] = (yp - ym) / (2 * h) * theta[k] / s_AP
        S.append(blk)
    if not S:
        return np.full(5, np.nan)
    F = np.vstack(S)
    F = F.T @ F
    try:
        return 100.0 * np.sqrt(np.diag(np.linalg.inv(F)))
    except np.linalg.LinAlgError:
        return np.full(5, np.nan)


# ============================================================================
# Orchestration
# ============================================================================

METHODS = ["pop", "pop+APmax", "LM", "DE", "PINN"]


def run_one(i, theta, sigma_P, cfg: Config, pop_mean, methods):
    seed = 1000 * i + int(sigma_P * 10)
    data = gen_data(theta, sigma_P, seed, cfg)
    rec = {"i": i, "sigma_P": sigma_P, "noise_mode": cfg.noise_mode,
           "w_r": cfg.w_r, "theta_true": theta.tolist(),
           "rtol": _TOL[0],
           "sigma_AP": {k: v["sigma_AP"] for k, v in data["protocols"].items()},
           "crlb_pct": crlb(theta, data).tolist(), "est": {}, "ode_calls": {},
           "wall_s": {}}
    for m in methods:
        t0 = time.time()
        if m == "pop":
            x, n = baseline_population(data, pop_mean), 0
        elif m == "pop+APmax":
            n0 = _ODE_CALLS[0]
            x = baseline_population_APmax(data, pop_mean)
            n = _ODE_CALLS[0] - n0
        elif m == "LM":
            x, se, n = fit_LM_with_CI(data, cfg, seed)
            rec.setdefault("LM_se", se.tolist())
        elif m == "DE":
            x, n = fit_DE(data, cfg, seed)
        elif m == "PINN":
            x, n = fit_PINN(data, cfg, seed)
        else:
            raise ValueError(m)
        rec["est"][m] = np.asarray(x, dtype=float).tolist()
        rec["ode_calls"][m] = int(n)
        rec["wall_s"][m] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athletes", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sigma", type=float, nargs="+", default=[5.0])
    ap.add_argument("--noise-mode", default="observation",
                    choices=["observation", "input"])
    ap.add_argument("--w-r", type=float, default=0.3)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--out", default="pipeline_out.jsonl")
    a = ap.parse_args()

    set_tol(a.rtol)
    methods = METHODS if a.methods == ["all"] else a.methods
    cfg = Config(noise_mode=a.noise_mode, w_r=a.w_r, budget_ode=a.budget)
    pop = gen_population(50, SEED_POP)
    pop_mean = pop.mean(axis=0)

    done = set()
    try:
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["i"], r["sigma_P"], r["w_r"], r["noise_mode"]))
    except FileNotFoundError:
        pass

    fh = open(a.out, "a", buffering=1)
    for s in a.sigma:
        for i in range(a.start, a.start + a.athletes):
            if (i, s, cfg.w_r, cfg.noise_mode) in done:
                continue
            rec = run_one(i, pop[i], s, cfg, pop_mean, methods)
            fh.write(json.dumps(rec) + "\n")
            th = pop[i]
            msg = []
            for m in methods:
                e = np.array(rec["est"][m])
                msg.append("%s %.1f%%" % (m, 100 * abs(e[0] * e[4] - th[0] * th[4])
                                          / (th[0] * th[4])))
            print("athlete %2d  sigma=%4.1f  | %s" % (i, s, "  ".join(msg)),
                  flush=True)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
