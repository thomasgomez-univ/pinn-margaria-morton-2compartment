#!/usr/bin/env python3
"""
E6 -- Conception de protocole par la matrice d'information de Fisher.

Teste la prediction inscrite dans la Discussion : un protocole place dans le
domaine severe (105-140 % de la CP VRAIE, et non de M_O*eta) doit reduire le
conditionnement et resserrer les bornes sur M_R et A_Omax, sans instrument
supplementaire. Si ce n'est pas le cas, le mal-conditionnement est structurel.

Aucune dependance a torch : uniquement des resolutions d'EDO.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
import json, sys, time
import numpy as np
from mm2c import pipeline as P

P.set_tol(1e-6)
NAMES = ["M_O", "A_Omax", "A_Pmax", "M_R", "eta"]


def cp_true(th):
    """CP = M_O * x_ss * eta, x_ss = M_R*A_Omax / (M_R*A_Omax + M_O)."""
    M_O, A_Om, A_Pm, M_R, eta = th
    x = M_R * A_Om / (M_R * A_Om + M_O)
    return M_O * x * eta


def make_profile(spec, th):
    """spec = dict decrivant le protocole ; renvoie (P, t_max, switches)."""
    cp = cp_true(th)
    if spec["kind"] == "const":
        p = spec["hi"] * cp
        return (lambda t, p=p: p), spec["t_max"], ()
    if spec["kind"] == "interm":
        hi, lo, on, off = spec["hi"]*cp, spec["lo"]*cp, spec["on"], spec["off"]
        per = on + off
        f = lambda t, hi=hi, lo=lo, on=on, per=per: (hi if (t % per) < on else lo)
        return f, spec["t_max"], tuple(np.arange(on, spec["t_max"], on)) if on == off \
               else tuple(sorted(set(list(np.arange(on, spec["t_max"], per)) +
                                     list(np.arange(per, spec["t_max"], per)))))
    if spec["kind"] == "legacy":            # protocoles A / B du manuscrit
        return P.power_profile(spec["name"], th)
    raise ValueError(spec)


def fim_for(th, specs, sigma_P=5.0, seed=0, n_obs_total=120):
    """FIM cumulee sur la liste de protocoles.

    Le budget d'observation TOTAL est fixe (n_obs_total) et reparti a parts
    egales entre les epreuves : un protocole a deux epreuves ne beneficie donc
    pas de deux fois plus de mesures qu'un protocole a une seule epreuve.
    Renvoie None si un protocole ne produit pas d'epuisement exploitable."""
    blocks = []
    n_obs = max(8, n_obs_total // len(specs))
    for spec in specs:
        try:
            Pf, t_max, sw = make_profile(spec, th)
            _, te = P.simulate(th, Pf, t_max, switches=sw)
        except Exception:
            return None
        if not np.isfinite(te) or te < 30.0:
            return None
        te = min(te, spec["t_max"])
        grid = np.linspace(0.0, te * 0.999, n_obs)
        # ecart-type d'observation propage depuis sigma_P, par Monte-Carlo court
        sA = sigma_AP(th, spec, sigma_P, seed)
        if not np.isfinite(sA) or sA <= 0:
            return None
        blk = np.zeros((len(grid), 5))
        for k in range(5):
            tp, tm = th.copy(), th.copy()
            h = 1e-6 * abs(th[k]); tp[k] += h; tm[k] -= h
            yp, _ = P.simulate(tp, Pf, t_max, grid, switches=sw)
            ym, _ = P.simulate(tm, Pf, t_max, grid, switches=sw)
            blk[:, k] = (yp - ym) / (2 * h) * th[k] / sA
        if not np.all(np.isfinite(blk)):
            return None
        blocks.append(blk)
    S = np.vstack(blocks)
    return S.T @ S


def sigma_AP(th, spec, sigma_P, seed, n=40):
    """Propagation de sigma_P vers l'echelle de A_P, par Monte-Carlo."""
    rng = np.random.default_rng(seed)
    Pf, t_max, sw = make_profile(spec, th)
    _, te = P.simulate(th, Pf, t_max, switches=sw)
    te = min(te, spec["t_max"])
    grid = np.linspace(0.0, te * 0.999, 20)
    ref, _ = P.simulate(th, Pf, t_max, grid, switches=sw)
    d = []
    for _ in range(n):
        tt = np.arange(0.0, te + 1.0, 1.0)
        pn = np.array([Pf(t) for t in tt]) + rng.normal(0, sigma_P, len(tt))
        Pn = lambda t, tt=tt, pn=pn: float(np.interp(t, tt, pn))
        y, _ = P.simulate(th, Pn, t_max, grid, switches=sw)
        d.append(np.std(y - ref))
    return float(np.mean(d))


def summarize(F):
    w = np.linalg.eigvalsh(F)
    w = np.sort(w)[::-1]
    if w[-1] <= 0:
        return None
    crlb = 100 * np.sqrt(np.diag(np.linalg.inv(F)))
    return {"cond": float(w[0] / w[-1]), "crlb": crlb.tolist(),
            "logdet": float(np.sum(np.log(w))), "lam_min": float(w[-1])}
