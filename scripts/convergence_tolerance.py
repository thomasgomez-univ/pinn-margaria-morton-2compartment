#!/usr/bin/env python3
"""
Etude de convergence de la tolerance d'integration.

Le manuscrit d'origine genere les donnees a rtol = 1e-7 et evalue les residus
a rtol = 1e-4 avec un integrateur different (RK23). Un jacobien par differences
finies calcule sur une EDO integree a 1e-4 est domine par le bruit
d'integration : c'est l'explication la plus probable des echecs de LM.

Ce script mesure, pour chaque tolerance, (i) l'ecart de trajectoire par rapport
a la reference 1e-10, (ii) l'ecart des parametres estimes par LM et DE par
rapport a ceux obtenus a 1e-10, et (iii) le cout en temps. On retient la
tolerance la plus large dont les estimations coincident avec la reference.

Usage : python3 convergence_tolerance.py --athletes 3
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")

import argparse
import time

import numpy as np

from mm2c import pipeline as P

TOLS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
REF_TOL = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athletes", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--tols", type=float, nargs="+", default=None)
    ap.add_argument("--budget", type=int, default=4000)
    a = ap.parse_args()

    pop = P.gen_population(50, P.SEED_POP)
    pop_mean = pop.mean(axis=0)
    cfg = P.Config(noise_mode="observation", budget_ode=a.budget)
    global TOLS
    TOLS = a.tols if a.tols else TOLS

    print("Reference : rtol = atol = %g\n" % REF_TOL)

    traj = {t: [] for t in TOLS}
    est = {t: {"LM": [], "DE": []} for t in TOLS}
    ref_est = {"LM": [], "DE": []}
    cost_s = {t: [] for t in TOLS}

    for i in range(a.athletes):
        th = pop[i]
        seed = 1000 * i + int(a.sigma * 10)

        # jeu de donnees construit UNE FOIS a la tolerance de reference
        P.set_tol(REF_TOL)
        data = P.gen_data(th, a.sigma, seed, cfg)
        Pf, t_max = P.power_profile("B", th)
        grid = np.asarray(data["protocols"]["B"]["t"])
        ref_traj, _ = P.simulate(th, Pf, t_max, grid)
        for m, fn in (("LM", P.fit_LM), ("DE", P.fit_DE)):
            x, _ = fn(data, cfg, seed)
            ref_est[m].append(x)

        for tol in TOLS:
            P.set_tol(tol)
            y, _ = P.simulate(th, Pf, t_max, grid)
            traj[tol].append(np.max(np.abs(y - ref_traj)) / th[2] * 100)
            for m, fn in (("LM", P.fit_LM), ("DE", P.fit_DE)):
                t0 = time.time()
                x, _ = fn(data, cfg, seed)
                est[tol][m].append(x)
                if m == "LM":
                    cost_s[tol].append(time.time() - t0)
        print("  athlete %d traite" % i, flush=True)

    print("\n%-10s %14s %14s %14s %10s" % (
        "rtol", "ecart traj (%)", "ecart LM (%)", "ecart DE (%)", "temps LM (s)"))
    print("-" * 68)
    for tol in TOLS:
        dLM = np.median([100 * np.max(np.abs(np.array(e) - np.array(r)) / np.array(r))
                         for e, r in zip(est[tol]["LM"], ref_est["LM"])])
        dDE = np.median([100 * np.max(np.abs(np.array(e) - np.array(r)) / np.array(r))
                         for e, r in zip(est[tol]["DE"], ref_est["DE"])])
        print("%-10g %14.2e %14.3f %14.3f %10.1f" % (
            tol, np.median(traj[tol]), dLM, dDE, np.median(cost_s[tol])))

    print("\nLecture : l'ecart des parametres estimes est ce qui compte, pas")
    print("l'ecart de trajectoire. Retenir la tolerance la plus large dont")
    print("l'ecart d'estimation reste tres inferieur a l'erreur d'estimation")
    print("elle-meme (de l'ordre du pourcent).")


if __name__ == "__main__":
    main()
