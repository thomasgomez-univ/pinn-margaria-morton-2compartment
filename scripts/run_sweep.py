#!/usr/bin/env python3
"""
Balayage d'un hyperparametre du PINN, conception appariee.

Le meme jeu de donnees sert a toutes les valeurs balayees pour un athlete
donne, de sorte que l'effet de l'hyperparametre est mesure INTRA-athlete.
Les graines reproduisent exactement celles des campagnes precedentes
(seed = 1000 * i + int(10 * sigma_P)), donc les resultats se cumulent
directement avec ceux deja obtenus.

Sortie : un fichier JSONL, une ligne par execution, ecrit et synchronise au
fur et a mesure. Le script est relancable : il relit le fichier au demarrage
et ne recalcule que ce qui manque.

Exemples
--------
    # E5 : balayage du poids du residu physique, population complete
    python3 run_sweep.py --param w_r --values 0.03 0.1 0.3 1 3 \
                         --athletes 0-49 --jobs 8 --out e5_wr_full.jsonl

    # E5b : balayage du nombre de points de collocation
    python3 run_sweep.py --param n_colloc --values 25 50 100 200 400 \
                         --athletes 0-24 --jobs 8 --out e5_colloc.jsonl
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np

from mm2c import pipeline as P

CASTS = {"w_r": float, "n_colloc": int, "pinn_adam": int, "pinn_lbfgs": int}


def parse_athletes(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def one(job):
    """Une execution : (athlete, valeur). Processus isole, 1 thread torch."""
    i, val, param, sigma, tol = job
    P.set_tol(tol)
    cfg = P.Config(noise_mode="observation", **{param: val})
    pop = P.gen_population(50, P.SEED_POP)
    th = pop[i]
    seed = 1000 * i + int(sigma * 10)
    data = P.gen_data(th, sigma, seed, cfg)
    t0 = time.time()
    x, _ = P.fit_PINN(data, cfg, seed)
    return {"i": i, "param": param, "value": val, "sigma_P": sigma,
            "rtol": tol, "theta_true": th.tolist(),
            "est": list(map(float, x)), "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param", required=True, choices=sorted(CASTS),
                    help="hyperparametre balaye")
    ap.add_argument("--values", required=True, nargs="+",
                    help="valeurs balayees")
    ap.add_argument("--athletes", default="0-49",
                    help="indices, ex. '0-49' ou '0,3,7-10' (defaut 0-49)")
    ap.add_argument("--sigma", type=float, default=5.0, help="sigma_P en W")
    ap.add_argument("--tol", type=float, default=1e-6, help="rtol = atol")
    ap.add_argument("--jobs", type=int, default=1,
                    help="processus en parallele (1 coeur chacun)")
    ap.add_argument("--out", required=True, help="fichier JSONL de sortie")
    a = ap.parse_args()

    cast = CASTS[a.param]
    values = [cast(v) for v in a.values]
    athletes = parse_athletes(a.athletes)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["i"], r["value"]))
            except Exception:
                pass

    jobs = [(i, v, a.param, a.sigma, a.tol)
            for i in athletes for v in values if (i, v) not in done]
    total = len(athletes) * len(values)
    print("%s : %d executions au total, %d deja faites, %d a lancer"
          % (a.param, total, len(done), len(jobs)), flush=True)
    if not jobs:
        print("rien a faire."); return

    t0 = time.time()
    with open(a.out, "a") as f:
        if a.jobs > 1:
            with Pool(a.jobs) as pool:
                for k, rec in enumerate(pool.imap_unordered(one, jobs), 1):
                    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
                    el = time.time() - t0
                    print("[%d/%d] i=%d %s=%s  (%.0f s ecoulees, ~%.0f min restantes)"
                          % (k, len(jobs), rec["i"], a.param, rec["value"],
                             el, el / k * (len(jobs) - k) / 60), flush=True)
        else:
            for k, job in enumerate(jobs, 1):
                rec = one(job)
                f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
                el = time.time() - t0
                print("[%d/%d] i=%d %s=%s  (%.0f s ecoulees, ~%.0f min restantes)"
                      % (k, len(jobs), rec["i"], a.param, rec["value"],
                         el, el / k * (len(jobs) - k) / 60), flush=True)
    print("TERMINE en %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
