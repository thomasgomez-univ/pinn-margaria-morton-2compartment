#!/usr/bin/env python3
"""Controle du 'crime inverse' : donnees generees par DOP853 a rtol = atol = 1e-10,
estimation par LM, DE et PINN v4 avec l'integrateur de la campagne (RK45, 1e-8).
15 athletes, sigma_P = 5 W, memes graines que la campagne."""
import json, os, sys, time, numpy as np
from scipy.integrate import solve_ivp as _siv
from common_v4 import p3, load_v3, data_like_campaign, BUDGET, HERE, _load
p4 = _load("p4", "pipeline_v4.py")
V3 = load_v3(); OUT = os.path.join(HERE, "inverse_crime_v3.jsonl")
shard, nsh = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
OUT = OUT.replace(".jsonl", "_%d.jsonl" % shard)
done = set()
if os.path.exists(OUT):
    for l in open(OUT): r = json.loads(l); done.add(r["i"])
jobs = [i for i in range(15) if i % nsh == shard and i not in done]
cfg = p3.Config(budget_ode=BUDGET); t0 = time.time()
def gen_dop853(i):
    """Genere les donnees avec DOP853 a 1e-10, puis restaure RK45 a 1e-8."""
    orig = p3.solve_ivp
    def siv(fun, span, y0, **kw):
        kw["method"] = "DOP853"; kw["rtol"] = 1e-10; kw["atol"] = 1e-10; return _siv(fun, span, y0, **kw)
    p3.solve_ivp = siv
    try: th, data, sg = data_like_campaign(V3, i, 5.0)
    finally: p3.solve_ivp = orig
    return th, data, sg
for i in jobs:
    th, data, sg = gen_dop853(i); seed = 1000 * i + 50
    rec = dict(i=i, theta_true=th.tolist(), est={}, ode_calls={}, wall_s={})
    for m, fn in (("LM", lambda: p3.fit_LM(data, cfg, seed)), ("DE", lambda: p3.fit_DE(data, cfg, seed)),
                  ("PINN_v4", lambda: (p4.fit_PINN_v4(data, seed=seed)[0], 0))):
        t1 = time.time(); x, n = fn(); rec["est"][m] = np.asarray(x, dtype=float).tolist(); rec["ode_calls"][m] = int(n); rec["wall_s"][m] = round(time.time() - t1, 1)
    # reference : estimation sur les donnees de la campagne (RK45 1e-8) est deja dans camp_v3 / camp_v4
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print("i=%2d  LM CP-err ... fait (%.0f s)" % (i, time.time() - t0), flush=True)
print("TERMINE inverse_crime shard", shard)
