#!/usr/bin/env python3
"""Balayage du poids du residu physique w_r pour le PINN v4 : 15 athletes,
sigma_P = 5 W, w_r in {0.03, 0.1, 0.3, 1, 3}, memes donnees que la campagne."""
import json, os, sys, time, numpy as np
from common_v4 import p3, load_v3, data_like_campaign, HERE, _load
p4 = _load("p4", "pipeline_v4.py")
V3 = load_v3(); OUT = os.path.join(HERE, "wr_sweep_v4.jsonl")
shard, nsh = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
OUT = OUT.replace(".jsonl", "_%d.jsonl" % shard)
done = set()
if os.path.exists(OUT):
    for l in open(OUT): r = json.loads(l); done.add((r["i"], r["w_r"]))
W = [0.03, 0.1, 0.3, 1.0, 3.0]
jobs = [(i, w) for i in range(15) for w in W]
jobs = [j for n, j in enumerate(jobs) if n % nsh == shard and j not in done]
t0 = time.time()
for (i, w) in jobs:
    th, data, sg = data_like_campaign(V3, i, 5.0); seed = 1000 * i + 50
    cfg = dict(p4.H2); cfg["w_r"] = w
    t1 = time.time(); x, parts = p4.fit_PINN_v4(data, seed=seed, cfg=cfg)
    rec = dict(i=i, w_r=w, theta_true=th.tolist(), est=x.tolist(), wall_s=round(time.time() - t1, 1),
               rmse_fit=p4.rmse_fit(x, data), rmse_true=p4.rmse_fit(th, data), **parts)
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print("i=%2d w_r=%.2f err%%=%s (%.0f s)" % (i, w, np.round(100 * np.abs(x - th) / th, 1).tolist(), time.time() - t0), flush=True)
print("TERMINE wr_sweep shard", shard)
