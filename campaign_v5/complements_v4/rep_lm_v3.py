#!/usr/bin/env python3
"""Efficacite de LM sur realisations repetees : 8 athletes x 20 realisations
de bruit, sigma_P = 5 W, graines disjointes de la campagne (50000 + ...)."""
import json, os, sys, time, numpy as np
from common_v4 import p3, load_v3, data_like_campaign, BUDGET, HERE
V3 = load_v3(); OUT = os.path.join(HERE, "rep_lm_v3.jsonl")
shard, nsh = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
OUT = OUT.replace(".jsonl", "_%d.jsonl" % shard)
done = set()
if os.path.exists(OUT):
    for l in open(OUT): r = json.loads(l); done.add((r["i"], r["k"]))
jobs = [(i, k) for i in range(8) for k in range(20)]
jobs = [j for n, j in enumerate(jobs) if n % nsh == shard and j not in done]
cfg = p3.Config(budget_ode=BUDGET); t0 = time.time()
for (i, k) in jobs:
    seed = 50000 + 1000 * i + k
    th, data, sg = data_like_campaign(V3, i, 5.0, seed=seed)
    x, se, n = p3.fit_LM_with_CI(data, cfg, seed)
    rec = dict(i=i, k=k, seed=seed, theta_true=th.tolist(), est=x.tolist(), se=se.tolist(), ode_calls=int(n),
               crlb_pct=V3[(i, 5.0)]["crlb_pct"])
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print("i=%d k=%2d  err%%=%s  (%.0f s)" % (i, k, np.round(100 * np.abs(x - th) / th, 2).tolist(), time.time() - t0), flush=True)
print("TERMINE rep_lm shard", shard)
