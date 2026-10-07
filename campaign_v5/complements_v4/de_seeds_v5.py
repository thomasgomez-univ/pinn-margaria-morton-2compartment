#!/usr/bin/env python3
"""M-19 / M-11 : variabilite de Differential Evolution entre graines. 20 athletes, sigma_P = 5 W, donnees de la campagne
(inchangees), graines campagne + 1 et + 2 (la graine 0 est dans camp_v3). Usage : python3 de_seeds_v5.py [shard nshards]
-> de_seeds_v5_<shard>.jsonl ; synthese : python3 de_seeds_v5.py --analyse"""
import json, os, sys, time, glob, numpy as np
from common_v4 import p3, load_v3, data_like_campaign, BUDGET, HERE
N, SIG, SEEDS = 20, 5.0, (1, 2)
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def wp(t): return t[2] * t[4]
if "--analyse" in sys.argv:
    V3 = load_v3(); R = {}
    for f in sorted(glob.glob(os.path.join(HERE, "de_seeds_v5_*.jsonl"))):
        for l in open(f): r = json.loads(l); R[(r["i"], r["seed_k"])] = r["est"]
    ids = sorted(set(i for i, k in R)); E = {}
    for k in (0,) + SEEDS:
        E[k] = np.array([[100 * abs(cp((V3[(i, SIG)]["est"]["DE"] if k == 0 else R[(i, k)]) ) / cp(V3[(i, SIG)]["theta_true"]) - 1),
                          100 * abs(wp((V3[(i, SIG)]["est"]["DE"] if k == 0 else R[(i, k)])) / wp(V3[(i, SIG)]["theta_true"]) - 1)] for i in ids])
        print("graine %d : CP mediane %.2f [IQR %.2f-%.2f]  W' %.2f" % (k, np.median(E[k][:, 0]), *np.percentile(E[k][:, 0], [25, 75]), np.median(E[k][:, 1])))
    A = np.stack([E[k] for k in E]); sd_ath = np.std(A, axis=0, ddof=1)
    print("ecart-type inter-graines de l'erreur CP par athlete : mediane %.2f pt, max %.2f pt ; W' : %.2f / %.2f" % (np.median(sd_ath[:, 0]), sd_ath[:, 0].max(), np.median(sd_ath[:, 1]), sd_ath[:, 1].max()))
    print("mediane CP sur les 3 graines : %s -> ecart-type de la mediane %.3f pt" % (np.round([np.median(E[k][:, 0]) for k in E], 3).tolist(), np.std([np.median(E[k][:, 0]) for k in E], ddof=1)))
    print("n athletes : %d" % len(ids)); sys.exit(0)
shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0; nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT = os.path.join(HERE, "de_seeds_v5_%d.jsonl" % shard); done = set()
if os.path.exists(OUT):
    for l in open(OUT): r = json.loads(l); done.add((r["i"], r["seed_k"]))
V3 = load_v3(); cfg = p3.Config(budget_ode=BUDGET); t0 = time.time()
for i in [i for i in range(N) if i % nsh == shard]:
    th, data, sg = data_like_campaign(V3, i, SIG)
    for k in SEEDS:
        if (i, k) in done: continue
        x, n = p3.fit_DE(data, cfg, 1000 * i + 50 + k)
        open(OUT, "a").write(json.dumps(dict(i=i, seed_k=k, sigma_P=SIG, theta_true=th.tolist(), est=np.asarray(x, float).tolist(), ode_calls=int(n))) + "\n")
        print("i=%2d graine %d : CP err %.3f %% (%.0f s)" % (i, k, 100 * abs(cp(x) / cp(th) - 1), time.time() - t0), flush=True)
print("TERMINE de_seeds_v5 shard", shard)
