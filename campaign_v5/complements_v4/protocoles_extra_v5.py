#!/usr/bin/env python3
"""Designs hors domaine severe (objection P-9 de l'expertise) : paliers sous-critiques, repos passif,
sprints a repos croissants. Meme machinerie que protocoles_v3 (FIM a 5 W, 120 observations par design,
calibration v5 du bruit, 15 athletes). Sortie : protocoles_extra_v5.jsonl (meme format que protocoles_v3.jsonl).
Usage : python3 protocoles_extra_v5.py [n_ath]"""
import json, os, sys, time, numpy as np
import protocoles_v3 as P
from common_v4 import p3, HERE

def seq(segments, tmax=3600.0):
    """segments : liste de (fraction de CP, duree en s) ; le dernier segment court jusqu'a l'epuisement (duree None)."""
    return {"kind": "seq", "seg": segments, "t_max": tmax}

EXTRA = {
  "80% 6 min (sous-critique seul)":        [seq([(0.80, None)], tmax=360.0)],
  "80% 6 min + 110%":                      [seq([(0.80, 360.0), (1.10, None)])],
  "110% 90 s + repos 3 min + 110%":        [seq([(1.10, 90.0), (0.0, 180.0), (1.10, None)])],
  "110% 90 s + repos 5 min (recuperation)": [seq([(1.10, 90.0), (0.0, None)], tmax=390.0)],
  "sprints 150% 30 s, repos 30/60/120/240 s": [seq([(1.50, 30.0), (0.0, 30.0), (1.50, 30.0), (0.0, 60.0), (1.50, 30.0), (0.0, 120.0), (1.50, 30.0), (0.0, 240.0), (1.50, None)])],
  "A + B + 80% 6 min":                     [{"kind": "legacy", "name": "A", "t_max": 900.0}, {"kind": "legacy", "name": "B", "t_max": 900.0}, seq([(0.80, None)], tmax=360.0)],
  "A + B + repos 3 min":                   [{"kind": "legacy", "name": "A", "t_max": 900.0}, {"kind": "legacy", "name": "B", "t_max": 900.0}, seq([(1.10, 90.0), (0.0, 180.0), (1.10, None)])],
}

_orig_make = P.make_profile
def make_profile(spec, th):
    if spec.get("kind") != "seq": return _orig_make(spec, th)
    cp = P.cp_true(th); segs = spec["seg"]; bounds = []; t = 0.0
    for frac, dur in segs[:-1]:
        t += dur; bounds.append(t)
    fr = [s[0] * cp for s in segs]
    def f(t, bounds=bounds, fr=fr):
        for k, b in enumerate(bounds):
            if t < b: return fr[k]
        return fr[-1]
    return f, spec["t_max"], tuple(bounds)
P.make_profile = make_profile

def main():
    n_ath = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    OUT = os.path.join(HERE, "protocoles_extra_v5.jsonl"); pop = p3.gen_population(50)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT): r = json.loads(l); done.add((r["proto"], r["i"]))
    f = open(OUT, "a"); t0 = time.time()
    for name, specs in EXTRA.items():
        for i in range(n_ath):
            if (name, i) in done: continue
            th = pop[i]; F = P.fim_for(th, specs, 5.0, 1000 * i + 50); s = P.summarize(F) if F is not None else None
            rec = {"proto": name, "i": i, "ok": s is not None, "cp_true": float(P.cp_true(th))}
            if s: rec.update(s)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print("%-42s i=%2d %s (%.0f s)" % (name, i, ("cond=%.3g  CRLB AOmax %.1f MR %.1f  expo %.2f" % (s["cond"], s["crlb"][1], s["crlb"][3], s["expo"])) if s else "ECHEC", time.time() - t0), flush=True)
    f.close(); print("TERMINE protocoles_extra_v5")
if __name__ == "__main__": main()
