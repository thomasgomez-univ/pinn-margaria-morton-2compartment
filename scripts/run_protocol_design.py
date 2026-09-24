#!/usr/bin/env python3
"""Catalogue de protocoles evalues par la FIM, sur la population."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
import json, os, sys, time
import numpy as np
from protocol_design_lib import P, fim_for, summarize, cp_true, NAMES

OUT = _os.path.join(_ROOT, "data", _os.path.join(_ROOT, "data", "protocol_design.jsonl"))

def const(hi, tmax=1800.0): return {"kind": "const", "hi": hi, "t_max": tmax}
def interm(hi, lo, on, off, tmax=1800.0):
    return {"kind": "interm", "hi": hi, "lo": lo, "on": on, "off": off, "t_max": tmax}

CATALOGUE = {
  # reference : les protocoles du manuscrit (fractions de M_O*eta)
  "manuscrit A+B":        [{"kind":"legacy","name":"A","t_max":900.0},
                           {"kind":"legacy","name":"B","t_max":1200.0}],
  "manuscrit A seul":     [{"kind":"legacy","name":"A","t_max":900.0}],
  # constants dans le domaine severe, en fraction de la CP VRAIE
  "const 105%":           [const(1.05)],
  "const 110%":           [const(1.10)],
  "const 120%":           [const(1.20)],
  "const 140%":           [const(1.40)],
  "const 105+140":        [const(1.05), const(1.40)],
  "const 110+140":        [const(1.10), const(1.40)],
  # intermittents severes
  "interm 120/60 30s":    [interm(1.20, 0.60, 30, 30)],
  "interm 120/60 60s":    [interm(1.20, 0.60, 60, 60)],
  "interm 120/60 120s":   [interm(1.20, 0.60, 120, 120)],
  "interm 140/50 30s":    [interm(1.40, 0.50, 30, 30)],
  "interm 140/80 60s":    [interm(1.40, 0.80, 60, 60)],
  "interm 160/40 15s":    [interm(1.60, 0.40, 15, 15)],
  # combinaisons
  "105 + interm120/60":   [const(1.05), interm(1.20, 0.60, 60, 60)],
  "140 + interm120/60":   [const(1.40), interm(1.20, 0.60, 60, 60)],
  "105 + 140 + interm":   [const(1.05), const(1.40), interm(1.20, 0.60, 60, 60)],
}

def main():
    n_ath = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    pop = P.gen_population(50, P.SEED_POP)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                r = json.loads(l); done.add((r["proto"], r["i"]))
            except Exception: pass
    f = open(OUT, "a")
    t0 = time.time()
    for name, specs in CATALOGUE.items():
        for i in range(n_ath):
            if (name, i) in done: continue
            th = pop[i]
            F = fim_for(th, specs, 5.0, 1000*i+50)
            s = summarize(F) if F is not None else None
            rec = {"proto": name, "i": i, "ok": s is not None,
                   "cp_true": float(cp_true(th))}
            if s: rec.update(s)
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
            print("%-22s i=%2d %s (%.0f s)" % (name, i, "ok" if s else "ECHEC",
                                               time.time()-t0), flush=True)
    f.close(); print("TERMINE")

if __name__ == "__main__":
    main()
