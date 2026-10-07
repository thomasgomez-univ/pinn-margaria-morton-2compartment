#!/usr/bin/env python3
"""Conception de protocole par la FIM, population recalibree, sensibilites
convergentes (rtol 1e-10, pas 1e-4). Budget d'observation total fixe (120)."""
import json, os, sys, time, numpy as np
from common_v4 import p3, HERE
p3.set_tol(1e-8)
NAMES = ["M_O", "A_Omax", "A_Pmax", "M_R", "eta"]
def const(hi, tmax=1800.0): return {"kind": "const", "hi": hi, "t_max": tmax}
def interm(hi, lo, on, off, tmax=1800.0): return {"kind": "interm", "hi": hi, "lo": lo, "on": on, "off": off, "t_max": tmax}
CATALOGUE = {
  "manuscrit A+B":        [{"kind": "legacy", "name": "A", "t_max": 900.0}, {"kind": "legacy", "name": "B", "t_max": 900.0}],
  "manuscrit A seul":     [{"kind": "legacy", "name": "A", "t_max": 900.0}],
  "const 105%": [const(1.05)], "const 110%": [const(1.10)], "const 120%": [const(1.20)], "const 140%": [const(1.40)],
  "const 105+140": [const(1.05), const(1.40)], "const 110+140": [const(1.10), const(1.40)],
  "interm 120/60 30s": [interm(1.20, 0.60, 30, 30)], "interm 120/60 60s": [interm(1.20, 0.60, 60, 60)],
  "interm 120/60 120s": [interm(1.20, 0.60, 120, 120)], "interm 140/50 30s": [interm(1.40, 0.50, 30, 30)],
  "interm 140/80 60s": [interm(1.40, 0.80, 60, 60)], "interm 160/40 15s": [interm(1.60, 0.40, 15, 15)],
  "105 + interm120/60": [const(1.05), interm(1.20, 0.60, 60, 60)], "140 + interm120/60": [const(1.40), interm(1.20, 0.60, 60, 60)],
  "105 + 140 + interm": [const(1.05), const(1.40), interm(1.20, 0.60, 60, 60)],
}
def cp_true(th):
    M_O, A_Om, A_Pm, M_R, eta = th; return M_O * (M_R * A_Om / (M_R * A_Om + M_O)) * eta
def make_profile(spec, th):
    cp = cp_true(th)
    if spec["kind"] == "const":
        p = spec["hi"] * cp; return (lambda t, p=p: p), spec["t_max"], ()
    if spec["kind"] == "interm":
        hi, lo, on, off = spec["hi"] * cp, spec["lo"] * cp, spec["on"], spec["off"]; per = on + off
        f = lambda t, hi=hi, lo=lo, on=on, per=per: (hi if (t % per) < on else lo)
        sw = sorted(set([x for x in np.arange(on, spec["t_max"], per)] + [x for x in np.arange(per, spec["t_max"], per)]))
        return f, spec["t_max"], tuple(sw)
    if spec["kind"] == "legacy":
        return p3.power_profile(spec["name"], th)
    raise ValueError(spec)
def sigma_AP(th, spec, sigma_P, seed, n=100):
    """v5 : bruit ajoute a la puissance exacte (commutations conservees), deviation mesuree avant
    l'epuisement le plus precoce des realisations -- meme correction que calibrate_sigma_AP (M-1)."""
    rng = np.random.default_rng(seed); Pf, t_max, sw = make_profile(spec, th)
    _, te = p3.simulate(th, Pf, t_max, switches=sw); te = min(te, spec["t_max"])
    grid = np.linspace(0.0, te * 0.999, 20); ref, _ = p3.simulate(th, Pf, t_max, grid, switches=sw); d = []; tends = []
    tt = np.arange(0.0, te + 1.0, 1.0)
    for _ in range(n):
        noise = rng.normal(0, sigma_P, len(tt)); Pn = lambda t, noise=noise: Pf(t) + float(np.interp(t, tt, noise))
        y, te2 = p3.simulate(th, Pn, t_max, grid, switches=sw); d.append(y - ref); tends.append(min(te2, spec["t_max"]))
    mask = grid < 0.98 * min(tends)
    return float(np.sqrt(np.mean(np.square(np.array(d)[:, mask]))))
def fim_for(th, specs, sigma_P=5.0, seed=0, n_obs_total=120):
    blocks = []; n_obs = max(8, n_obs_total // len(specs))
    for spec in specs:
        try:
            Pf, t_max, sw = make_profile(spec, th); _, te = p3.simulate(th, Pf, t_max, switches=sw)
        except Exception: return None
        if not np.isfinite(te) or te < 30.0: return None
        te = min(te, spec["t_max"]); grid = np.linspace(0.0, te * 0.999, n_obs)
        sA = sigma_AP(th, spec, sigma_P, seed)
        if not np.isfinite(sA) or sA <= 0: return None
        keep = p3._TOL[0]; p3.set_tol(1e-10)
        try:
            blk = np.zeros((len(grid), 5))
            for k in range(5):
                tp, tm = th.copy(), th.copy(); h = 1e-4 * abs(th[k]); tp[k] += h; tm[k] -= h
                yp, _ = p3.simulate(tp, Pf, t_max, grid, switches=sw); ym, _ = p3.simulate(tm, Pf, t_max, grid, switches=sw)
                blk[:, k] = (yp - ym) / (2 * h) * th[k] / sA
        finally: p3.set_tol(keep)
        if not np.all(np.isfinite(blk)): return None
        blocks.append(blk)
    S = np.vstack(blocks); return S.T @ S
def summarize(F):
    w, V = np.linalg.eigh(F); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    if w[-1] <= 0: return None
    v = V[:, -1] / np.abs(V[:, -1]).max()
    if v[3] < 0: v = -v
    return {"cond": float(w[0] / w[-1]), "crlb": (100 * np.sqrt(np.diag(np.linalg.inv(F)))).tolist(),
            "logdet": float(np.sum(np.log(w))), "lam_min": float(w[-1]), "vmin": v.tolist(),
            "expo": float(v[3] / v[1]) if abs(v[1]) > 1e-12 else float("nan")}
def main():
    n_ath = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    OUT = os.path.join(HERE, "protocoles_v3.jsonl"); pop = p3.gen_population(50)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r["proto"], r["i"]))
    f = open(OUT, "a"); t0 = time.time()
    for name, specs in CATALOGUE.items():
        for i in range(n_ath):
            if (name, i) in done: continue
            th = pop[i]; F = fim_for(th, specs, 5.0, 1000 * i + 50); s = summarize(F) if F is not None else None
            rec = {"proto": name, "i": i, "ok": s is not None, "cp_true": float(cp_true(th))}
            if s: rec.update(s)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print("%-22s i=%2d %s (%.0f s)" % (name, i, ("cond=%.3g" % s["cond"]) if s else "ECHEC", time.time() - t0), flush=True)
    f.close(); print("TERMINE protocoles_v3")
if __name__ == "__main__": main()
