#!/usr/bin/env python3
"""Rev.3 (R1 M5, R2 R6) : robustesse des conclusions a la population virtuelle.
Populations de 20 athletes (sigma_P = 5 W, protocoles A + B, meme generateur que la campagne, make_population_v5) :
  seed8, seed9 : memes lois, autres graines ;
  rho05        : graine 7, correlation 0,5 entre CP et A_Pmax ;
  tau_wide     : graine 7, 1/M_R ~ U[12, 150] s au lieu de U[18, 91] s ;
  eta_low      : graine 7, eta ~ N(0.21, 0.02) (rendement brut) au lieu de N(0.25, 0.02).
Par athlete : donnees generees comme la campagne (calibration Monte-Carlo de sigma_AP, gen_data, graine 1000 i + 50),
information de Fisher (sensibilites relatives, rtol 1e-10), bornes par parametre, sur CP, W' et le flux M_R A_Omax,
exposant de la direction sloppy ; option --fit : LM multi-depart (10 premiers athletes) de la campagne, IC du jacobien.
Usage : python3 rev3_population.py POP [shard nshards] [--fit] ; synthese : python3 rev3_population.py --analyse"""
import json, os, sys, glob, time, argparse, importlib.util, numpy as np
from common_v4 import p3, sens_matrix, fim_summary, BUDGET, HERE
_MK = next(f for f in [os.path.join(HERE, "make_population_v5.py"), os.path.join(os.path.dirname(HERE), "make_population_v5.py")] if os.path.exists(f))   # a cote du script, sinon dans campagne_v5/
spec = importlib.util.spec_from_file_location("mkpop", _MK); mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
POPS = {"seed8": dict(seed=8), "seed9": dict(seed=9), "rho05": dict(seed=7, rho_cp_ap=0.5), "tau_wide": dict(seed=7, tau_lo=12.0, tau_hi=150.0), "eta_low": dict(seed=7, eta_mu=0.21)}
N = 20; NFIT = 10   # information de Fisher sur 20 athletes, ajustement LM sur les 10 premiers
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def wp(t): return t[2] * t[4]

if "--analyse" in sys.argv:
    from scipy.stats import beta
    def cpi(k, n): return (0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(0.975, k + 1, n - k))
    V3 = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "camp_v3_*.jsonl"))) for l in open(f)]
    ref = np.array([r["crlb_pct"] for r in V3 if r["sigma_P"] == 5.0]); print("campagne (50 athletes) bornes medianes", np.round(np.median(ref, 0), 3))
    for name in POPS:
        R = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "rev3_pop_%s_*.jsonl" % name))) for l in open(f)]
        if not R: continue
        c = np.array([r["crlb"] for r in R]); q = lambda k: np.array([r[k] for r in R])
        print("\n%s : %d athletes ; bornes medianes %s ; CP %.3f W' %.3f flux %.3f ; expo %.2f [%.2f, %.2f] ; cond %.3g ; proj CP %.3g"
              % (name, len(R), np.round(np.median(c, 0), 3), np.median(q("crlb_CP")), np.median(q("crlb_Wp")), np.median(q("crlb_flux")),
                 np.median(q("expo")), *np.percentile(q("expo"), [25, 75]), np.median(q("cond")), np.median(q("proj_CP"))))
        F = [r for r in R if r.get("est")]
        if F:
            th = np.array([r["theta_true"] for r in F]); x = np.array([r["est"] for r in F]); se = np.array([r["LM_se"] for r in F]); cr = np.array([r["crlb"] for r in F])
            err = 100 * np.abs(x / th - 1); n = len(F); k1 = np.sum(err < 1.96 * cr, 0); k2 = np.sum(np.abs(x - th) <= 1.96 * se, 0)
            print("  LM (%d ajustes, %d hors bornes sur %d) : erreur med par param %s ; CP %.2f W' %.2f ; couverture +-1.96 borne %s ; IC jacobien %s"
                  % (n, sum(not r.get("in_bounds", True) for r in R), len(R), np.round(np.median(err, 0), 2), np.median([100 * abs(cp(a) / cp(b) - 1) for a, b in zip(x, th)]),
                     np.median([100 * abs(wp(a) / wp(b) - 1) for a, b in zip(x, th)]),
                     ["%d/%d [%.0f-%.0f]" % (kk, n, 100 * cpi(kk, n)[0], 100 * cpi(kk, n)[1]) for kk in k1], ["%d/%d" % (kk, n) for kk in k2]))
    sys.exit()

ap = argparse.ArgumentParser(); ap.add_argument("pop", choices=list(POPS)); ap.add_argument("shard", nargs="?", type=int, default=0)
ap.add_argument("nshards", nargs="?", type=int, default=1); ap.add_argument("--fit", action="store_true"); a = ap.parse_args()
TH = mk.recalibrate(N, **POPS[a.pop]); cfg = p3.Config(budget_ode=BUDGET)
OUT = os.path.join(HERE, "rev3_pop_%s_%d.jsonl" % (a.pop, a.shard))
done = {json.loads(l)["i"] for l in open(OUT)} if os.path.exists(OUT) else set()
t0 = time.time()
for i in [i for i in range(N) if i % a.nshards == a.shard and i not in done]:
    th = TH[i]; seed = 1000 * i + 50
    data = p3.gen_data(th, 5.0, seed, cfg)
    F = sens_matrix(th, data); F = F.T @ F; s = fim_summary(F, th); Fi = np.linalg.inv(F)
    gfl = np.array([0., 1., 0., 1., 0.])
    rec = dict(i=i, pop=a.pop, theta_true=th.tolist(), CP=cp(th), Wp=wp(th), sigma_AP={k: v["sigma_AP"] for k, v in data["protocols"].items()},
               crlb=s["crlb"], crlb_CP=s["crlb_CP"], crlb_Wp=s["crlb_Wp"], crlb_flux=float(100 * np.sqrt(gfl @ Fi @ gfl)),
               expo=s["expo"], cond=s["cond"], vmin=s["vmin"], proj_CP=s["proj_CP"], proj_Wp=s["proj_Wp"])
    inb = bool(np.all(th >= p3.PARAM_LB) and np.all(th <= p3.PARAM_UB)); rec["in_bounds"] = inb
    if a.fit and inb and i < NFIT:
        p3.set_tol(1e-8); x, se, n = p3.fit_LM_with_CI(data, cfg, seed)
        rec.update(est=np.asarray(x, float).tolist(), LM_se=np.asarray(se, float).tolist(), ode_calls=int(n))
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print("%s i=%2d bornes %s CP %.3f flux %.3f expo %.2f %s (%.0f s)" % (a.pop, i, np.round(s["crlb"], 2), s["crlb_CP"], rec["crlb_flux"], s["expo"],
          ("LM CP %.2f%%" % (100 * abs(cp(rec["est"]) / cp(th) - 1))) if "est" in rec else ("hors bornes" if not inb else ""), time.time() - t0), flush=True)
print("TERMINE pop", a.pop, a.shard)
