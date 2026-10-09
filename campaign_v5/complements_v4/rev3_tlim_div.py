#!/usr/bin/env python3
"""Rev.3 (R1 M3, R2 R4) : equilibres a puissance constante et divergence du temps d'epuisement pres de la CP.
A puissance constante P < CP, le systeme admet l'equilibre A_O* = A_Om (1 - P / (eta M_R A_Om)),
A_P* = A_Pm [1 - P / (eta M_O x(P))], x(P) = A_O* / A_Om, avec A_P* > 0 ; A_P* = 0 exactement a P = CP.
Jacobien J = [[-M_R - a, b], [a, -b]], a = (M_O/A_Om)(1 - A_P/A_Pm), b = M_O A_O / (A_Om A_Pm) :
trace < 0, det = M_R b > 0, discriminant >= 0, donc noeud stable (valeurs propres reelles negatives) pour tout A_O > 0.
Verifications numeriques par athlete (50, theta vrais de la campagne) :
 (i) valeurs propres a P/CP = 0.5, 0.8, 0.9, 0.95, 0.99, 1.0 ; convergence d'une simulation depuis le repos vers A_P* (horizon 2e5 s) ;
 (ii) T_lim a P = (1 + eps) CP, eps de 1e-4 a 0.5 (horizon 1e7 s) ; pente de T_lim contre ln(1/eps) sur eps <= 1e-2,
      comparee a 1/|lambda_lent| a l'equilibre P = CP (prediction : divergence logarithmique, non hyperbolique).
Usage : python3 rev3_tlim_div.py [shard nshards] ; synthese : --analyse"""
import json, os, sys, glob, time, numpy as np
from common_v4 import p3, load_v3, HERE
EPS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.25, 0.5]; FR = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
def cp(t): return t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]
def equil(th, P):
    M_O, A_Om, A_Pm, M_R, eta = th; AO = A_Om - P / (eta * M_R); x = AO / A_Om; AP = A_Pm * (1 - P / (eta * M_O * x)); return AO, AP
def jac(th, AO, AP):
    M_O, A_Om, A_Pm, M_R, eta = th; a = (M_O / A_Om) * (1 - AP / A_Pm); b = M_O * AO / (A_Om * A_Pm)
    return np.array([[-M_R - a, b], [a, -b]])
if "--analyse" in sys.argv:
    R = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "rev3_tlim_div_*.jsonl"))) for l in open(f)]; print("athletes", len(R))
    mx = np.array([r["max_re_eig"] for r in R]); print("max partie reelle des valeurs propres (toutes intensites <= CP) : max sur athletes %.3g ; valeurs complexes : %d" % (mx.max(), sum(r["complex"] for r in R)))
    print("ecart relatif max a l'equilibre apres 2e5 s : %.2g ; epuisement sous CP : %d" % (max(r["conv_err"] for r in R), sum(r["exhausted_below_CP"] for r in R)))
    ls = np.array([r["lam_slow"] for r in R]); lf = np.array([r["lam_fast"] for r in R])
    print("lambda lent a CP : mediane %.4g s^-1 (1/|l| = %.0f s, IQR %.0f-%.0f) ; lambda rapide %.4g" % (np.median(ls), np.median(-1 / ls), *np.percentile(-1 / ls, [25, 75]), np.median(lf)))
    q = np.array([r["slope_x_lam"] for r in R]); print("pente(T, ln 1/eps) x |lambda_lent| : mediane %.3f [%.3f, %.3f] min %.3f max %.3f" % (np.median(q), *np.percentile(q, [25, 75]), q.min(), q.max()))
    T = np.array([r["T_lim"] for r in R]); H = np.array([r["T_hyp"] for r in R])
    for k, e in enumerate(EPS): print("eps %-7g T_lim mediane %9.0f s [%9.0f-%9.0f] ; hyperbole W'/(eps CP) %10.0f s ; rapport %.3g" % (e, np.median(T[:, k]), *np.percentile(T[:, k], [25, 75]), np.median(H[:, k]), np.median(T[:, k] / H[:, k])))
    sys.exit()
sh, ns = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
V3 = load_v3(); OUT = os.path.join(HERE, "rev3_tlim_div_%d.jsonl" % sh)
done = {json.loads(l)["i"] for l in open(OUT)} if os.path.exists(OUT) else set()
p3.set_tol(1e-10); t0 = time.time()
for i in [i for i in range(50) if i % ns == sh and i not in done]:
    th = np.array(V3[(i, 5.0)]["theta_true"]); c = cp(th); Wp = th[2] * th[4]
    eigs, cplx, conv, exh = {}, 0, 0.0, 0
    for f in FR:
        AO, AP = equil(th, f * c); w = np.linalg.eigvals(jac(th, AO, AP)); cplx += int(np.any(np.abs(w.imag) > 0)); eigs[str(f)] = sorted(w.real.tolist())
        if f < 1.0:
            y, te = p3.simulate(th, (lambda t, p=f * c: p), 2.0e5, [2.0e5 - 1.0]); exh += int(te < 2.0e5 - 1.0)
            conv = max(conv, abs(float(y[0]) - AP) / th[2])
    ls, lf = eigs["1.0"][1], eigs["1.0"][0]
    T = []
    for e in EPS:
        _, te = p3.simulate(th, (lambda t, p=(1 + e) * c: p), 1.0e7); T.append(float(te))
    m = np.array(EPS) <= 1e-2; sl = np.polyfit(np.log(1 / np.array(EPS)[m]), np.array(T)[m], 1)[0]
    rec = dict(i=i, CP=c, Wp=Wp, eig=eigs, max_re_eig=max(max(v) for v in eigs.values()), complex=cplx, conv_err=conv, exhausted_below_CP=exh,
               lam_slow=ls, lam_fast=lf, T_lim=T, T_hyp=[Wp / (e * c) for e in EPS], slope=float(sl), slope_x_lam=float(sl * abs(ls)))
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print("i=%2d lam_lent %.4g  pente x |lam| %.3f  T(1e-4) %.0f s vs hyperbole %.0f s (%.0f s)" % (i, ls, rec["slope_x_lam"], T[0], rec["T_hyp"][0], time.time() - t0), flush=True)
