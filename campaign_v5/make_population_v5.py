#!/usr/bin/env python3
"""Population virtuelle du manuscrit (Table 1) : regenere pop_recal.npy.

CP ~ N(300, 45) W tronquee a [216, 402] ; x_ss ~ U[0.78, 0.88] ; eta ~ N(0.25, 0.02) ;
M_O = CP / (x_ss eta), accepte si M_O / 20.9 J/mL est dans 3.0-6.5 L/min ; 1/M_R ~ U[18, 91] s ;
A_Omax = x_ss / (1 - x_ss) * M_O / M_R ; A_Pmax ~ N(80, 18) kJ, > 30 kJ. Colonnes : M_O, A_Omax, A_Pmax, M_R, eta.
Usage : python3 make_population_v5.py [--seed 7] [--n 50] [--out pop_recal.npy] [--check]"""
import argparse, os, numpy as np
JO2 = 20.9
def W_to_Lmin(w): return w / JO2 * 60 / 1000.0
def recalibrate(n=50, seed=7, cp_mu=300.0, cp_sd=45.0, cp_lo=216.0, cp_hi=402.0, xss_lo=0.78, xss_hi=0.88,
                tau_lo=18.0, tau_hi=91.0, rho_cp_ap=0.0, eta_mu=0.25, eta_sd=0.02):
    r = np.random.default_rng(seed); out = []
    while len(out) < n:
        cp = r.normal(cp_mu, cp_sd)
        if not cp_lo <= cp <= cp_hi: continue
        xss = r.uniform(xss_lo, xss_hi); eta = r.normal(eta_mu, eta_sd)
        MO = cp / (xss * eta)
        if not 3.0 <= W_to_Lmin(MO) <= 6.5: continue
        MR = 1.0 / r.uniform(tau_lo, tau_hi)
        AOm = (xss / (1 - xss)) * MO / MR
        z = r.normal()
        if rho_cp_ap:   # option : correlation entre CP et A_Pmax (analyse de robustesse)
            z = rho_cp_ap * (cp - cp_mu) / cp_sd + np.sqrt(1 - rho_cp_ap ** 2) * z
        APm = 80000.0 + 18000.0 * z
        if APm <= 30000: continue
        out.append([MO, AOm, APm, MR, eta])
    return np.array(out)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="pop_recal.npy"); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    P = recalibrate(a.n, a.seed)
    if a.check:
        ref = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pop_recal.npy"))
        print("ecart relatif max avec pop_recal.npy :", float(np.max(np.abs(P / ref[:a.n] - 1))))
    else:
        np.save(a.out, P); print("ecrit", a.out, P.shape)
