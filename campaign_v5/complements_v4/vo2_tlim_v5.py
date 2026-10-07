#!/usr/bin/env python3
"""Identifiabilite pratique sous les observables realisables : VO2 (flux delivre phi1), temps d'epuisement T_lim
(derivee implicite, Eq. dTlim du manuscrit), et leur paire, contre A_P idealise. Regeneration (v5) du script
vo2_tlim.py perdu ; memes conventions que vo2_tlim.md : 10 athletes de la population recalibree, protocoles A et B,
60 points d'observation par essai (grille de la campagne), DOP853 a rtol = atol = 1e-10, sensibilites relatives (theta_k dy/dtheta_k)
par differences finies centrees (pas relatif 1e-4) a temps fixe sans detection d'evenement, profil de puissance
fige au profil nominal. Blocs normalises : VO2 par (c_V . moyenne(phi1))^2, T_lim par (c_T . T_lim)^2, r = c_T/c_V ;
A_P par sigma_AP^2 (calibration v5, sigma_P = 5 W, lue dans camp_v3_*.jsonl).
Bornes absolues : sigma_V = 100 et 200 mL/min a 20,9 J/mL (34,8 et 69,7 W), cv_T = 3 et 10 %.
Sorties : vo2_tlim_v5.json, vo2_tlim_v5.log (stdout)."""
import json, glob, os, sys, numpy as np
from scipy.integrate import solve_ivp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); import pipeline_v3 as p3
N_ATH, N_OBS, TOL, H = 10, 60, 1e-10, 1e-4
NAMES = ["M_O", "A_Omax", "A_Pmax", "M_R", "eta"]
JML = 20.9  # J par mL d'O2

def rhs(t, y, th, P):
    M_O, A_Om, A_Pm, M_R, eta = th; A_O, A_P = y
    phi1 = M_O * (A_O / A_Om) * (1.0 - A_P / A_Pm)
    return [M_R * (A_Om - A_O) - phi1, phi1 - P(t) / eta]
def ev(t, y, th, P): return y[1]
ev.terminal = True; ev.direction = -1

def integrate(th, P, t_max, switches, grid=None, event=True):
    """DOP853 segment par segment ; renvoie (t_end, A_O(grid), A_P(grid))."""
    bnds = [0.0] + [s for s in switches if 0.0 < s < t_max] + [t_max]
    state = [th[1], th[2]]; sols = []; t_end = t_max
    for k in range(len(bnds) - 1):
        sol = solve_ivp(rhs, (bnds[k], bnds[k + 1]), state, args=(th, P), method="DOP853", rtol=TOL, atol=TOL,
                        events=ev if event else None, dense_output=True)
        sols.append(sol)
        if event and sol.t_events[0].size: t_end = float(sol.t_events[0][0]); break
        state = [sol.y[0][-1], sol.y[1][-1]]; t_end = sol.t[-1]
    if grid is None: return t_end, None, None
    AO = np.empty(len(grid)); AP = np.empty(len(grid))
    for j, x in enumerate(grid):
        for sol in sols:
            if sol.t[0] - 1e-12 <= x <= sol.t[-1] + 1e-12: AO[j], AP[j] = sol.sol(x); break
        else: AO[j], AP[j] = sols[-1].sol(sols[-1].t[-1])
    return t_end, AO, AP

def phi1_of(th, AO, AP): return th[0] * (AO / th[1]) * (1.0 - AP / th[2])
def lncp_grad(th):
    cp = lambda t: np.log(t[0] * (t[3] * t[1] / (t[3] * t[1] + t[0])) * t[4]); g = np.zeros(5)
    for k in range(5):
        tp, tm = th.copy(), th.copy(); tp[k] *= 1 + H; tm[k] *= 1 - H; g[k] = (cp(tp) - cp(tm)) / (2 * H)
    return g
GRAD_WP = np.array([0.0, 0.0, 1.0, 0.0, 1.0])     # d ln W' / d ln theta

def sens_athlete(th, sg):
    """Pour A et B : grilles, sensibilites relatives de A_P et phi1 (temps fixe, sans evenement), de T_lim (implicite),
    validation de T_lim par differences finies (evenement actif)."""
    out = {}
    for kind in ("A", "B"):
        P, t_max, sw = p3.power_profile(kind, th)            # profil nominal, fige
        T, _, _ = integrate(th, P, t_max, sw)
        grid = np.linspace(0.0, 0.999 * T, N_OBS)
        _, AO, AP = integrate(th, P, t_max, sw, grid, event=False)
        # sensibilites a temps fixe, a T_lim nominal inclus (sans evenement)
        g2 = np.append(grid, T); S_AP = np.zeros((N_OBS, 5)); S_V = np.zeros((N_OBS, 5)); S_APT = np.zeros(5)
        for k in range(5):
            tp, tm = th.copy(), th.copy(); tp[k] *= 1 + H; tm[k] *= 1 - H
            _, AOp, APp = integrate(tp, P, t_max, sw, g2, event=False); _, AOm, APm = integrate(tm, P, t_max, sw, g2, event=False)
            S_AP[:, k] = (APp[:-1] - APm[:-1]) / (2 * H); S_V[:, k] = (phi1_of(tp, AOp[:-1], APp[:-1]) - phi1_of(tm, AOm[:-1], APm[:-1])) / (2 * H)
            S_APT[k] = (APp[-1] - APm[-1]) / (2 * H)
        # derivee implicite de T_lim
        _, AOT, APT = integrate(th, P, t_max, sw, np.array([T]), event=False)
        dAP = th[0] * (AOT[0] / th[1]) - P(T) / th[4]          # Adot_P(T) avec A_P(T) = 0
        S_T = -S_APT / dAP                                     # theta_k dT/dtheta_k
        # validation par differences finies sur T_lim (evenement actif)
        S_T_fd = np.zeros(5)
        for k in range(5):
            tp, tm = th.copy(), th.copy(); tp[k] *= 1 + H; tm[k] *= 1 - H
            S_T_fd[k] = (integrate(tp, P, t_max, sw)[0] - integrate(tm, P, t_max, sw)[0]) / (2 * H)
        phi = phi1_of(th, AO, AP)
        out[kind] = dict(T=T, S_AP=S_AP, S_V=S_V, S_T=S_T, S_T_fd=S_T_fd, phi_mean=float(np.mean(phi)), sigma_AP=sg[kind], dAP=dAP)
    return out

def fim(S, var): return S.T @ S / var
def analyse(F, th):
    w, V = np.linalg.eigh(F); w = np.clip(w, 0, None); lam = w / w[-1]
    v = V[:, 0]; g = lncp_grad(th)
    rank = int(np.sum(lam > 1e-10))
    return dict(rank=rank, lam_ratio=float(lam[0]), proj_CP=float(abs(g @ v) / np.linalg.norm(g)), proj_Wp=float(abs(GRAD_WP @ v) / np.linalg.norm(GRAD_WP)), vmin=(v / v[np.argmax(np.abs(v))]).tolist())
def crlb(F, th):
    C = np.linalg.pinv(F); g = lncp_grad(th)
    return dict(par=(100 * np.sqrt(np.clip(np.diag(C), 0, None))).tolist(), CP=float(100 * np.sqrt(max(g @ C @ g, 0))), Wp=float(100 * np.sqrt(max(GRAD_WP @ C @ GRAD_WP, 0))))

if __name__ == "__main__":
    V3 = {}
    for f in sorted(glob.glob(os.path.join(HERE, "..", "camp_v3_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if r["sigma_P"] == 5.0: V3[r["i"]] = r
    res = {"athletes": [], "valid_Tlim_max_rel_err": 0.0}
    R_LIST = [0.2, 1.0, 5.0]; SV = {"100 mL/min": 100 * JML / 60, "200 mL/min": 200 * JML / 60}; CVT = {"3 %": 0.03, "10 %": 0.10}
    for i in range(N_ATH):
        th = np.array(V3[i]["theta_true"]); sg = V3[i]["sigma_AP"]; S = sens_athlete(th, sg)
        errs = {k: float(np.max(np.abs(S[k]["S_T"] - S[k]["S_T_fd"]) / np.abs(S[k]["S_T_fd"]))) for k in ("A", "B")}
        at_switch = {k: bool(abs(S[k]["T"] / 30.0 - round(S[k]["T"] / 30.0)) < 1e-3) for k in ("A", "B")}   # epuisement a une commutation (B)
        err = max(errs[k] for k in errs if not at_switch[k])
        res["valid_Tlim_max_rel_err"] = max(res["valid_Tlim_max_rel_err"], err)
        res.setdefault("Tlim_at_switch", []).extend([(i, k, S[k]["T"], errs[k]) for k in errs if at_switch[k]])
        F_AP = sum(fim(S[k]["S_AP"], S[k]["sigma_AP"] ** 2) for k in "AB")
        F_V1 = sum(fim(S[k]["S_V"], S[k]["phi_mean"] ** 2) for k in "AB")            # c_V = 1
        F_T1 = sum(np.outer(S[k]["S_T"], S[k]["S_T"]) / S[k]["T"] ** 2 for k in "AB")  # c_T = 1
        a = dict(i=i, T=[S["A"]["T"], S["B"]["T"]], valid=err)
        a["norm"] = {"A_P": analyse(F_AP, th), "VO2": analyse(F_V1, th), "Tlim": analyse(F_T1, th)}
        for r in R_LIST: a["norm"]["VO2+Tlim r=%g" % r] = analyse(F_V1 + F_T1 / r ** 2, th)
        a["abs"] = {"A_P (sigma_AP v5, 60 pts)": crlb(F_AP, th)}
        for sv_name, sv in SV.items():
            F_V = sum(fim(S[k]["S_V"], sv ** 2) for k in "AB"); a["abs"]["VO2 " + sv_name] = crlb(F_V, th)
            for ct_name, ct in CVT.items():
                F_T = sum(np.outer(S[k]["S_T"], S[k]["S_T"]) / (ct * S[k]["T"]) ** 2 for k in "AB")
                a["abs"]["VO2+Tlim %s, cv_T %s" % (sv_name, ct_name)] = crlb(F_V + F_T, th)
        res["athletes"].append(a); print("athlete %d : T_A %.0f s, T_B %.0f s, validation T_lim %.2e" % (i, a["T"][0], a["T"][1], err), flush=True)
    json.dump(res, open(os.path.join(HERE, "vo2_tlim_v5.json"), "w"), indent=1)
    print("\nvalidation de la derivee implicite de T_lim : erreur relative max %.2e (hors essais ou l'epuisement tombe sur une commutation : %s)" % (res["valid_Tlim_max_rel_err"], res.get("Tlim_at_switch")))
    print("\n== Blocs normalises, mediane sur %d athletes ==" % N_ATH)
    print("%-22s rang  lam_min/lam_max   proj CP   proj W'   v_min (max comp = 1)" % "observable")
    for key in res["athletes"][0]["norm"]:
        A = [a["norm"][key] for a in res["athletes"]]
        print("%-22s %4s  %9.2e    %6.3f    %6.3f   %s" % (key, int(np.median([x["rank"] for x in A])), np.median([x["lam_ratio"] for x in A]),
              np.median([x["proj_CP"] for x in A]), np.median([x["proj_Wp"] for x in A]), np.round(np.median([x["vmin"] for x in A], axis=0), 2).tolist()))
    print("\n== Bornes de Cramer-Rao absolues (ecart-type relatif minimal, %%), mediane sur %d athletes ==" % N_ATH)
    print("%-40s %s    CP     W'" % ("observable", "  ".join("%7s" % n for n in NAMES)))
    for key in res["athletes"][0]["abs"]:
        A = [a["abs"][key] for a in res["athletes"]]
        print("%-40s %s  %6.2f %6.2f" % (key, "  ".join("%7.2f" % v for v in np.median([x["par"] for x in A], axis=0)), np.median([x["CP"] for x in A]), np.median([x["Wp"] for x in A])))
    print("TERMINE vo2_tlim_v5")
