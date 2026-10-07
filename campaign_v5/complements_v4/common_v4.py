"""Outils communs aux complements v4 : chargement de pipeline_v3/v4, donnees
identiques a la campagne (sigma_AP relu dans camp_v3_*.jsonl)."""
import os, sys, glob, json, importlib.util, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
p3 = _load("p3", "pipeline_v3.py")
BUDGET = 4000
def load_v3():
    V3 = {}
    files = sorted(glob.glob(os.path.join(HERE, "camp_v3_*.jsonl"))) or sorted(glob.glob(os.path.join(HERE, "..", "camp_v3_*.jsonl")))
    for f in files:
        for l in open(f):
            r = json.loads(l); V3[(r["i"], r["sigma_P"])] = r
    if not V3: sys.exit("camp_v3_*.jsonl introuvables dans " + HERE)
    return V3
def data_like_campaign(V3, i, s, seed=None):
    """Regenere le jeu de donnees de la campagne v3 (meme graine, meme sigma_AP)."""
    r3 = V3[(i, s)]; th = np.array(r3["theta_true"]); sg = r3["sigma_AP"]
    p3.calibrate_sigma_AP = lambda t_, kind, s_, n_mc=100, seed=0, _s=sg: _s[kind]
    seed = 1000 * i + int(s * 10) if seed is None else seed
    return th, p3.gen_data(th, s, seed, p3.Config(budget_ode=BUDGET)), sg
def sens_matrix(theta, data, rel_step=1e-4, tol=1e-10):
    """Sensibilites relatives S_ik = theta_k dy_i/dtheta_k / sigma, empilees sur
    les protocoles, au couple (rtol, pas) verifie convergent (v3)."""
    keep = p3._TOL[0]; p3.set_tol(tol)
    try:
        blocks = []
        for kind, d in data["protocols"].items():
            grid = np.asarray(d["t"]); sA = d["sigma_AP"]
            blk = np.zeros((len(grid), 5))
            for k in range(5):
                tp, tm = theta.copy(), theta.copy(); h = rel_step * abs(theta[k]); tp[k] += h; tm[k] -= h
                yp, _ = p3.simulate(tp, d["_P"], d["_t_max"], grid, switches=d["_sw"])
                ym, _ = p3.simulate(tm, d["_P"], d["_t_max"], grid, switches=d["_sw"])
                blk[:, k] = (yp - ym) / (2 * h) * theta[k] / sA
            blocks.append(blk)
        return np.vstack(blocks)
    finally:
        p3.set_tol(keep)
def fim_summary(F, theta):
    w, V = np.linalg.eigh(F); o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    v = V[:, -1] / np.abs(V[:, -1]).max()
    if v[3] < 0: v = -v
    M_O, A_Om, A_Pm, M_R, eta = theta; xss = M_R * A_Om / (M_R * A_Om + M_O)
    gcp = np.array([xss, 1 - xss, 0., 1 - xss, 1.]); gwp = np.array([0., 0., 1., 0., 1.])   # grad ln CP, ln W' sur ln theta
    u = V[:, -1]
    Fi = np.linalg.inv(F)
    return dict(spec=(w / w[0]).tolist(), cond=float(w[0] / w[-1]), vmin=v.tolist(),
                expo=float(v[3] / v[1]) if abs(v[1]) > 1e-12 else float("nan"),
                crlb=(100 * np.sqrt(np.diag(Fi))).tolist(),
                crlb_CP=float(100 * np.sqrt(gcp @ Fi @ gcp)), crlb_Wp=float(100 * np.sqrt(gwp @ Fi @ gwp)),
                proj_CP=float(abs(gcp @ u) / np.linalg.norm(gcp)), proj_Wp=float(abs(gwp @ u) / np.linalg.norm(gwp)))
